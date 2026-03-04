from flask import Flask, request
import os, json, re, tempfile, io
import requests
from datetime import datetime, timedelta
from PIL import Image
from google.cloud import vision

# =========================
# ENV
# =========================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
GCP_SA_JSON = os.getenv("GCP_SA_JSON")
CRON_SECRET = os.getenv("CRON_SECRET", "")

app = Flask(__name__)
MEALS_FILE = "meals.json"

DAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
MEAL_NAME = {"breakfast": "아침", "lunch": "점심", "dinner": "저녁"}

# =========================
# (중요) 템플릿 고정 기반 "큰 영역"만 비율로 잡음
# - 경계는 점심에서만 OCR로 자동 검출
# =========================
CROP = {
    # 전체 이미지에서 표 영역 (대략)
    "table_left": 0.05,
    "table_top": 0.10,
    "table_right": 0.985,
    "table_bottom": 0.93,

    # 표 내부에서 요일 영역 (왼쪽 '구분' 컬럼 제외)
    "days_left": 0.12,
    "days_right": 0.995,

    # 관리직 기준: 1조/관리 조식
    "breakfast_top": 0.08,
    "breakfast_bottom": 0.28,

    # 관리직 기준: 1조 중식/관리 중식 (점심 전체 블록)
    "lunch_top": 0.28,
    "lunch_bottom": 0.73,

    # 관리직 기준: 2조 조식/1조 석식 (저녁 블록)
    "dinner_top": 0.73,
    "dinner_bottom": 0.93,
}

# =========================
# 필터: 공통/SELF/PLUS/오늘의차 등 제거
# =========================
NOISE_EXACT = {"한식", "일품", "공통", "SELF", "PLUS", "오늘의차"}
NOISE_CONTAINS = [
    "공통", "SELF", "PLUS", "오늘의차",
    "라면", "코너", "누룽지", "도시락김", "샐러드",
    "STEMS", "aramark", "구분",
    "조식", "중식", "석식", "1조", "2조", "관리"
]

# =========================
# Time helpers (KST)
# =========================
def kst_now() -> datetime:
    return datetime.utcnow() + timedelta(hours=9)

def day_key(offset_days: int = 0) -> str:
    d = (kst_now().date() + timedelta(days=offset_days))
    return DAY_KEYS[d.weekday()]

# =========================
# Telegram helpers
# =========================
def tg_send(text: str, keyboard: dict | None = None):
    data = {"chat_id": CHANNEL_ID, "text": text}
    if keyboard is not None:
        data["reply_markup"] = json.dumps(keyboard, ensure_ascii=False)
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data=data, timeout=20)
    r.raise_for_status()

def tg_edit(chat_id, message_id, text: str, keyboard: dict | None = None):
    data = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if keyboard is not None:
        data["reply_markup"] = json.dumps(keyboard, ensure_ascii=False)
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/editMessageText", data=data, timeout=20)
    r.raise_for_status()

def tg_answer_callback(callback_query_id: str):
    requests.post(
        f"https://api.telegram.org/bot{TOKEN}/answerCallbackQuery",
        data={"callback_query_id": callback_query_id},
        timeout=10,
    )

def download_telegram_photo(file_id: str) -> bytes:
    # 1) file_path 얻기
    r = requests.get(
        f"https://api.telegram.org/bot{TOKEN}/getFile",
        params={"file_id": file_id},
        timeout=20,
    )
    r.raise_for_status()
    file_path = r.json()["result"]["file_path"]

    # 2) 이미지 다운로드
    img = requests.get(
        f"https://api.telegram.org/file/bot{TOKEN}/{file_path}",
        timeout=30,
    )
    img.raise_for_status()
    return img.content

# =========================
# Vision client / OCR
# =========================
def vision_client() -> vision.ImageAnnotatorClient:
    if not GCP_SA_JSON:
        raise RuntimeError("GCP_SA_JSON env var is missing")

    sa = json.loads(GCP_SA_JSON)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sa, f)
        sa_path = f.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
    return vision.ImageAnnotatorClient()

def ocr_document(client: vision.ImageAnnotatorClient, image_bytes: bytes):
    """document_text_detection 결과 전체를 반환(텍스트 + 좌표 사용 위해)."""
    image = vision.Image(content=image_bytes)
    resp = client.document_text_detection(image=image)
    if resp.error and resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp

# =========================
# Image helpers
# =========================
def crop_by_ratio(img: Image.Image, l: float, t: float, r: float, b: float) -> Image.Image:
    w, h = img.size
    return img.crop((int(w*l), int(h*t), int(w*r), int(h*b)))

def to_png_bytes(img: Image.Image, scale: int = 2) -> bytes:
    # OCR 정확도 위해 확대
    img2 = img.resize((img.size[0]*scale, img.size[1]*scale))
    buf = io.BytesIO()
    img2.save(buf, format="PNG")
    return buf.getvalue()

def clean_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if len(s) <= 1:
            continue
        if re.search(r"\d{1,2}:\d{2}", s):  # 시간 제거
            continue
        if s in NOISE_EXACT:
            continue
        bad = False
        for tok in NOISE_CONTAINS:
            if tok in s:
                bad = True
                break
        if bad:
            continue
        s = s.replace("•", "").replace("·", " ").replace("|", " ").strip()
        lines.append(s)

    # 중복 제거(순서 유지)
    seen = set()
    uniq = []
    for x in lines:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq

# =========================
# (핵심) 점심 한식/일품 경계 자동 검출
# -------------------------
# lunch_block에서 OCR 결과의 bounding box로 "한식", "일품" 단어의 y좌표를 찾고,
# 한식 메뉴 영역 = 한식 라벨 아래 ~ 일품 라벨 위
# 일품 메뉴 영역 = 일품 라벨 아래 ~ lunch_block 끝
# 라벨이 인식 안 되면 fallback(반반 분할) 사용
# =========================
def _iter_words_with_boxes(resp):
    """
    Vision full_text_annotation에서 (word_text, bbox) iterate.
    bbox: [(x,y), ... 4 points] 형태
    """
    if not resp.full_text_annotation or not resp.full_text_annotation.pages:
        return
    for page in resp.full_text_annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    wtxt = "".join([s.text for s in word.symbols]).strip()
                    if not wtxt:
                        continue
                    bbox = word.bounding_box.vertices  # 4 vertices
                    yield wtxt, bbox

def _bbox_bottom_y(bbox) -> int:
    return max(v.y for v in bbox if v.y is not None)

def _bbox_top_y(bbox) -> int:
    return min(v.y for v in bbox if v.y is not None)

def find_label_y(resp, label: str) -> tuple[int | None, int | None]:
    """
    label 단어(예: '한식', '일품')의 (top_y, bottom_y) 반환.
    여러개면 가장 위쪽(최소 top_y) 사용.
    """
    candidates = []
    for wtxt, bbox in _iter_words_with_boxes(resp):
        if wtxt == label:
            candidates.append(( _bbox_top_y(bbox), _bbox_bottom_y(bbox) ))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    return candidates[0]

def split_lunch_by_labels(lunch_img: Image.Image, client: vision.ImageAnnotatorClient):
    """
    lunch_img(요일 1칸의 점심 블록)에서 한식/일품 메뉴 영역을 자동 분리.
    반환: (han_menu_img, ilpum_menu_img)
    """
    # 1) lunch 블록 OCR(좌표 얻기 위해)
    resp = ocr_document(client, to_png_bytes(lunch_img, scale=2))

    # 2) 라벨 위치 찾기
    han_top, han_bottom = find_label_y(resp, "한식")
    il_top, il_bottom = find_label_y(resp, "일품")

    W, H = lunch_img.size

    # 3) fallback: 라벨이 하나라도 못 잡히면 반반 분할
    if han_bottom is None or il_top is None or il_bottom is None:
        mid = int(H * 0.52)  # 경험적
        han_menu = lunch_img.crop((0, 0, W, mid))
        il_menu = lunch_img.crop((0, mid, W, H))
        return han_menu, il_menu

    # Vision OCR은 우리가 scale=2로 넣었으니 y도 2배 스케일임 → 원본 lunch_img 좌표로 환산
    # (scale=2)
    scale = 2.0
    han_menu_top = int((han_bottom + 4) / scale)  # 라벨 아래로 약간 내려가기
    il_label_top = int((il_top - 2) / scale)
    il_menu_top = int((il_bottom + 4) / scale)

    # 4) clamp & sanity
    han_menu_top = max(0, min(H, han_menu_top))
    il_label_top = max(0, min(H, il_label_top))
    il_menu_top = max(0, min(H, il_menu_top))

    # 한식 메뉴 영역: han_menu_top ~ il_label_top
    if il_label_top <= han_menu_top + 10:
        # 너무 가까우면 fallback
        mid = int(H * 0.52)
        han_menu = lunch_img.crop((0, 0, W, mid))
        il_menu = lunch_img.crop((0, mid, W, H))
        return han_menu, il_menu

    han_menu = lunch_img.crop((0, han_menu_top, W, il_label_top))
    il_menu = lunch_img.crop((0, il_menu_top, W, H))
    return han_menu, il_menu

# =========================
# Date range (옵션)
# =========================
def extract_date_range(text: str) -> str | None:
    m = re.findall(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일\s*(월|화|수|목|금|토|일)\s*요일", text)
    if len(m) >= 2:
        s = m[0]; e = m[-1]
        return f"{s[0]}/{s[1]}({s[2]}) ~ {e[0]}/{e[1]}({e[2]})"
    d = re.findall(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", text)
    if len(d) >= 2:
        s = d[0]; e = d[-1]
        return f"{s[0]}/{s[1]} ~ {e[0]}/{e[1]}"
    return None

# =========================
# Parse weekly (관리직 규칙)
# - 아침: 한식만
# - 점심: 한식/일품 라벨 기반 자동 분리
# - 저녁: 한식만
# =========================
def parse_weekly(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    table = crop_by_ratio(img, CROP["table_left"], CROP["table_top"], CROP["table_right"], CROP["table_bottom"])
    days_area = crop_by_ratio(table, CROP["days_left"], 0.0, CROP["days_right"], 1.0)

    W, H = days_area.size
    col_w = W / 7.0

    client = vision_client()

    out = {
        dk: {
            "breakfast": {"han": []},
            "lunch": {"han": [], "ilpum": []},
            "dinner": {"han": []},
        } for dk in DAY_KEYS
    }

    # 날짜 범위는 전체 OCR 한 번으로
    full_resp = ocr_document(client, to_png_bytes(img, scale=2))
    full_text = full_resp.full_text_annotation.text if full_resp.full_text_annotation else ""
    date_range = extract_date_range(full_text) or "날짜 인식 실패"

    bf_y0 = int(H * CROP["breakfast_top"])
    bf_y1 = int(H * CROP["breakfast_bottom"])
    ln_y0 = int(H * CROP["lunch_top"])
    ln_y1 = int(H * CROP["lunch_bottom"])
    dn_y0 = int(H * CROP["dinner_top"])
    dn_y1 = int(H * CROP["dinner_bottom"])

    for i, dk in enumerate(DAY_KEYS):
        x0 = int(col_w * i)
        x1 = int(col_w * (i + 1))

        # 아침(한식만)
        bf_cell = days_area.crop((x0, bf_y0, x1, bf_y1))
        bf_resp = ocr_document(client, to_png_bytes(bf_cell, scale=2))
        bf_text = bf_resp.full_text_annotation.text if bf_resp.full_text_annotation else ""
        out[dk]["breakfast"]["han"] = clean_lines(bf_text)

        # 점심(한식/일품 자동 분리)
        ln_block = days_area.crop((x0, ln_y0, x1, ln_y1))
        han_img, il_img = split_lunch_by_labels(ln_block, client)

        han_resp = ocr_document(client, to_png_bytes(han_img, scale=2))
        il_resp = ocr_document(client, to_png_bytes(il_img, scale=2))
        han_text = han_resp.full_text_annotation.text if han_resp.full_text_annotation else ""
        il_text = il_resp.full_text_annotation.text if il_resp.full_text_annotation else ""

        out[dk]["lunch"]["han"] = clean_lines(han_text)
        out[dk]["lunch"]["ilpum"] = clean_lines(il_text)

        # 저녁(한식만)
        dn_cell = days_area.crop((x0, dn_y0, x1, dn_y1))
        dn_resp = ocr_document(client, to_png_bytes(dn_cell, scale=2))
        dn_text = dn_resp.full_text_annotation.text if dn_resp.full_text_annotation else ""
        out[dk]["dinner"]["han"] = clean_lines(dn_text)

    return {
        "range": date_range,
        "saved_at_kst": kst_now().isoformat(timespec="seconds"),
        "days": out,
    }

# =========================
# Storage
# =========================
def save_meals(data: dict):
    with open(MEALS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_meals() -> dict:
    if not os.path.exists(MEALS_FILE):
        return {}
    try:
        with open(MEALS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# =========================
# Button UX
# =========================
def format_meal_full(day_selector: str, meal_key: str, option: str | None = None) -> str:
    label = "오늘" if day_selector == "today" else "내일"
    data = load_meals()
    rng = data.get("range")
    dkey = day_key(0 if day_selector == "today" else 1)
    day = (data.get("days", {}) or {}).get(dkey, {})

    header = f"📋 {label} {MEAL_NAME[meal_key]} 전체 메뉴"
    if rng:
        header += f"\n(저장된 식단표: {rng})"

    if meal_key == "lunch":
        opt = option or "han"
        opt_name = "한식" if opt == "han" else "일품"
        menu = (day.get("lunch", {}) or {}).get(opt, [])
        if not menu:
            return header + f"\n\n[{opt_name}] (메뉴 없음/파싱 실패)"
        return header + f"\n\n[{opt_name}]\n" + "\n".join(menu)

    menu = (day.get(meal_key, {}) or {}).get("han", [])
    if not menu:
        return header + "\n\n(메뉴 없음/파싱 실패)"
    return header + "\n\n" + "\n".join(menu)

def next_buttons(day_selector: str, meal_key: str) -> dict:
    if day_selector == "today" and meal_key == "breakfast":
        return {"inline_keyboard": [[
            {"text": "점심(한식)", "callback_data": "view|today|lunch|han"},
            {"text": "점심(일품)", "callback_data": "view|today|lunch|ilpum"},
        ], [
            {"text": "저녁", "callback_data": "view|today|dinner"},
        ]]}
    if day_selector == "today" and meal_key == "lunch":
        return {"inline_keyboard": [[
            {"text": "점심(한식)", "callback_data": "view|today|lunch|han"},
            {"text": "점심(일품)", "callback_data": "view|today|lunch|ilpum"},
        ], [
            {"text": "저녁", "callback_data": "view|today|dinner"},
        ]]}
    if day_selector == "today" and meal_key == "dinner":
        return {"inline_keyboard": [[
            {"text": "내일 아침", "callback_data": "view|tomorrow|breakfast"},
        ]]}
    return {"inline_keyboard": []}

# =========================
# Alerts
# =========================
def send_meal_alert(meal_key: str):
    data = load_meals()
    dkey = day_key(0)
    day = (data.get("days", {}) or {}).get(dkey, {})

    if meal_key == "lunch":
        han = (day.get("lunch", {}) or {}).get("han", [])
        il = (day.get("lunch", {}) or {}).get("ilpum", [])
        han_main = han[0] if han else "메뉴없음"
        il_main = il[0] if il else None

        if il_main and il_main != "메뉴없음":
            text = f"🍱 오늘 점심 메뉴\n한식: {han_main}\n일품: {il_main}"
        else:
            text = f"🍱 오늘 점심 메뉴는 {han_main} 입니다"

        keyboard = {"inline_keyboard": [[
            {"text": "전체 메뉴 보기", "callback_data": "view|today|lunch|han"}
        ]]}
        tg_send(text, keyboard)
        return

    menu = (day.get(meal_key, {}) or {}).get("han", [])
    main = menu[0] if menu else "메뉴 정보 없음"
    text = f"🍱 오늘 {MEAL_NAME[meal_key]} 메뉴는 {main} 입니다"
    keyboard = {"inline_keyboard": [[
        {"text": "전체 메뉴 보기", "callback_data": f"view|today|{meal_key}"}
    ]]}
    tg_send(text, keyboard)

# =========================
# Routes
# =========================
@app.route("/", methods=["POST"])
def webhook():
    data = request.json or {}

    # 관리자 사진 업로드
    if "message" in data:
        msg = data["message"]
        user_id = msg.get("from", {}).get("id")

        if "photo" in msg and user_id == ADMIN_ID:
            try:
                file_id = msg["photo"][-1]["file_id"]
                img_bytes = download_telegram_photo(file_id)

                parsed = parse_weekly(img_bytes)
                save_meals(parsed)

                keyboard = {"inline_keyboard": [[
                    {"text": "점심(한식) 보기", "callback_data": "view|today|lunch|han"},
                    {"text": "점심(일품) 보기", "callback_data": "view|today|lunch|ilpum"},
                ]]}
                tg_send(f"📅 {parsed['range']} 식단표 업로드 완료!", keyboard)

            except Exception as e:
                tg_send(f"⚠️ 업로드/파싱 오류: {type(e).__name__}")

    # 버튼 클릭
    if "callback_query" in data:
        q = data["callback_query"]
        cb = q.get("data", "")
        tg_answer_callback(q.get("id"))

        msg = q.get("message", {})
        chat_id = msg.get("chat", {}).get("id")
        message_id = msg.get("message_id")

        if cb.startswith("view|"):
            try:
                parts = cb.split("|")
                # view|today|lunch|han
                day_selector = parts[1]
                meal_key = parts[2]
                opt = parts[3] if len(parts) >= 4 else None

                text = format_meal_full(day_selector, meal_key, opt)
                keyboard = next_buttons(day_selector, meal_key)
                tg_edit(chat_id, message_id, text, keyboard)
            except Exception:
                pass

    return "ok"

@app.route("/cron/send", methods=["GET"])
def cron_send():
    secret = request.args.get("secret", "")
    if not CRON_SECRET or secret != CRON_SECRET:
        return "forbidden", 403

    meal = request.args.get("meal", "")
    if meal not in {"breakfast", "lunch", "dinner"}:
        return "bad meal", 400

    try:
        send_meal_alert(meal)
        return "ok", 200
    except Exception as e:
        return f"error: {type(e).__name__}", 500

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
