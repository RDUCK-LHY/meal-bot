from flask import Flask, request
import os, json, re, tempfile, io
import requests
from datetime import datetime, timedelta
from PIL import Image
from google.cloud import vision

# ============================================================
# ENV
# ============================================================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")          # -100...
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))    # 너의 user id
GCP_SA_JSON = os.getenv("GCP_SA_JSON")        # 서비스계정 JSON 문자열
CRON_SECRET = os.getenv("CRON_SECRET", "")    # cron 보호용

app = Flask(__name__)
MEALS_FILE = "meals.json"

DAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
DAY_KR = {"mon":"월","tue":"화","wed":"수","thu":"목","fri":"금","sat":"토","sun":"일"}
MEAL_NAME = {"breakfast":"아침", "lunch":"점심", "dinner":"저녁"}

# ============================================================
# 템플릿 고정 기반: 큰 영역만 비율로 자르고,
# 점심의 한식/일품 경계는 OCR 좌표로 자동 잡음
# (필요하면 아래 값만 미세조정)
# ============================================================
CROP = {
    # 이미지 전체에서 "표" 영역 (대략)
    "table_left": 0.05,
    "table_top": 0.10,
    "table_right": 0.985,
    "table_bottom": 0.93,

    # 표 내부에서 "요일 7칸" 영역 (왼쪽 구분 컬럼 제외)
    "days_left": 0.12,
    "days_right": 0.995,

    # 관리직 기준 사용 블록
    # 아침: 1조/관리 조식
    "breakfast_top": 0.08,
    "breakfast_bottom": 0.28,

    # 점심: 1조 중식 / 관리 중식 (점심 전체 블록)
    "lunch_top": 0.28,
    "lunch_bottom": 0.73,

    # 저녁: 2조 조식 / 1조 석식 (저녁 전체 블록)
    "dinner_top": 0.73,
    "dinner_bottom": 0.93,
}

# ============================================================
# 노이즈 제거
# ============================================================
NOISE_EXACT = {"한식", "일품", "공통", "SELF", "PLUS", "오늘의차"}
NOISE_CONTAINS = [
    "공통", "SELF", "PLUS", "오늘의차",
    "라면", "코너", "누룽지", "도시락김", "샐러드", "그린",
    "STEMS", "aramark", "구분",
    "조식", "중식", "석식", "1조", "2조", "관리",
    "차", "우유", "요거트",
]

# ============================================================
# 메인메뉴 추정(휴리스틱)
# ============================================================
MAIN_HINTS = [
    "국", "찌개", "탕", "전골", "순두부", "육개장", "설렁탕", "감자탕", "부대찌개", "미역국", "된장",
    "덮밥", "비빔밥", "볶음밥", "카레", "짜장", "짬뽕", "우동", "국수", "라면",
    "돈까스", "스테이크", "파스타", "스파게티", "피자", "햄버거",
    "갈비", "불고기", "제육", "닭갈비", "찜", "구이", "조림", "볶음",
]
NOT_MAIN_HINTS = [
    "김치", "깍두기", "단무지", "무말랭이", "나물", "샐러드", "도시락김",
    "밥", "현미밥", "잡곡밥", "누룽지", "차", "과일",
    "소스", "케찹", "드레싱", "피클", "초장", "양념장",
]

# ============================================================
# Time (KST)
# ============================================================
def kst_now() -> datetime:
    return datetime.utcnow() + timedelta(hours=9)

def today_day_key(offset_days: int = 0) -> str:
    d = kst_now().date() + timedelta(days=offset_days)
    return DAY_KEYS[d.weekday()]

# ============================================================
# Telegram helpers
# ============================================================
def tg_send(text: str, keyboard: dict | None = None):
    data = {"chat_id": CHANNEL_ID, "text": text}
    if keyboard is not None:
        data["reply_markup"] = json.dumps(keyboard, ensure_ascii=False)
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data=data, timeout=20)
    r.raise_for_status()
    return r.json()

def tg_edit(chat_id, message_id, text: str, keyboard: dict | None = None):
    data = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if keyboard is not None:
        data["reply_markup"] = json.dumps(keyboard, ensure_ascii=False)
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/editMessageText", data=data, timeout=20)
    r.raise_for_status()
    return r.json()

def tg_answer_callback(callback_query_id: str):
    requests.post(
        f"https://api.telegram.org/bot{TOKEN}/answerCallbackQuery",
        data={"callback_query_id": callback_query_id},
        timeout=10,
    )

def download_telegram_photo(file_id: str) -> bytes:
    r = requests.get(
        f"https://api.telegram.org/bot{TOKEN}/getFile",
        params={"file_id": file_id},
        timeout=20,
    )
    r.raise_for_status()
    file_path = r.json()["result"]["file_path"]

    img = requests.get(
        f"https://api.telegram.org/file/bot{TOKEN}/{file_path}",
        timeout=30,
    )
    img.raise_for_status()
    return img.content

# ============================================================
# Vision OCR
# ============================================================
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
    image = vision.Image(content=image_bytes)
    resp = client.document_text_detection(image=image)
    if resp.error and resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp

# ============================================================
# Image helpers
# ============================================================
def crop_by_ratio(img: Image.Image, l: float, t: float, r: float, b: float) -> Image.Image:
    w, h = img.size
    return img.crop((int(w*l), int(h*t), int(w*r), int(h*b)))

def to_png_bytes(img: Image.Image, scale: int = 2) -> bytes:
    img2 = img.resize((img.size[0]*scale, img.size[1]*scale))
    buf = io.BytesIO()
    img2.save(buf, format="PNG")
    return buf.getvalue()

# ============================================================
# Text cleaning & main menu
# ============================================================
def clean_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if len(s) <= 1:
            continue
        if re.search(r"\d{1,2}:\d{2}", s):
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

def is_not_main(line: str) -> bool:
    for x in NOT_MAIN_HINTS:
        if x in line:
            return True
    return False

def score_main(line: str) -> int:
    s = 0
    if len(line) <= 2:
        s -= 3
    if is_not_main(line):
        s -= 6
    for h in MAIN_HINTS:
        if h in line:
            s += 5
            break
    if "*" in line:
        s -= 1
    return s

def pick_main_menu(lines: list[str]) -> str:
    if not lines:
        return "메뉴 정보 없음"
    scored = [(score_main(x), i, x) for i, x in enumerate(lines)]
    scored.sort(reverse=True, key=lambda t: (t[0], -t[1]))
    best_score, _, best_line = scored[0]
    if best_score <= -3:
        return lines[0]
    return best_line

def normalize_menu_list(lines: list[str]) -> list[str]:
    if not lines:
        return lines
    main = pick_main_menu(lines)
    if main in lines:
        return [main] + [x for x in lines if x != main]
    return lines

# ============================================================
# Date range extraction (업로드 완료 메시지용)
# ============================================================
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

# ============================================================
# Lunch split by label boxes (핵심)
# ============================================================
def _iter_words_with_boxes(resp):
    if not resp.full_text_annotation or not resp.full_text_annotation.pages:
        return
    for page in resp.full_text_annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    wtxt = "".join([s.text for s in word.symbols]).strip()
                    if not wtxt:
                        continue
                    bbox = word.bounding_box.vertices
                    yield wtxt, bbox

def _bbox_bottom_y(bbox) -> int:
    return max(v.y for v in bbox if v.y is not None)

def _bbox_top_y(bbox) -> int:
    return min(v.y for v in bbox if v.y is not None)

def find_label_y(resp, label: str) -> tuple[int | None, int | None]:
    candidates = []
    for wtxt, bbox in _iter_words_with_boxes(resp):
        if wtxt == label:
            candidates.append((_bbox_top_y(bbox), _bbox_bottom_y(bbox)))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    return candidates[0]

def split_lunch_by_labels(lunch_img: Image.Image, client: vision.ImageAnnotatorClient):
    """
    lunch_img(요일 1칸의 점심 블록)에서
    한식 메뉴 영역 / 일품 메뉴 영역을 자동 분리해 반환.
    """
    scale = 2
    resp = ocr_document(client, to_png_bytes(lunch_img, scale=scale))
    han_top, han_bottom = find_label_y(resp, "한식")
    il_top, il_bottom = find_label_y(resp, "일품")

    W, H = lunch_img.size

    # 라벨 인식 실패 시 fallback (대략 반반)
    if han_bottom is None or il_top is None or il_bottom is None:
        mid = int(H * 0.55)
        return lunch_img.crop((0, 0, W, mid)), lunch_img.crop((0, mid, W, H))

    # OCR 좌표는 scale배 -> 원본으로 환산
    # 메인 누락 방지 위해 한식 시작을 조금 "위로" 당김
    han_menu_top = int((han_bottom - 12) / scale)   # <-- 여기 핵심 보정
    il_label_top = int((il_top - 2) / scale)
    il_menu_top = int((il_bottom + 4) / scale)

    # clamp
    han_menu_top = max(0, min(H, han_menu_top))
    il_label_top = max(0, min(H, il_label_top))
    il_menu_top = max(0, min(H, il_menu_top))

    # sanity: 너무 가까우면 fallback
    if il_label_top <= han_menu_top + 10:
        mid = int(H * 0.55)
        return lunch_img.crop((0, 0, W, mid)), lunch_img.crop((0, mid, W, H))

    han_menu = lunch_img.crop((0, han_menu_top, W, il_label_top))
    il_menu = lunch_img.crop((0, il_menu_top, W, H))
    return han_menu, il_menu

# ============================================================
# Storage (안전 로드)
# ============================================================
def save_meals(data: dict):
    with open(MEALS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_meals() -> dict:
    if not os.path.exists(MEALS_FILE):
        return {"range": "", "days": {}}
    try:
        with open(MEALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"range": "", "days": {}}
        if "days" not in data or not isinstance(data["days"], dict):
            data["days"] = {}
        if "range" not in data:
            data["range"] = ""
        return data
    except Exception:
        return {"range": "", "days": {}}

# ============================================================
# Parse weekly menu (관리직 기준)
# ============================================================
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

    # 날짜 범위
    full_resp = ocr_document(client, to_png_bytes(img, scale=2))
    full_text = full_resp.full_text_annotation.text if full_resp.full_text_annotation else ""
    date_range = extract_date_range(full_text) or "날짜 인식 실패"

    bf_y0, bf_y1 = int(H*CROP["breakfast_top"]), int(H*CROP["breakfast_bottom"])
    ln_y0, ln_y1 = int(H*CROP["lunch_top"]), int(H*CROP["lunch_bottom"])
    dn_y0, dn_y1 = int(H*CROP["dinner_top"]), int(H*CROP["dinner_bottom"])

    for i, dk in enumerate(DAY_KEYS):
        x0, x1 = int(col_w*i), int(col_w*(i+1))

        # breakfast (한식만)
        bf_cell = days_area.crop((x0, bf_y0, x1, bf_y1))
        bf_resp = ocr_document(client, to_png_bytes(bf_cell, scale=2))
        bf_text = bf_resp.full_text_annotation.text if bf_resp.full_text_annotation else ""
        out[dk]["breakfast"]["han"] = normalize_menu_list(clean_lines(bf_text))

        # lunch (한식/일품 자동 분리)
        ln_block = days_area.crop((x0, ln_y0, x1, ln_y1))
        han_img, il_img = split_lunch_by_labels(ln_block, client)

        han_resp = ocr_document(client, to_png_bytes(han_img, scale=2))
        il_resp = ocr_document(client, to_png_bytes(il_img, scale=2))
        han_text = han_resp.full_text_annotation.text if han_resp.full_text_annotation else ""
        il_text = il_resp.full_text_annotation.text if il_resp.full_text_annotation else ""

        out[dk]["lunch"]["han"] = normalize_menu_list(clean_lines(han_text))
        out[dk]["lunch"]["ilpum"] = normalize_menu_list(clean_lines(il_text))

        # dinner (한식만)
        dn_cell = days_area.crop((x0, dn_y0, x1, dn_y1))
        dn_resp = ocr_document(client, to_png_bytes(dn_cell, scale=2))
        dn_text = dn_resp.full_text_annotation.text if dn_resp.full_text_annotation else ""
        out[dk]["dinner"]["han"] = normalize_menu_list(clean_lines(dn_text))

    return {
        "range": date_range,
        "saved_at_kst": kst_now().isoformat(timespec="seconds"),
        "days": out,
    }

# ============================================================
# UI: view full menu
# ============================================================
def format_meal_full(day_selector: str, meal_key: str, option: str | None = None) -> str:
    label = "오늘" if day_selector == "today" else "내일"
    data = load_meals()
    rng = data.get("range", "")
    dkey = today_day_key(0 if day_selector == "today" else 1)
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

# ============================================================
# Alerts
# ============================================================
def send_meal_alert(meal_key: str):
    if meal_key not in ("breakfast", "lunch", "dinner"):
        tg_send("⚠️ cron 호출 오류: meal 파라미터가 잘못됐어요.")
        return

    data = load_meals()
    days = data.get("days", {})
    if not days:
        tg_send("⚠️ 아직 식단표가 저장되지 않았어요. 관리자(업로더)가 식단표 사진을 먼저 올려야 합니다.")
        return

    dkey = today_day_key(0)
    day = days.get(dkey, {})

    if meal_key == "lunch":
        han = (day.get("lunch", {}) or {}).get("han", [])
        il = (day.get("lunch", {}) or {}).get("ilpum", [])

        han_main = pick_main_menu(han) if han else "메뉴없음"
        il_main = pick_main_menu(il) if il else None

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
    main = pick_main_menu(menu) if menu else "메뉴 정보 없음"
    text = f"🍱 오늘 {MEAL_NAME[meal_key]} 메뉴는 {main} 입니다"
    keyboard = {"inline_keyboard": [[
        {"text": "전체 메뉴 보기", "callback_data": f"view|today|{meal_key}"}
    ]]}
    tg_send(text, keyboard)

# 월요일 8시 업로드 요청(수동 업로드 유도)
def remind_upload():
    tg_send("📸 이번 주 식단표 사진을 업로드해주세요! (관리자만 업로드 가능)")

# ============================================================
# Routes
# ============================================================
@app.route("/", methods=["POST"])
def webhook():
    data = request.json or {}

    # 1) 관리자 사진 업로드
    if "message" in data:
        msg = data["message"]
        user_id = msg.get("from", {}).get("id")

        if "photo" in msg:
            if user_id != ADMIN_ID:
                # 관리자 외 업로드 무시 (필요하면 안내 메시지 보낼 수도 있음)
                return "ok"

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

            return "ok"

    # 2) 버튼 클릭
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
    if request.args.get("secret", "") != CRON_SECRET:
        return "forbidden", 403

    meal = request.args.get("meal", "")
    if meal not in ("breakfast", "lunch", "dinner"):
        return "bad meal", 400

    try:
        send_meal_alert(meal)
        return "ok", 200
    except Exception as e:
        return f"error: {type(e).__name__}", 500

@app.route("/cron/remind_upload", methods=["GET"])
def cron_remind_upload():
    if request.args.get("secret", "") != CRON_SECRET:
        return "forbidden", 403
    try:
        remind_upload()
        return "ok", 200
    except Exception as e:
        return f"error: {type(e).__name__}", 500

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
