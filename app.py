from flask import Flask, request
import os, json, re, tempfile, io
import requests
from datetime import datetime, timedelta
from google.cloud import vision
from PIL import Image

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
GCP_SA_JSON = os.getenv("GCP_SA_JSON")
CRON_SECRET = os.getenv("CRON_SECRET", "")

app = Flask(__name__)
MEALS_FILE = "meals.json"

DAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
MEAL_NAME = {"breakfast": "아침", "lunch": "점심", "dinner": "저녁"}

# -----------------------
# 크롭 비율(템플릿 고정 기준 1차값)
# -----------------------
CROP = {
    # 전체 이미지에서 표 영역
    "table_left": 0.05,
    "table_top": 0.10,
    "table_right": 0.985,
    "table_bottom": 0.93,

    # 표 내부에서 요일 영역(왼쪽 '구분' 컬럼 제외)
    "days_left": 0.12,
    "days_right": 0.995,

    # (관리직 기준) "1조/관리 조식" 영역
    "breakfast_top": 0.08,
    "breakfast_bottom": 0.28,

    # (관리직 기준) 점심은 "1조 중식/관리 중식" 블록만
    "lunch_top": 0.28,
    "lunch_bottom": 0.73,

    # 점심 블록 안에서 '한식/일품' 상하 분리
    # (위=한식, 아래=일품) — 1차값 (필요시 0.02씩 조정)
    "lunch_han_rel_top": 0.00,
    "lunch_han_rel_bottom": 0.52,
    "lunch_il_rel_top": 0.52,
    "lunch_il_rel_bottom": 1.00,

    # (관리직 기준) 저녁은 "2조 조식/1조 석식" 블록만 사용
    "dinner_top": 0.73,
    "dinner_bottom": 0.93,
}

# OCR 후 제거할 잡음(표 레이블/코너/부가항목)
NOISE_CONTAINS = [
    "SELF", "self", "PLUS", "plus", "공통", "오늘의차", "라면", "코너",
    "누룽지", "도시락김", "그린", "샐러드", "결명자", "매실", "둥글레",
    "조식", "중식", "석식", "1조", "2조", "관리",
    "STEMS", "구분", "aramark",
]
NOISE_EXACT = {"한식", "일품", "공통", "SELF", "PLUS", "오늘의차"}

# -----------------------
# Time (KST)
# -----------------------
def kst_now() -> datetime:
    return datetime.utcnow() + timedelta(hours=9)

def day_key(offset_days: int = 0) -> str:
    d = (kst_now().date() + timedelta(days=offset_days))
    return DAY_KEYS[d.weekday()]

# -----------------------
# Telegram helpers
# -----------------------
def tg_send_message(text: str, keyboard: dict | None = None):
    data = {"chat_id": CHANNEL_ID, "text": text}
    if keyboard is not None:
        data["reply_markup"] = json.dumps(keyboard, ensure_ascii=False)
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data=data, timeout=20)
    r.raise_for_status()
    return r.json()

def tg_edit_message(chat_id: int | str, message_id: int, text: str, keyboard: dict | None = None):
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

# -----------------------
# Telegram photo download
# -----------------------
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

# -----------------------
# Vision client / OCR
# -----------------------
def vision_client() -> vision.ImageAnnotatorClient:
    if not GCP_SA_JSON:
        raise RuntimeError("GCP_SA_JSON env var is missing")
    sa = json.loads(GCP_SA_JSON)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sa, f)
        sa_path = f.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
    return vision.ImageAnnotatorClient()

def ocr_text_doc(client: vision.ImageAnnotatorClient, image_bytes: bytes) -> str:
    image = vision.Image(content=image_bytes)
    resp = client.document_text_detection(image=image)
    if resp.error and resp.error.message:
        raise RuntimeError(resp.error.message)
    if resp.full_text_annotation and resp.full_text_annotation.text:
        return resp.full_text_annotation.text
    return ""

# -----------------------
# Date range from full OCR
# -----------------------
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

# -----------------------
# Image crop helpers
# -----------------------
def crop_by_ratio(img: Image.Image, l: float, t: float, r: float, b: float) -> Image.Image:
    w, h = img.size
    return img.crop((int(w*l), int(h*t), int(w*r), int(h*b)))

def to_png_bytes(img: Image.Image) -> bytes:
    # OCR 안정성 위해 확대
    img2 = img.resize((img.size[0]*2, img.size[1]*2))
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
        # 기호 정리
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

# -----------------------
# Parse weekly menu (관리직 규칙)
# -----------------------
def parse_weekly(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    table = crop_by_ratio(img, CROP["table_left"], CROP["table_top"], CROP["table_right"], CROP["table_bottom"])
    days_area = crop_by_ratio(table, CROP["days_left"], 0.0, CROP["days_right"], 1.0)

    W, H = days_area.size
    col_w = W / 7.0

    client = vision_client()

    # 결과 구조: breakfast(한식), lunch(한식/일품), dinner(한식)
    out_days = {
        dk: {
            "breakfast": {"han": []},
            "lunch": {"han": [], "ilpum": []},
            "dinner": {"han": []},
        } for dk in DAY_KEYS
    }

    # 전체 OCR로 날짜범위도 얻기
    full_text = ocr_text_doc(client, to_png_bytes(img))
    date_range = extract_date_range(full_text) or "날짜 인식 실패"

    # 영역 계산 (days_area 기준 y)
    def y_range(top_ratio, bottom_ratio):
        return int(H*top_ratio), int(H*bottom_ratio)

    bf_y0, bf_y1 = y_range(CROP["breakfast_top"], CROP["breakfast_bottom"])
    ln_y0, ln_y1 = y_range(CROP["lunch_top"], CROP["lunch_bottom"])
    dn_y0, dn_y1 = y_range(CROP["dinner_top"], CROP["dinner_bottom"])

    for i, dk in enumerate(DAY_KEYS):
        x0 = int(col_w * i)
        x1 = int(col_w * (i+1))

        # 아침(한식만)
        bf_cell = days_area.crop((x0, bf_y0, x1, bf_y1))
        bf_text = ocr_text_doc(client, to_png_bytes(bf_cell))
        out_days[dk]["breakfast"]["han"] = clean_lines(bf_text)

        # 점심(상하: 위=한식, 아래=일품)
        ln_block = days_area.crop((x0, ln_y0, x1, ln_y1))
        lW, lH = ln_block.size

        han = crop_by_ratio(
            ln_block,
            0.0, CROP["lunch_han_rel_top"], 1.0, CROP["lunch_han_rel_bottom"]
        )
        il = crop_by_ratio(
            ln_block,
            0.0, CROP["lunch_il_rel_top"], 1.0, CROP["lunch_il_rel_bottom"]
        )

        han_text = ocr_text_doc(client, to_png_bytes(han))
        il_text = ocr_text_doc(client, to_png_bytes(il))

        out_days[dk]["lunch"]["han"] = clean_lines(han_text)
        out_days[dk]["lunch"]["ilpum"] = clean_lines(il_text)

        # 저녁(한식만) — "2조 조식/1조 석식" 블록에서 한식만이므로 전체를 한식으로 읽고 잡음 제거
        dn_cell = days_area.crop((x0, dn_y0, x1, dn_y1))
        dn_text = ocr_text_doc(client, to_png_bytes(dn_cell))
        out_days[dk]["dinner"]["han"] = clean_lines(dn_text)

    return {
        "range": date_range,
        "saved_at_kst": kst_now().isoformat(timespec="seconds"),
        "days": out_days,
    }

# -----------------------
# Storage
# -----------------------
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

# -----------------------
# Button UX
# -----------------------
def format_meal_full(day_selector: str, meal_key: str, option: str | None = None) -> str:
    label = "오늘" if day_selector == "today" else "내일"
    data = load_meals()
    rng = data.get("range")
    dkey = day_key(0 if day_selector == "today" else 1)

    days = data.get("days", {})
    day = days.get(dkey, {})

    header = f"📋 {label} {MEAL_NAME[meal_key]} 전체 메뉴"
    if rng:
        header += f"\n(저장된 식단표: {rng})"

    if meal_key == "lunch":
        # option: han / ilpum
        opt = option or "han"
        opt_name = "한식" if opt == "han" else "일품"
        menu = day.get("lunch", {}).get(opt, [])
        if not menu:
            return header + f"\n\n[{opt_name}] (메뉴 없음 또는 파싱 실패)"
        return header + f"\n\n[{opt_name}]\n" + "\n".join(menu)

    # breakfast/dinner: han only
    menu = day.get(meal_key, {}).get("han", [])
    if not menu:
        return header + "\n\n(메뉴 없음 또는 파싱 실패)"
    return header + "\n\n" + "\n".join(menu)

def next_buttons(day_selector: str, meal_key: str) -> dict:
    if day_selector == "today" and meal_key == "breakfast":
        return {"inline_keyboard": [[
            {"text": "점심(한식)", "callback_data": "view|today|lunch|han"},
            {"text": "점심(일품)", "callback_data": "view|today|lunch|ilpum"},
        ], [
            {"text": "저녁", "callback_data": "view|today|dinner"}
        ]]}
    if day_selector == "today" and meal_key == "lunch":
        return {"inline_keyboard": [[
            {"text": "점심(한식)", "callback_data": "view|today|lunch|han"},
            {"text": "점심(일품)", "callback_data": "view|today|lunch|ilpum"},
        ], [
            {"text": "저녁", "callback_data": "view|today|dinner"}
        ]]}
    if day_selector == "today" and meal_key == "dinner":
        return {"inline_keyboard": [[
            {"text": "내일 아침", "callback_data": "view|tomorrow|breakfast"}
        ]]}
    return {"inline_keyboard": []}

# -----------------------
# Alerts
# -----------------------
def send_meal_alert(meal_key: str):
    data = load_meals()
    days = data.get("days", {})
    dkey = day_key(0)
    day = days.get(dkey, {})

    if meal_key == "lunch":
        han = day.get("lunch", {}).get("han", [])
        il = day.get("lunch", {}).get("ilpum", [])
        han_main = han[0] if han else "메뉴없음"
        il_main = il[0] if il else None

        if il_main and il_main != "메뉴없음":
            text = f"🍱 오늘 점심 메뉴\n한식: {han_main}\n일품: {il_main}"
        else:
            text = f"🍱 오늘 점심 메뉴는 {han_main} 입니다"

        keyboard = {"inline_keyboard": [[
            {"text": "전체 메뉴 보기", "callback_data": "view|today|lunch|han"}
        ]]}
        tg_send_message(text, keyboard=keyboard)
        return

    # breakfast/dinner
    menu = day.get(meal_key, {}).get("han", [])
    main = menu[0] if menu else "메뉴 정보 없음"
    text = f"🍱 오늘 {MEAL_NAME[meal_key]} 메뉴는 {main} 입니다"
    keyboard = {"inline_keyboard": [[
        {"text": "전체 메뉴 보기", "callback_data": f"view|today|{meal_key}"}
    ]]}
    tg_send_message(text, keyboard=keyboard)

# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["POST"])
def webhook():
    data = request.json or {}

    # 관리자 사진 업로드
    if "message" in data:
        msg = data["message"]
        user_id = msg.get("from", {}).get("id")

        if "photo" in msg:
            if user_id != ADMIN_ID:
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
                tg_send_message(f"📅 {parsed['range']} 식단표 업로드 완료!", keyboard=keyboard)

            except Exception as e:
                tg_send_message(f"⚠️ 업로드/파싱 오류: {type(e).__name__}")

            return "ok"

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
                tg_edit_message(chat_id, message_id, text, keyboard=keyboard)
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
