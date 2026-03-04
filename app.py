from flask import Flask, request
import os, json, re, tempfile
import requests
from datetime import datetime, timedelta
from google.cloud import vision
from PIL import Image
import io

# -----------------------
# ENV
# -----------------------
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")          # -100... (채널)
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))    # 내 개인 텔레그램 user id
GCP_SA_JSON = os.getenv("GCP_SA_JSON")        # 서비스계정 JSON 전체
CRON_SECRET = os.getenv("CRON_SECRET", "")    # /cron 호출 보호용

app = Flask(__name__)

MEALS_FILE = "meals.json"

# -----------------------
# CROP CONFIG (템플릿 고정일 때 비율로 자르기)
# -----------------------
# 전체 이미지에서 "메뉴 표 영역"을 잘라내는 비율(좌/상/우/하)
# (현재 너가 올린 샘플 기준으로 1차 세팅)
CROP = {
    "table_left": 0.05,
    "table_top": 0.10,
    "table_right": 0.985,
    "table_bottom": 0.93,

    # table 안에서 "요일 컬럼 7개" 영역 (왼쪽 '구분' 컬럼 제외)
    "days_left": 0.12,
    "days_right": 0.995,

    # table 안에서 식사 row 영역(대략): 아침/점심/저녁
    # (표가 고정이면 이 비율도 고정이라 매우 안정적)
    "breakfast_top": 0.08,
    "breakfast_bottom": 0.28,

    "lunch_top": 0.28,
    "lunch_bottom": 0.73,

    "dinner_top": 0.73,
    "dinner_bottom": 0.93,
}

DAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
MEAL_KEYS = ["breakfast", "lunch", "dinner"]
MEAL_NAME = {"breakfast": "아침", "lunch": "점심", "dinner": "저녁"}

# OCR 결과에서 제거할 잡음 단어(표에 반복적으로 뜨는 레이블)
NOISE_TOKENS = {
    "한식", "일품", "SELF", "PLUS", "공룡", "오늘의차", "self", "plus",
    "누룽지", "도시락김", "그린샐러드",
    "조식", "중식", "석식", "관리", "1조", "2조",
    "예약", "수량", "아침", "점심", "저녁",
}

# -----------------------
# Time helpers (KST)
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
# Google Vision OCR
# -----------------------
def _vision_client() -> vision.ImageAnnotatorClient:
    if not GCP_SA_JSON:
        raise RuntimeError("GCP_SA_JSON env var is missing")

    sa = json.loads(GCP_SA_JSON)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sa, f)
        sa_path = f.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
    return vision.ImageAnnotatorClient()

def vision_ocr_text(image_bytes: bytes) -> str:
    client = _vision_client()
    image = vision.Image(content=image_bytes)

    # 표/문서류는 document_text_detection이 더 안정적인 경우가 많음
    resp = client.document_text_detection(image=image)

    if resp.error and resp.error.message:
        raise RuntimeError(f"Vision OCR error: {resp.error.message}")

    if resp.full_text_annotation and resp.full_text_annotation.text:
        return resp.full_text_annotation.text
    return ""

# -----------------------
# Date range extraction (전체 OCR 텍스트에서 월~일 찾기)
# -----------------------
def extract_date_range_from_ocr(text: str) -> str | None:
    # 강한 패턴: "3월 2일 월요일"
    m = re.findall(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일\s*(월|화|수|목|금|토|일)\s*요일", text)
    if len(m) >= 2:
        s = m[0]; e = m[-1]
        return f"{s[0]}/{s[1]}({s[2]}) ~ {e[0]}/{e[1]}({e[2]})"

    # 약한 패턴(요일 누락): "3월 2일"
    d = re.findall(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", text)
    if len(d) >= 2:
        s = d[0]; e = d[-1]
        return f"{s[0]}/{s[1]} ~ {e[0]}/{e[1]}"
    return None

# -----------------------
# Image crop helpers
# -----------------------
def crop_by_ratio(img: Image.Image, left: float, top: float, right: float, bottom: float) -> Image.Image:
    w, h = img.size
    box = (int(w * left), int(h * top), int(w * right), int(h * bottom))
    return img.crop(box)

def img_to_png_bytes(img: Image.Image) -> bytes:
    # OCR 성능을 위해 약간 키워주기(2x)
    img2 = img.resize((img.size[0] * 2, img.size[1] * 2))
    buf = io.BytesIO()
    img2.save(buf, format="PNG")
    return buf.getvalue()

def clean_menu_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue

        # 너무 짧은/의미 없는 조각 제거
        if len(s) <= 1:
            continue

        # 시간/범위 같은 거 제거
        if re.search(r"\d{1,2}:\d{2}", s):
            continue

        # 구분 라벨/잡음 제거
        if s in NOISE_TOKENS:
            continue

        # 별표/기호 정리
        s = s.replace("•", "").replace("·", " ").replace("|", " ").strip()

        # 'S' 같은 소스표기 뒤에 붙는 경우가 있어서 과도한 단독 영문 제거
        if re.fullmatch(r"[A-Za-z]{1,2}", s):
            continue

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
# Parse weekly menu from image (7x3 cells)
# -----------------------
def parse_weekly_menu_from_image(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 1) 표 전체 영역 crop
    table = crop_by_ratio(
        img,
        CROP["table_left"], CROP["table_top"], CROP["table_right"], CROP["table_bottom"]
    )

    # 2) table 안에서 요일 컬럼 영역 crop (왼쪽 '구분' 제외)
    days_area = crop_by_ratio(
        table,
        CROP["days_left"], 0.0, CROP["days_right"], 1.0
    )

    # 3) 식사별 y 범위 (table 기준 비율)
    row_ranges = {
        "breakfast": (CROP["breakfast_top"], CROP["breakfast_bottom"]),
        "lunch": (CROP["lunch_top"], CROP["lunch_bottom"]),
        "dinner": (CROP["dinner_top"], CROP["dinner_bottom"]),
    }

    client = _vision_client()

    result_days = {dk: {mk: [] for mk in MEAL_KEYS} for dk in DAY_KEYS}

    # 4) 요일 7등분
    W, H = days_area.size
    col_w = W / 7.0

    for col_idx, dk in enumerate(DAY_KEYS):
        x0 = int(col_w * col_idx)
        x1 = int(col_w * (col_idx + 1))

        for mk in MEAL_KEYS:
            ry0, ry1 = row_ranges[mk]
            y0 = int(H * ry0)
            y1 = int(H * ry1)

            cell = days_area.crop((x0, y0, x1, y1))
            cell_bytes = img_to_png_bytes(cell)

            # OCR (document_text_detection)
            image = vision.Image(content=cell_bytes)
            resp = client.document_text_detection(image=image)
            if resp.error and resp.error.message:
                raise RuntimeError(f"Vision OCR error: {resp.error.message}")

            text = resp.full_text_annotation.text if resp.full_text_annotation else ""
            menu_lines = clean_menu_lines(text)

            # 첫 줄이 메인 메뉴가 되도록 "너무 긴 빈/잡음" 제거 후 저장
            result_days[dk][mk] = menu_lines

    # 5) 전체 OCR로 날짜 범위도 구하기 (표 전체에서 한번만)
    full_text = vision_ocr_text(image_bytes)
    date_range = extract_date_range_from_ocr(full_text) or "날짜 인식 실패"

    return {
        "range": date_range,
        "saved_at_kst": kst_now().isoformat(timespec="seconds"),
        "days": result_days,
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
def format_meal_full(day_selector: str, meal_key: str) -> str:
    label = "오늘" if day_selector == "today" else "내일"
    data = load_meals()
    days = data.get("days", {})
    dkey = day_key(0 if day_selector == "today" else 1)
    menu = days.get(dkey, {}).get(meal_key, [])

    rng = data.get("range")
    header = f"📋 {label} {MEAL_NAME[meal_key]} 전체 메뉴"
    if rng:
        header += f"\n(저장된 식단표: {rng})"

    if not menu:
        return header + "\n\n(메뉴가 비어있어요. 업로드한 식단표가 파싱되지 않았거나 크롭이 맞지 않을 수 있어요.)"

    return header + "\n\n" + "\n".join(menu)

def next_buttons(day_selector: str, meal_key: str) -> dict:
    if day_selector == "today" and meal_key == "breakfast":
        return {"inline_keyboard": [[
            {"text": "점심 전체", "callback_data": "view|today|lunch"},
            {"text": "저녁 전체", "callback_data": "view|today|dinner"},
        ]]}
    if day_selector == "today" and meal_key == "lunch":
        return {"inline_keyboard": [[
            {"text": "저녁 전체", "callback_data": "view|today|dinner"},
        ]]}
    if day_selector == "today" and meal_key == "dinner":
        return {"inline_keyboard": [[
            {"text": "내일 아침", "callback_data": "view|tomorrow|breakfast"},
        ]]}
    return {"inline_keyboard": []}

# -----------------------
# Alert send (one-line + button)
# -----------------------
def send_meal_alert(meal_key: str):
    data = load_meals()
    days = data.get("days", {})
    dkey = day_key(0)
    menu = days.get(dkey, {}).get(meal_key, [])
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

    # 1) 관리자 사진 업로드
    if "message" in data:
        msg = data["message"]
        user_id = msg.get("from", {}).get("id")

        if "photo" in msg:
            if user_id != ADMIN_ID:
                return "ok"

            try:
                file_id = msg["photo"][-1]["file_id"]
                img_bytes = download_telegram_photo(file_id)

                parsed = parse_weekly_menu_from_image(img_bytes)
                save_meals(parsed)

                # 업로드 완료 + (테스트용) 점심 전체 보기 버튼
                keyboard = {"inline_keyboard": [[
                    {"text": "전체 메뉴 보기(점심)", "callback_data": "view|today|lunch"}
                ]]}
                tg_send_message(f"📅 {parsed['range']} 식단표 업로드 완료!", keyboard=keyboard)

            except Exception as e:
                tg_send_message(f"⚠️ 업로드/파싱 오류: {type(e).__name__}")

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
                _, day_selector, meal_key = cb.split("|", 2)
                text = format_meal_full(day_selector, meal_key)
                keyboard = next_buttons(day_selector, meal_key)
                tg_edit_message(chat_id, message_id, text, keyboard=keyboard)
            except Exception:
                pass

    return "ok"

@app.route("/cron/send", methods=["GET"])
def cron_send():
    # 예: /cron/send?meal=lunch&secret=XXXX
    secret = request.args.get("secret", "")
    if not CRON_SECRET or secret != CRON_SECRET:
        return "forbidden", 403

    meal = request.args.get("meal", "")
    if meal not in MEAL_KEYS:
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
