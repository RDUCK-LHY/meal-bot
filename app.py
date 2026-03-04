from flask import Flask, request
import os, json, re, tempfile
import requests
from datetime import datetime, timedelta
from google.cloud import vision

# -----------------------
# ENV
# -----------------------
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")          # -100... (채널)
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))    # 내 개인 텔레그램 user id
GCP_SA_JSON = os.getenv("GCP_SA_JSON")        # 서비스계정 JSON 전체

app = Flask(__name__)

MEALS_FILE = "meals.json"  # Render 디스크는 영구 보장 X (추후 DB/Supabase 추천)


# -----------------------
# Time helpers (KST)
# -----------------------
def kst_now() -> datetime:
    return datetime.utcnow() + timedelta(hours=9)

def day_key(offset_days: int = 0) -> str:
    # mon..sun
    keys = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    d = kst_now().date() + timedelta(days=offset_days)
    return keys[d.weekday()]


# -----------------------
# Telegram helpers
# -----------------------
def tg_send_message(text: str, keyboard: dict | None = None):
    data = {"chat_id": CHANNEL_ID, "text": text}
    if keyboard is not None:
        data["reply_markup"] = json.dumps(keyboard, ensure_ascii=False)

    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data=data,
        timeout=20,
    )
    r.raise_for_status()
    return r.json()

def tg_edit_message(chat_id: int | str, message_id: int, text: str, keyboard: dict | None = None):
    data = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if keyboard is not None:
        data["reply_markup"] = json.dumps(keyboard, ensure_ascii=False)

    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/editMessageText",
        data=data,
        timeout=20,
    )
    r.raise_for_status()
    return r.json()

def tg_answer_callback(callback_query_id: str):
    # 버튼 눌렀을 때 텔레그램 로딩 동그라미 빨리 없애기
    requests.post(
        f"https://api.telegram.org/bot{TOKEN}/answerCallbackQuery",
        data={"callback_query_id": callback_query_id},
        timeout=10,
    )


# -----------------------
# Telegram photo download
# -----------------------
def download_telegram_photo(file_id: str) -> bytes:
    # 1) file_path 얻기
    r = requests.get(
        f"https://api.telegram.org/bot{TOKEN}/getFile",
        params={"file_id": file_id},
        timeout=20,
    )
    r.raise_for_status()
    file_path = r.json()["result"]["file_path"]

    # 2) 실제 이미지 bytes 다운로드
    img = requests.get(
        f"https://api.telegram.org/file/bot{TOKEN}/{file_path}",
        timeout=30,
    )
    img.raise_for_status()
    return img.content


# -----------------------
# Google Vision OCR
# -----------------------
def vision_ocr_text(image_bytes: bytes) -> str:
    if not GCP_SA_JSON:
        raise RuntimeError("GCP_SA_JSON env var is missing")

    # env에 넣은 JSON을 임시 파일로 만들어 인증
    sa = json.loads(GCP_SA_JSON)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sa, f)
        sa_path = f.name

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    resp = client.text_detection(image=image)

    if resp.error and resp.error.message:
        raise RuntimeError(f"Vision OCR error: {resp.error.message}")

    if resp.text_annotations:
        return resp.text_annotations[0].description
    return ""


# -----------------------
# Date range extraction
# -----------------------
def extract_date_range_from_ocr(text: str) -> str | None:
    """
    이 식단표는 상단에 '3월 2일 월요일' ... '3월 8일 일요일' 형태로 들어감.
    OCR이 약간 깨져도 월/일/요일 패턴으로 범위를 만들 수 있음.
    """
    # 가장 강한 패턴: "3월 2일 월요일"
    m = re.findall(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일\s*(월|화|수|목|금|토|일)\s*요일", text)

    # 약한 패턴(요일 누락): "3월 2일"
    if len(m) < 2:
        d = re.findall(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", text)
        if len(d) >= 2:
            s = d[0]
            e = d[-1]
            return f"{s[0]}/{s[1]} ~ {e[0]}/{e[1]}"
        return None

    s = m[0]
    e = m[-1]
    return f"{s[0]}/{s[1]}({s[2]}) ~ {e[0]}/{e[1]}({e[2]})"


# -----------------------
# meals.json (stub 저장)
# -----------------------
def save_meals_stub(date_range: str, ocr_text: str):
    """
    다음 단계에서 '표 전체 파싱'해서 days[mon..sun][breakfast/lunch/dinner] 채울 예정.
    지금은 날짜/원문 OCR만 저장해두고 버튼 UX 테스트 가능하게.
    """
    data = {
        "range": date_range,
        "saved_at_kst": kst_now().isoformat(timespec="seconds"),
        "ocr_text": ocr_text[:5000],  # 너무 길면 잘라서 저장
        "days": {
            "mon": {"breakfast": [], "lunch": [], "dinner": []},
            "tue": {"breakfast": [], "lunch": [], "dinner": []},
            "wed": {"breakfast": [], "lunch": [], "dinner": []},
            "thu": {"breakfast": [], "lunch": [], "dinner": []},
            "fri": {"breakfast": [], "lunch": [], "dinner": []},
            "sat": {"breakfast": [], "lunch": [], "dinner": []},
            "sun": {"breakfast": [], "lunch": [], "dinner": []},
        },
    }
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
# Button UX helpers
# -----------------------
def format_meal_full(day_selector: str, meal_key: str) -> str:
    """
    day_selector: today / tomorrow
    meal_key: breakfast / lunch / dinner
    """
    meal_name = {"breakfast": "아침", "lunch": "점심", "dinner": "저녁"}[meal_key]
    label = "오늘" if day_selector == "today" else "내일"

    data = load_meals()
    days = data.get("days", {})
    dkey = day_key(0 if day_selector == "today" else 1)
    menu = days.get(dkey, {}).get(meal_key, [])

    if not menu:
        rng = data.get("range")
        if rng:
            return f"📋 {label} {meal_name} 전체 메뉴\n(저장된 식단표: {rng})\n\n(아직 메뉴 파싱이 적용되지 않았어요. 다음 단계에서 자동으로 채워집니다.)"
        return f"📋 {label} {meal_name} 전체 메뉴\n\n(아직 식단표가 저장되지 않았어요. 먼저 사진을 업로드해주세요.)"

    return f"📋 {label} {meal_name} 전체 메뉴\n\n" + "\n".join(menu)


def next_buttons(day_selector: str, meal_key: str) -> dict:
    # 요청한 UX: 해당 식사만 먼저 보여주고, 다음 건 버튼으로만
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
# Webhook
# -----------------------
@app.route("/", methods=["POST"])
def webhook():
    data = request.json or {}

    # 1) 사진 업로드(관리자만)
    if "message" in data:
        msg = data["message"]
        user_id = msg.get("from", {}).get("id")

        if "photo" in msg:
            if user_id != ADMIN_ID:
                # 권한 없는 사람은 무시(원하면 안내 메시지로 바꿔도 됨)
                return "ok"

            try:
                file_id = msg["photo"][-1]["file_id"]
                img_bytes = download_telegram_photo(file_id)

                ocr_text = vision_ocr_text(img_bytes)
                date_range = extract_date_range_from_ocr(ocr_text) or "날짜 인식 실패"

                save_meals_stub(date_range, ocr_text)

                keyboard = {"inline_keyboard": [[
                    {"text": "전체 메뉴 보기(테스트)", "callback_data": "view|today|lunch"}
                ]]}
                tg_send_message(f"📅 {date_range} 식단표 업로드 완료!", keyboard=keyboard)

            except Exception as e:
                # 운영용: 구체 오류를 너무 노출하기 싫으면 type만 표시
                tg_send_message(f"⚠️ 업로드 처리 오류: {type(e).__name__}")

            return "ok"

    # 2) 버튼 클릭 처리
    if "callback_query" in data:
        q = data["callback_query"]
        cb = q.get("data", "")
        callback_id = q.get("id")
        tg_answer_callback(callback_id)

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
                # 편집 실패 시 무시(권한/메시지 상태 등)
                pass

    return "ok"


# -----------------------
# Run (for local). Render에서는 gunicorn app:app 권장
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
