from flask import Flask, request
import os, json
import requests
from datetime import datetime, timedelta

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")  # -100... 채널
ADMIN_ID = int(os.getenv("ADMIN_ID"))  # 내 개인 id

app = Flask(__name__)

MEALS_FILE = "meals.json"  # 나중에 OCR 결과 저장용

def kst_now():
    return datetime.utcnow() + timedelta(hours=9)

def send_message(text, keyboard=None):
    data = {"chat_id": CHANNEL_ID, "text": text}
    if keyboard:
        data["reply_markup"] = json.dumps(keyboard, ensure_ascii=False)
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data=data, timeout=20)
    r.raise_for_status()

def edit_message(chat_id, message_id, text, keyboard=None):
    data = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if keyboard is not None:
        data["reply_markup"] = json.dumps(keyboard, ensure_ascii=False)
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/editMessageText", data=data, timeout=20)
    r.raise_for_status()

def load_meals():
    # 아직 OCR 구현 전이라 없으면 더미 리턴
    if not os.path.exists(MEALS_FILE):
        return {}
    with open(MEALS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def format_meal_full(day_key, meal_key):
    # day_key: today/tomorrow
    # meal_key: breakfast/lunch/dinner
    meal_name = {"breakfast":"아침", "lunch":"점심", "dinner":"저녁"}[meal_key]
    day_label = "오늘" if day_key == "today" else "내일"
    data = load_meals()

    # 아직 데이터 없을 때도 UX 유지
    menu = data.get(day_key, {}).get(meal_key, [])
    if not menu:
        return f"📋 {day_label} {meal_name} 전체 메뉴\n\n(아직 식단표가 저장되지 않았어요. 먼저 사진을 업로드해주세요.)"
    return f"📋 {day_label} {meal_name} 전체 메뉴\n\n" + "\n".join(menu)

def next_buttons(day_key, meal_key):
    # 요청한 UX: 해당 식사만 먼저 보여주고, 다음 것들은 버튼으로
    if day_key == "today" and meal_key == "breakfast":
        return {"inline_keyboard":[[
            {"text":"점심 전체", "callback_data":"view|today|lunch"},
            {"text":"저녁 전체", "callback_data":"view|today|dinner"},
        ]]}
    if day_key == "today" and meal_key == "lunch":
        return {"inline_keyboard":[[
            {"text":"저녁 전체", "callback_data":"view|today|dinner"},
        ]]}
    if day_key == "today" and meal_key == "dinner":
        return {"inline_keyboard":[[
            {"text":"내일 아침", "callback_data":"view|tomorrow|breakfast"},
        ]]}
    return {"inline_keyboard": []}

@app.route("/", methods=["POST"])
def webhook():
    data = request.json or {}

    # 1) 메시지(사진 업로드 등)
    if "message" in data:
        msg = data["message"]
        user_id = msg.get("from", {}).get("id")

        # 관리자만 사진 업로드 허용
        if "photo" in msg:
            if user_id != ADMIN_ID:
                # 권한 없는 사람은 조용히 무시(또는 안내 메시지 보내도 됨)
                return "ok"

            # 아직 OCR 전이라 "저장됨"만 알려주기 + 버튼 테스트용 메시지
            keyboard = {"inline_keyboard":[[
                {"text":"전체 메뉴 보기(테스트)", "callback_data":"view|today|lunch"}
            ]]}
            send_message("📅 식단표 업로드 완료! (버튼 테스트용)", keyboard=keyboard)
            return "ok"

    # 2) 버튼 클릭(callback_query)
    if "callback_query" in data:
        q = data["callback_query"]
        cb = q.get("data","")
        msg = q.get("message", {})
        chat_id = msg.get("chat", {}).get("id")
        message_id = msg.get("message_id")

        if cb.startswith("view|"):
            _, day_key, meal_key = cb.split("|", 2)
            text = format_meal_full(day_key, meal_key)
            keyboard = next_buttons(day_key, meal_key)
            edit_message(chat_id, message_id, text, keyboard=keyboard)

    return "ok"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
