from flask import Flask, request
import requests
import os

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

app = Flask(__name__)

@app.route("/", methods=["POST"])
def webhook():
    data = request.json

    if "message" in data:
        if "photo" in data["message"]:
            requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                data={
                    "chat_id": CHAT_ID,
                    "text": "✅ 식단표 저장 완료!"
                }
            )

    return "ok"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
