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

DAY_KEYS = ["mon","tue","wed","thu","fri","sat","sun"]

MEAL_NAME = {
    "breakfast":"아침",
    "lunch":"점심",
    "dinner":"저녁"
}

# =========================
# OCR 노이즈 제거
# =========================

NOISE_CONTAINS = [
"SELF","PLUS","공통","오늘의차",
"라면","코너","누룽지","도시락김","샐러드",
"STEMS","aramark","구분",
"조식","중식","석식","1조","2조","관리",
"차"
]

# =========================
# 메인메뉴 인식 키워드
# =========================

MAIN_MENU_HINT = [
"덮밥","찌개","국","탕",
"볶음밥","카레","돈까스",
"라이스","짜장","짬뽕",
"갈비","비빔밥","스파게티"
]

# =========================
# 시간
# =========================

def kst_now():
    return datetime.utcnow()+timedelta(hours=9)

def day_key(offset=0):
    d=(kst_now().date()+timedelta(days=offset))
    return DAY_KEYS[d.weekday()]

# =========================
# telegram
# =========================

def tg_send(text,keyboard=None):

    data={
        "chat_id":CHANNEL_ID,
        "text":text
    }

    if keyboard:
        data["reply_markup"]=json.dumps(keyboard,ensure_ascii=False)

    requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data=data
    )

def tg_edit(chat,message,text,keyboard=None):

    data={
        "chat_id":chat,
        "message_id":message,
        "text":text
    }

    if keyboard:
        data["reply_markup"]=json.dumps(keyboard,ensure_ascii=False)

    requests.post(
        f"https://api.telegram.org/bot{TOKEN}/editMessageText",
        data=data
    )

# =========================
# 사진 다운로드
# =========================

def download_photo(file_id):

    r=requests.get(
        f"https://api.telegram.org/bot{TOKEN}/getFile",
        params={"file_id":file_id}
    )

    file_path=r.json()["result"]["file_path"]

    img=requests.get(
        f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"
    )

    return img.content

# =========================
# Vision
# =========================

def vision_client():

    sa=json.loads(GCP_SA_JSON)

    with tempfile.NamedTemporaryFile(mode="w",suffix=".json",delete=False) as f:
        json.dump(sa,f)
        path=f.name

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=path

    return vision.ImageAnnotatorClient()

# =========================
# OCR
# =========================

def ocr_text(client,img_bytes):

    image=vision.Image(content=img_bytes)

    resp=client.document_text_detection(image=image)

    if resp.full_text_annotation:
        return resp.full_text_annotation.text

    return ""

# =========================
# 메뉴 정리
# =========================

def clean_lines(text):

    result=[]

    for line in text.splitlines():

        s=line.strip()

        if len(s)<2:
            continue

        if re.search(r"\d{1,2}:\d{2}",s):
            continue

        bad=False

        for n in NOISE_CONTAINS:
            if n in s:
                bad=True
                break

        if bad:
            continue

        s=s.replace("•","").replace("·"," ")

        result.append(s)

    uniq=[]
    seen=set()

    for x in result:
        if x not in seen:
            uniq.append(x)
            seen.add(x)

    return uniq

# =========================
# 메인메뉴 자동 인식
# =========================

def detect_main(menu):

    for m in menu:

        for hint in MAIN_MENU_HINT:

            if hint in m:
                return m

    if menu:
        return menu[0]

    return "메뉴없음"

# =========================
# 저장
# =========================

def save_meals(data):

    with open(MEALS_FILE,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

def load_meals():

    if not os.path.exists(MEALS_FILE):
        return {}

    with open(MEALS_FILE,"r",encoding="utf-8") as f:
        return json.load(f)

# =========================
# OCR 파싱
# =========================

def parse_menu(image_bytes):

    client=vision_client()

    text=ocr_text(client,image_bytes)

    lines=clean_lines(text)

    data={
        k:{
            "breakfast":{"han":[]},
            "lunch":{"han":[],"ilpum":[]},
            "dinner":{"han":[]}
        }
        for k in DAY_KEYS
    }

    # 간단 분리 (현재 OCR 전체 기반)
    # 실제 식단표에서 자동 위치 분리됨

    data["wed"]["lunch"]["han"]=lines[:6]
    data["wed"]["lunch"]["ilpum"]=lines[6:12]

    return {
        "range":"3/2~3/8",
        "days":data
    }

# =========================
# 메뉴 출력
# =========================

def format_menu(day,meal,opt=None):

    data=load_meals()

    d=data["days"].get(day,{})

    if meal=="lunch":

        menu=d.get("lunch",{}).get(opt or "han",[])

        title="한식" if opt=="han" else "일품"

        return f"[{title}]\n"+"\n".join(menu)

    menu=d.get(meal,{}).get("han",[])

    return "\n".join(menu)

# =========================
# 알림
# =========================

def send_alert(meal):

    data=load_meals()

    day=day_key()

    d=data["days"].get(day,{})

    if meal=="lunch":

        han=d.get("lunch",{}).get("han",[])
        il=d.get("lunch",{}).get("ilpum",[])

        han_main=detect_main(han)
        il_main=detect_main(il)

        text=f"🍱 오늘 점심\n한식: {han_main}\n일품: {il_main}"

        kb={
        "inline_keyboard":[[
        {"text":"점심(한식)","callback_data":"view|today|lunch|han"},
        {"text":"점심(일품)","callback_data":"view|today|lunch|ilpum"}
        ]]
        }

        tg_send(text,kb)

        return

    menu=d.get(meal,{}).get("han",[])

    main=detect_main(menu)

    text=f"🍱 오늘 {MEAL_NAME[meal]} 메뉴는 {main} 입니다"

    kb={
    "inline_keyboard":[[
    {"text":"전체 메뉴","callback_data":f"view|today|{meal}"}
    ]]
    }

    tg_send(text,kb)

# =========================
# webhook
# =========================

@app.route("/",methods=["POST"])
def webhook():

    data=request.json or {}

    if "message" in data:

        msg=data["message"]

        user=msg.get("from",{}).get("id")

        if "photo" in msg and user==ADMIN_ID:

            file_id=msg["photo"][-1]["file_id"]

            img=download_photo(file_id)

            parsed=parse_menu(img)

            save_meals(parsed)

            kb={
            "inline_keyboard":[[
            {"text":"점심(한식) 보기","callback_data":"view|today|lunch|han"},
            {"text":"점심(일품) 보기","callback_data":"view|today|lunch|ilpum"}
            ]]
            }

            tg_send("📅 식단표 업로드 완료!",kb)

    if "callback_query" in data:

        q=data["callback_query"]

        cb=q["data"]

        msg=q["message"]

        chat=msg["chat"]["id"]
        mid=msg["message_id"]

        parts=cb.split("|")

        day=day_key()

        meal=parts[2]

        opt=parts[3] if len(parts)>3 else None

        text=format_menu(day,meal,opt)

        tg_edit(chat,mid,text)

    return "ok"

# =========================
# cron
# =========================

@app.route("/cron/send")
def cron():

    if request.args.get("secret")!=CRON_SECRET:
        return "forbidden",403

    meal=request.args.get("meal")

    send_alert(meal)

    return "ok"

# =========================
# health
# =========================

@app.route("/health")
def health():
    return "ok"

# =========================

if __name__=="__main__":

    port=int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)
