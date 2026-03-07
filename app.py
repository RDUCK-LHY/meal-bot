from flask import Flask, request
import os
import io
import re
import json
import requests
import tempfile
from datetime import datetime, timedelta, date
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from google.cloud import vision


# =========================================================
# ENV
# =========================================================

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
CRON_SECRET = os.getenv("CRON_SECRET","")
GCP_SA_JSON = os.getenv("GCP_SA_JSON")

PORT = int(os.environ.get("PORT",10000))

app = Flask(__name__)

MEALS_FILE = "meals.json"

DAY_KEYS = ["mon","tue","wed","thu","fri","sat","sun"]

MEAL_NAME = {
"breakfast":"아침",
"lunch":"점심",
"dinner":"저녁"
}


# =========================================================
# 식단표 좌표 (샘플 기준)
# =========================================================

X_REL = np.array([126,253,380,507,634,761,888,1014]) / 1015.0

Y_DATE_REL = np.array([61,84]) / 797.0
Y_BREAKFAST_REL = np.array([84,208]) / 797.0
Y_LUNCH_REL = np.array([208,422]) / 797.0

# 저녁은 한식 영역만
Y_DINNER_REL = np.array([422,505]) / 797.0


# =========================================================
# MAIN MENU KEYWORDS
# =========================================================

MAIN_HINTS = [
"국","찌개","탕","덮밥","비빔밥","볶음밥",
"카레","짜장","짬뽕","우동","국수",
"돈까스","스테이크","파스타",
"갈비","불고기","제육","닭갈비"
]


NOT_MAIN = [
"김치","깍두기","단무지","나물",
"샐러드","도시락김","밥","누룽지"
]


# =========================================================
# util
# =========================================================

def kst_now():
    return datetime.utcnow() + timedelta(hours=9)


def today_key():
    return DAY_KEYS[kst_now().weekday()]


# =========================================================
# storage
# =========================================================

def load_meals():

    if not os.path.exists(MEALS_FILE):
        return {"days":{}}

    with open(MEALS_FILE,"r",encoding="utf-8") as f:
        return json.load(f)


def save_meals(data):

    with open(MEALS_FILE,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)


# =========================================================
# telegram
# =========================================================

def tg_send(text,keyboard=None):

    data = {
    "chat_id":CHANNEL_ID,
    "text":text,
    "parse_mode":"HTML"
    }

    if keyboard:
        data["reply_markup"] = json.dumps(keyboard)

    requests.post(
    f"https://api.telegram.org/bot{TOKEN}/sendMessage",
    data=data
    )


def tg_edit(chat,message,text,keyboard=None):

    data = {
    "chat_id":chat,
    "message_id":message,
    "text":text,
    "parse_mode":"HTML"
    }

    if keyboard:
        data["reply_markup"] = json.dumps(keyboard)

    requests.post(
    f"https://api.telegram.org/bot{TOKEN}/editMessageText",
    data=data
    )


# =========================================================
# vision
# =========================================================

def vision_client():

    sa=json.loads(GCP_SA_JSON)

    with tempfile.NamedTemporaryFile(mode="w",suffix=".json",delete=False) as f:

        json.dump(sa,f)
        path=f.name

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=path

    return vision.ImageAnnotatorClient()


def ocr_text(client,img):

    buf=io.BytesIO()

    img.save(buf,format="PNG")

    image = vision.Image(content=buf.getvalue())

    resp = client.document_text_detection(image=image)

    if resp.full_text_annotation:
        return resp.full_text_annotation.text

    return ""


# =========================================================
# text clean
# =========================================================

def clean_lines(text):

    lines=[]

    for l in text.splitlines():

        s=l.strip()

        if not s:
            continue

        if len(s)<=1:
            continue

        if any(x in s for x in ["SELF","PLUS","오늘의차","공통"]):
            continue

        lines.append(s)

    return lines


# =========================================================
# main menu detection
# =========================================================

def score_main(line):

    s=0

    if any(x in line for x in MAIN_HINTS):
        s+=5

    if any(x in line for x in NOT_MAIN):
        s-=5

    return s


def pick_main(lines):

    if not lines:
        return ""

    scored=[(score_main(x),x) for x in lines]

    scored.sort(reverse=True)

    return scored[0][1]


def bold_main(lines):

    main=pick_main(lines)

    out=[]

    for l in lines:

        if l==main:
            out.append(f"<b>{l}</b>")
        else:
            out.append(l)

    return "\n".join(out)


# =========================================================
# image helpers
# =========================================================

def crop_boxes(img,rel):

    w,h=img.size

    xs=[int(r*w) for r in X_REL]

    y1=int(rel[0]*h)
    y2=int(rel[1]*h)

    boxes=[]

    for i in range(7):

        boxes.append((xs[i],y1,xs[i+1],y2))

    return boxes


# =========================================================
# lunch split
# =========================================================

def detect_lunch_split(img):

    cv=cv2.cvtColor(np.array(img),cv2.COLOR_RGB2GRAY)

    edges=cv2.Canny(cv,50,150)

    lines=cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)

    if lines is None:
        return img.size[1]//2

    ys=[]

    for l in lines:

        x1,y1,x2,y2=l[0]

        if abs(y1-y2)<5:
            ys.append(y1)

    if not ys:
        return img.size[1]//2

    ys.sort()

    return ys[len(ys)//2]


# =========================================================
# parse image
# =========================================================

def parse_week(image_bytes):

    img=Image.open(io.BytesIO(image_bytes)).convert("RGB")

    client=vision_client()

    breakfast_boxes=crop_boxes(img,Y_BREAKFAST_REL)

    lunch_boxes=crop_boxes(img,Y_LUNCH_REL)

    dinner_boxes=crop_boxes(img,Y_DINNER_REL)

    days={}

    for i,dk in enumerate(DAY_KEYS):

        # breakfast

        bf_img=img.crop(breakfast_boxes[i])

        bf_lines=clean_lines(ocr_text(client,bf_img))

        # lunch

        lunch_img=img.crop(lunch_boxes[i])

        split=detect_lunch_split(lunch_img)

        upper=lunch_img.crop((0,0,lunch_img.size[0],split))

        lower=lunch_img.crop((0,split,lunch_img.size[0],lunch_img.size[1]))

        han_lines=clean_lines(ocr_text(client,upper))

        il_lines=clean_lines(ocr_text(client,lower))

        mode="dual" if il_lines else "single"

        # dinner

        dn_img=img.crop(dinner_boxes[i])

        dn_lines=clean_lines(ocr_text(client,dn_img))

        days[dk]={

        "breakfast":{
        "han":bf_lines
        },

        "lunch":{
        "mode":mode,
        "han":han_lines,
        "ilpum":il_lines
        },

        "dinner":{
        "han":dn_lines
        }

        }

    return {"days":days}


# =========================================================
# menu format
# =========================================================

def format_lunch():

    data=load_meals()

    day=data["days"].get(today_key(),{})

    lunch=day.get("lunch",{})

    if lunch.get("mode")=="dual":

        return (
        "📋 오늘 점심 메뉴\n\n"
        "[한식]\n"
        +bold_main(lunch.get("han",[]))
        +"\n\n[일품]\n"
        +bold_main(lunch.get("ilpum",[]))
        )

    return (
    "📋 오늘 점심 메뉴\n\n"
    +bold_main(lunch.get("han",[]))
    )


def format_meal(meal):

    data=load_meals()

    day=data["days"].get(today_key(),{})

    menu=day.get(meal,{}).get("han",[])

    return (
    f"📋 오늘 {MEAL_NAME[meal]} 메뉴\n\n"
    +bold_main(menu)
    )


# =========================================================
# webhook
# =========================================================

@app.route("/",methods=["POST"])

def webhook():

    data=request.json

    if "message" in data:

        msg=data["message"]

        if "photo" in msg:

            if msg["from"]["id"]!=ADMIN_ID:
                return "ok"

            file_id=msg["photo"][-1]["file_id"]

            r=requests.get(
            f"https://api.telegram.org/bot{TOKEN}/getFile",
            params={"file_id":file_id}
            )

            file_path=r.json()["result"]["file_path"]

            img=requests.get(
            f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"
            ).content

            parsed=parse_week(img)

            save_meals(parsed)

            tg_send("📅 식단표 저장 완료")

    if "callback_query" in data:

        q=data["callback_query"]

        cmd=q["data"]

        chat=q["message"]["chat"]["id"]

        mid=q["message"]["message_id"]

        if cmd=="lunch":

            tg_edit(chat,mid,format_lunch())

        if cmd=="dinner":

            tg_edit(chat,mid,format_meal("dinner"))

    return "ok"


# =========================================================
# cron
# =========================================================

@app.route("/cron/send")

def cron():

    if request.args.get("secret")!=CRON_SECRET:
        return "forbidden"

    meal=request.args.get("meal")

    if meal=="breakfast":
        tg_send(format_meal("breakfast"))

    if meal=="lunch":
        tg_send(format_lunch())

    if meal=="dinner":
        tg_send(format_meal("dinner"))

    return "ok"


# =========================================================

if __name__=="__main__":

    app.run(host="0.0.0.0",port=PORT)
