from flask import Flask, request
import os
import io
import re
import json
import requests
import tempfile
from datetime import datetime, timedelta
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
CRON_SECRET = os.getenv("CRON_SECRET", "")
GCP_SA_JSON = os.getenv("GCP_SA_JSON")

PORT = int(os.environ.get("PORT", 10000))

app = Flask(__name__)
MEALS_FILE = "meals.json"

DAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

MEAL_NAME = {
    "breakfast": "아침",
    "lunch": "점심",
    "dinner": "저녁",
}

# =========================================================
# TEMPLATE RATIOS (based on your samples)
# =========================================================

X_REL = np.array([126, 253, 380, 507, 634, 761, 888, 1014], dtype=np.float32) / 1015.0

Y_DATE_REL = np.array([61, 84], dtype=np.float32) / 797.0
Y_BREAKFAST_REL = np.array([84, 208], dtype=np.float32) / 797.0
Y_LUNCH_REL = np.array([208, 422], dtype=np.float32) / 797.0
Y_DINNER_REL = np.array([422, 505], dtype=np.float32) / 797.0  # dinner han area only

LEFT_GUIDE_WIDTH_RATIO = 0.18
FALLBACK_SPLIT_RATIO = 85.0 / 214.0
FALLBACK_LOWER_END_RATIO = 0.79

# =========================================================
# OCR CLEANUP
# =========================================================

NOISE_EXACT = {"한식", "일품", "공통", "SELF", "PLUS", "오늘의차"}
NOISE_CONTAINS = [
    "공통", "SELF", "PLUS", "오늘의차",
    "라면", "코너", "누룽지", "도시락김", "샐러드", "그린",
    "STEMS", "aramark", "구분",
    "조식", "중식", "석식", "1조", "2조", "관리",
    "차", "우유", "요거트",
]

MAIN_HINTS = [
    "국", "찌개", "탕", "전골", "순두부", "육개장", "설렁탕", "감자탕", "부대찌개", "미역국", "된장",
    "덮밥", "비빔밥", "볶음밥", "카레", "짜장", "짬뽕", "우동", "국수", "라면",
    "돈까스", "스테이크", "파스타", "스파게티", "피자", "햄버거",
    "갈비", "불고기", "제육", "닭갈비", "찜", "구이", "조림", "볶음",
]

NOT_MAIN = [
    "김치", "깍두기", "단무지", "나물", "샐러드", "도시락김",
    "밥", "누룽지", "소스", "드레싱", "피클"
]

# =========================================================
# BASIC UTILS
# =========================================================

def kst_now():
    return datetime.utcnow() + timedelta(hours=9)

def today_key():
    return DAY_KEYS[kst_now().weekday()]

def day_key_with_offset(offset_days: int = 0) -> str:
    target = (kst_now() + timedelta(days=offset_days)).date()
    return DAY_KEYS[target.weekday()]

def escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

# =========================================================
# STORAGE
# =========================================================

def load_meals():
    if not os.path.exists(MEALS_FILE):
        return {"days": {}, "range": ""}
    try:
        with open(MEALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "days" not in data:
            data["days"] = {}
        if "range" not in data:
            data["range"] = ""
        return data
    except Exception:
        return {"days": {}, "range": ""}

def save_meals(data):
    with open(MEALS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =========================================================
# TELEGRAM
# =========================================================

def tg_send(text: str, chat_id: Optional[str] = None):
    data = {
        "chat_id": chat_id or CHANNEL_ID,
        "text": text,
        "parse_mode": "HTML"
    }

    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data=data,
        timeout=20
    )
    r.raise_for_status()

def tg_send_photo(photo_bytes: bytes, caption: str = "", chat_id: Optional[str] = None):
    data = {
        "chat_id": chat_id or CHANNEL_ID,
        "caption": caption,
        "parse_mode": "HTML",
    }

    files = {
        "photo": ("menu.jpg", photo_bytes, "image/jpeg")
    }

    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendPhoto",
        data=data,
        files=files,
        timeout=60
    )
    r.raise_for_status()

def download_photo(file_id):
    r = requests.get(
        f"https://api.telegram.org/bot{TOKEN}/getFile",
        params={"file_id": file_id},
        timeout=20
    )
    r.raise_for_status()
    file_path = r.json()["result"]["file_path"]

    img = requests.get(
        f"https://api.telegram.org/file/bot{TOKEN}/{file_path}",
        timeout=30
    )
    img.raise_for_status()
    return img.content

# =========================================================
# VISION OCR
# =========================================================

def vision_client():
    sa = json.loads(GCP_SA_JSON)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sa, f)
        path = f.name

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
    return vision.ImageAnnotatorClient()

def ocr_text(client, img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    image = vision.Image(content=buf.getvalue())
    resp = client.document_text_detection(image=image)

    if resp.full_text_annotation:
        return resp.full_text_annotation.text

    return ""

# =========================================================
# TEXT PROCESS
# =========================================================

def clean_lines(text):
    lines = []

    for l in text.splitlines():
        s = l.strip()

        if not s:
            continue
        if len(s) <= 1:
            continue
        if s in NOISE_EXACT:
            continue
        if any(x in s for x in NOISE_CONTAINS):
            continue
        if re.search(r"\d{1,2}:\d{2}", s):
            continue

        lines.append(s)

    out = []
    seen = set()
    for x in lines:
        if x not in seen:
            out.append(x)
            seen.add(x)

    return out

def score_main(line):
    s = 0

    if any(x in line for x in MAIN_HINTS):
        s += 5

    if any(x in line for x in NOT_MAIN):
        s -= 5

    return s

def pick_main(lines):
    if not lines:
        return ""

    scored = [(score_main(x), i, x) for i, x in enumerate(lines)]
    scored.sort(reverse=True, key=lambda t: (t[0], -t[1]))
    return scored[0][2]

def bold_main(lines: List[str]):
    if not lines:
        return "(메뉴 없음)"

    main = pick_main(lines)
    out = []

    for line in lines:
        safe = escape_html(line)
        if line == main:
            out.append(f"<b>{safe}</b>")
        else:
            out.append(safe)

    return "\n".join(out)

# =========================================================
# IMAGE HELPERS
# =========================================================

def crop_boxes(img, rel):
    w, h = img.size
    xs = [int(r * w) for r in X_REL]

    y1 = int(rel[0] * h)
    y2 = int(rel[1] * h)

    boxes = []
    for i in range(7):
        boxes.append((xs[i], y1, xs[i + 1], y2))

    return boxes

def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# =========================================================
# LUNCH SPLIT
# =========================================================

def detect_left_lunch_boundaries(img: Image.Image):
    cv_img = pil_to_cv(img)
    h, w = cv_img.shape[:2]

    guide_w = max(10, int(w * LEFT_GUIDE_WIDTH_RATIO))
    guide = cv_img[:, :guide_w]

    gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10
    )

    kernel_len = max(10, guide_w // 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, kernel, iterations=1)

    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ys = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > guide_w * 0.5 and ch < max(4, int(h * 0.06)):
            ys.append(y + ch // 2)

    ys = sorted(set(ys))

    target1 = int(h * 0.40)
    target2 = int(h * 0.79)

    cand1 = min(ys, key=lambda y: abs(y - target1)) if ys else None
    cand2 = min(ys, key=lambda y: abs(y - target2)) if ys else None

    if cand1 is not None and cand2 is not None and cand2 > cand1:
        return cand1, cand2

    split1 = int(round(h * FALLBACK_SPLIT_RATIO))
    split2 = int(round(h * FALLBACK_LOWER_END_RATIO))

    return split1, split2

def detect_blue_bands(img: Image.Image):
    cv_img = pil_to_cv(img)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

    lower = np.array([85, 30, 30], dtype=np.uint8)
    upper = np.array([145, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    row_sum = mask.sum(axis=1).astype(np.float32)

    if row_sum.max() <= 0:
        return 0, []

    row_sum /= row_sum.max()
    active_rows = np.where(row_sum > 0.16)[0].tolist()

    if not active_rows:
        return 0, []

    groups = [[int(active_rows[0])]]

    for y in active_rows[1:]:
        y = int(y)
        if y - groups[-1][-1] <= 5:
            groups[-1].append(y)
        else:
            groups.append([y])

    centers = []
    h = img.size[1]
    min_group_height = max(3, int(h * 0.012))

    for g in groups:
        if len(g) >= min_group_height:
            centers.append(int(sum(g) / len(g)))

    return len(centers), centers

def analyze_lunch_box(lunch_img: Image.Image):
    split1_y, split2_y = detect_left_lunch_boundaries(lunch_img)

    w, h = lunch_img.size

    upper = lunch_img.crop((0, 0, w, split1_y))
    lower = lunch_img.crop((0, split1_y, w, split2_y))

    upper_count, upper_centers = detect_blue_bands(upper)
    lower_count, lower_centers = detect_blue_bands(lower)

    mode = "dual" if (upper_count >= 1 and lower_count >= 1) else "single"

    return {
        "mode": mode,
        "split_y": split1_y,
        "lower_end_y": split2_y,
        "upper_img": upper,
        "lower_img": lower,
        "upper_blue_count": upper_count,
        "lower_blue_count": lower_count,
        "upper_blue_centers": upper_centers,
        "lower_blue_centers": lower_centers,
    }

# =========================================================
# DATE HEADER
# =========================================================

def extract_mmdd_from_text(text):
    patterns = [
        r"(\d{1,2})\s*월\s*(\d{1,2})\s*일",
        r"(\d{1,2})\s*/\s*(\d{1,2})",
        r"(\d{1,2})\s*\.\s*(\d{1,2})",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            return int(m.group(1)), int(m.group(2))

    return None

def build_range(date_boxes, img, client):
    dates = {}

    for dk, box in zip(DAY_KEYS, date_boxes):
        cell = img.crop(box)
        text = ocr_text(client, cell)
        mmdd = extract_mmdd_from_text(text)
        if mmdd:
            dates[dk] = mmdd

    if "mon" not in dates:
        return ""

    mon = dates["mon"]
    sun = dates.get("sun", mon)

    return f"{mon[0]}/{mon[1]}(월) ~ {sun[0]}/{sun[1]}(일)"

# =========================================================
# PARSE WEEK IMAGE
# =========================================================

def parse_week(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    client = vision_client()

    date_boxes = crop_boxes(img, Y_DATE_REL)
    breakfast_boxes = crop_boxes(img, Y_BREAKFAST_REL)
    lunch_boxes = crop_boxes(img, Y_LUNCH_REL)
    dinner_boxes = crop_boxes(img, Y_DINNER_REL)

    meal_range = build_range(date_boxes, img, client)

    days = {}

    for i, dk in enumerate(DAY_KEYS):
        bf_img = img.crop(breakfast_boxes[i])
        bf_lines = clean_lines(ocr_text(client, bf_img))

        lunch_img = img.crop(lunch_boxes[i])
        lunch_info = analyze_lunch_box(lunch_img)

        han_lines = clean_lines(ocr_text(client, lunch_info["upper_img"]))
        il_lines = clean_lines(ocr_text(client, lunch_info["lower_img"]))

        mode = lunch_info["mode"]

        if mode != "dual":
            il_lines = []

        dn_img = img.crop(dinner_boxes[i])
        dn_lines = clean_lines(ocr_text(client, dn_img))

        days[dk] = {
            "breakfast": {
                "han": bf_lines
            },
            "lunch": {
                "mode": mode,
                "han": han_lines,
                "ilpum": il_lines,
                "han_main": pick_main(han_lines),
                "ilpum_main": pick_main(il_lines) if il_lines else ""
            },
            "dinner": {
                "han": dn_lines
            }
        }

    return {
        "range": meal_range,
        "days": days
    }

# =========================================================
# FORMATTERS (stored data only)
# =========================================================

def format_lunch_by_day(day_offset: int):
    data = load_meals()
    dk = day_key_with_offset(day_offset)
    day = data.get("days", {}).get(dk, {})
    lunch = day.get("lunch", {})
    rng = data.get("range", "")

    label = "오늘" if day_offset == 0 else "내일"
    header = f"📋 {label} 점심 메뉴"
    if rng:
        header += f"\n(저장된 식단표: {escape_html(rng)})"

    if lunch.get("mode") == "dual":
        return (
            f"{header}\n\n"
            f"[한식]\n{bold_main(lunch.get('han', []))}\n\n"
            f"[일품]\n{bold_main(lunch.get('ilpum', []))}"
        )

    return (
        f"{header}\n\n"
        f"{bold_main(lunch.get('han', []))}"
    )

def format_meal_by_day(day_offset: int, meal: str):
    data = load_meals()
    dk = day_key_with_offset(day_offset)
    day = data.get("days", {}).get(dk, {})
    rng = data.get("range", "")

    label = "오늘" if day_offset == 0 else "내일"

    if meal == "lunch":
        return format_lunch_by_day(day_offset)

    menu = day.get(meal, {}).get("han", [])
    header = f"📋 {label} {MEAL_NAME[meal]} 메뉴"
    if rng:
        header += f"\n(저장된 식단표: {escape_html(rng)})"

    return f"{header}\n\n{bold_main(menu)}"

# =========================================================
# STRICT USER COMMANDS (private chat only)
# =========================================================

def parse_user_question(text: str):
    s = text.strip().replace(" ", "").replace('"', "").replace("“", "").replace("”", "")

    mapping = {
        "오늘아침": (0, "breakfast"),
        "오늘점심": (0, "lunch"),
        "오늘저녁": (0, "dinner"),
        "내일아침": (1, "breakfast"),
        "내일점심": (1, "lunch"),
        "내일저녁": (1, "dinner"),
    }

    return mapping.get(s)

# =========================================================
# WEBHOOK
# =========================================================

@app.route("/", methods=["POST"])
def webhook():
    data = request.json or {}

    incoming = None
    if "message" in data:
        incoming = data["message"]
    elif "channel_post" in data:
        incoming = data["channel_post"]

    if incoming:
        chat_type = incoming.get("chat", {}).get("type", "")

        # 관리자 사진 업로드 -> OCR/이미지 처리 여기서만
        if "photo" in incoming:
            if incoming.get("from", {}).get("id") != ADMIN_ID:
                return "ok"

            try:
                file_id = incoming["photo"][-1]["file_id"]
                img_bytes = download_photo(file_id)

                parsed = parse_week(img_bytes)
                save_meals(parsed)

                # 관리자에게는 텍스트만
                tg_send("📅 식단표 저장 완료", chat_id=incoming["chat"]["id"])

                # 채널에는 사진 + 캡션만
                rng = escape_html(parsed.get("range", ""))
                caption = "📅 <b>식단표 저장 완료</b>"
                if rng:
                    caption += f"\n{rng}"

                tg_send_photo(
                    photo_bytes=img_bytes,
                    caption=caption,
                    chat_id=CHANNEL_ID
                )

            except Exception as e:
                tg_send(
                    f"⚠️ 업로드/파싱 오류: {escape_html(type(e).__name__)}",
                    chat_id=incoming["chat"]["id"]
                )

            return "ok"

        # 텍스트 질의 -> private chat에서만 허용
        if "text" in incoming:
            if chat_type != "private":
                return "ok"

            parsed_q = parse_user_question(incoming["text"])
            if not parsed_q:
                return "ok"

            offset_days, meal = parsed_q
            reply = format_meal_by_day(offset_days, meal)

            tg_send(reply, chat_id=incoming["chat"]["id"])
            return "ok"

    return "ok"

# =========================================================
# CRON (stored values only)
# =========================================================

@app.route("/cron/send")
def cron():
    if request.args.get("secret") != CRON_SECRET:
        return "forbidden", 403

    meal = request.args.get("meal")

    if meal == "breakfast":
        tg_send(format_meal_by_day(0, "breakfast"))
    elif meal == "lunch":
        tg_send(format_meal_by_day(0, "lunch"))
    elif meal == "dinner":
        tg_send(format_meal_by_day(0, "dinner"))
    else:
        return "bad meal", 400

    return "ok"

# =========================================================
# HEALTH
# =========================================================

@app.route("/health")
def health():
    return "ok"

# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
