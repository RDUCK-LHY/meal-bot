from flask import Flask, request
import os
import io
import re
import json
import tempfile
from datetime import datetime, timedelta, date
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
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
DEBUG_DIR = "debug_output"
os.makedirs(DEBUG_DIR, exist_ok=True)

DAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
DAY_LABELS = ["월", "화", "수", "목", "금", "토", "일"]

MEAL_NAME = {
    "breakfast": "아침",
    "lunch": "점심",
    "dinner": "저녁",
}

# =========================================================
# 샘플 식단표 기준 비율
# =========================================================
# 날짜 7칸 x 경계
X_REL = np.array([126, 253, 380, 507, 634, 761, 888, 1014], dtype=np.float32) / 1015.0

# 큰 행 경계
Y_DATE_REL = np.array([61, 84], dtype=np.float32) / 797.0
Y_BREAKFAST_REL = np.array([84, 208], dtype=np.float32) / 797.0
Y_LUNCH_REL = np.array([208, 422], dtype=np.float32) / 797.0
Y_DINNER_REL = np.array([422, 529], dtype=np.float32) / 797.0

LEFT_GUIDE_WIDTH_RATIO = 0.18
FALLBACK_SPLIT_RATIO = 85.0 / 214.0  # lunch 높이 기준 split fallback

# =========================================================
# OCR / cleanup rules
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
NOT_MAIN_HINTS = [
    "김치", "깍두기", "단무지", "무말랭이", "나물", "샐러드", "도시락김",
    "밥", "현미밥", "잡곡밥", "누룽지", "차", "과일",
    "소스", "케찹", "드레싱", "피클", "초장", "양념장",
]

# =========================================================
# basic utils
# =========================================================
def kst_now() -> datetime:
    return datetime.utcnow() + timedelta(hours=9)

def today_kst() -> date:
    return kst_now().date()

def ensure_env():
    missing = []
    if not TOKEN:
        missing.append("TELEGRAM_TOKEN")
    if not CHANNEL_ID:
        missing.append("CHANNEL_ID")
    if not GCP_SA_JSON:
        missing.append("GCP_SA_JSON")
    if missing:
        raise RuntimeError("Missing env: " + ", ".join(missing))

def get_font(size: int = 14):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size,
        )
    except Exception:
        return ImageFont.load_default()

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def scale_box(box: Tuple[float, float], img_h: int) -> Tuple[int, int]:
    return int(round(box[0] * img_h)), int(round(box[1] * img_h))

# =========================================================
# telegram
# =========================================================
def tg_send(text: str, keyboard: Optional[dict] = None):
    data = {
        "chat_id": CHANNEL_ID,
        "text": text,
    }
    if keyboard:
        data["reply_markup"] = json.dumps(keyboard, ensure_ascii=False)

    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data=data,
        timeout=20,
    )
    r.raise_for_status()

def tg_edit(chat_id, message_id, text: str, keyboard: Optional[dict] = None):
    data = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
    }
    if keyboard:
        data["reply_markup"] = json.dumps(keyboard, ensure_ascii=False)

    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/editMessageText",
        data=data,
        timeout=20,
    )
    r.raise_for_status()

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

# =========================================================
# google vision
# =========================================================
def make_vision_client() -> vision.ImageAnnotatorClient:
    sa = json.loads(GCP_SA_JSON)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sa, f)
        sa_path = f.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
    return vision.ImageAnnotatorClient()

def pil_to_png_bytes(img: Image.Image, scale: int = 2) -> bytes:
    resized = img.resize((img.size[0] * scale, img.size[1] * scale))
    buf = io.BytesIO()
    resized.save(buf, format="PNG")
    return buf.getvalue()

def ocr_text(client: vision.ImageAnnotatorClient, img: Image.Image, scale: int = 2) -> str:
    content = pil_to_png_bytes(img, scale=scale)
    image = vision.Image(content=content)
    resp = client.document_text_detection(image=image)
    if resp.error and resp.error.message:
        raise RuntimeError(resp.error.message)
    if resp.full_text_annotation and resp.full_text_annotation.text:
        return resp.full_text_annotation.text
    return ""

# =========================================================
# text postprocess
# =========================================================
def clean_lines(text: str) -> List[str]:
    lines: List[str] = []

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
        if any(tok in s for tok in NOISE_CONTAINS):
            continue

        s = s.replace("•", "").replace("·", " ").replace("|", " ").strip()
        lines.append(s)

    seen = set()
    uniq = []
    for x in lines:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq

def is_not_main(line: str) -> bool:
    return any(x in line for x in NOT_MAIN_HINTS)

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

def pick_main_menu(lines: List[str]) -> str:
    if not lines:
        return "메뉴 정보 없음"
    scored = [(score_main(x), i, x) for i, x in enumerate(lines)]
    scored.sort(reverse=True, key=lambda t: (t[0], -t[1]))
    best_score, _, best_line = scored[0]
    if best_score <= -3:
        return lines[0]
    return best_line

def normalize_menu_list(lines: List[str]) -> List[str]:
    if not lines:
        return lines
    main = pick_main_menu(lines)
    return [main] + [x for x in lines if x != main]

def extract_mmdd_from_text(text: str) -> Optional[Tuple[int, int]]:
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

# =========================================================
# geometry
# =========================================================
def get_xs(img: Image.Image) -> List[int]:
    w, _ = img.size
    return [int(round(r * w)) for r in X_REL]

def get_row_range(img: Image.Image, rel: np.ndarray) -> Tuple[int, int]:
    _, h = img.size
    return int(round(rel[0] * h)), int(round(rel[1] * h))

def get_day_boxes(img: Image.Image, rel_y: np.ndarray) -> List[Tuple[int, int, int, int]]:
    xs = get_xs(img)
    y1, y2 = get_row_range(img, rel_y)
    return [(xs[i], y1, xs[i+1], y2) for i in range(7)]

# =========================================================
# lunch split logic
# =========================================================
def detect_left_guide_split_line(lunch_img: Image.Image) -> Optional[int]:
    cv_img = pil_to_cv(lunch_img)
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
        10,
    )

    kernel_len = max(10, guide_w // 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, kernel, iterations=1)

    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[Tuple[int, int, int, int, int]] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > guide_w * 0.5 and ch < max(4, int(h * 0.06)):
            center_y = y + ch // 2
            candidates.append((center_y, cw, x, y, ch))

    if not candidates:
        return None

    mid = h / 2.0
    candidates.sort(key=lambda t: (abs(t[0] - mid), -t[1]))
    split_y = candidates[0][0]

    if abs(split_y - mid) < h * 0.28:
        return split_y
    return None

def detect_blue_bands(region_img: Image.Image) -> Tuple[int, List[int]]:
    cv_img = pil_to_cv(region_img)
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

    groups: List[List[int]] = [[int(active_rows[0])]]
    for y in active_rows[1:]:
        y = int(y)
        if y - groups[-1][-1] <= 5:
            groups[-1].append(y)
        else:
            groups.append([y])

    centers: List[int] = []
    h = region_img.size[1]
    min_group_height = max(3, int(h * 0.012))

    for g in groups:
        if len(g) >= min_group_height:
            centers.append(int(sum(g) / len(g)))

    return len(centers), centers

def analyze_lunch_box(lunch_img: Image.Image) -> Dict:
    split_y = detect_left_guide_split_line(lunch_img)

    if split_y is None:
        split_y = int(round(lunch_img.size[1] * FALLBACK_SPLIT_RATIO))
        split_source = "fallback_ratio"
    else:
        split_source = "left_guide_line"

    w, h = lunch_img.size
    upper_box = (0, 0, w, split_y)
    lower_box = (0, split_y, w, h)

    upper_img = lunch_img.crop(upper_box)
    lower_img = lunch_img.crop(lower_box)

    upper_count, upper_centers = detect_blue_bands(upper_img)
    lower_count, lower_centers = detect_blue_bands(lower_img)

    mode = "dual" if (upper_count >= 1 and lower_count >= 1) else "single"

    return {
        "mode": mode,
        "split_source": split_source,
        "split_y": split_y,
        "upper_box": list(upper_box),
        "lower_box": list(lower_box),
        "upper_blue_count": upper_count,
        "lower_blue_count": lower_count,
        "upper_blue_centers": upper_centers,
        "lower_blue_centers": lower_centers,
        "upper_img": upper_img,
        "lower_img": lower_img,
    }

# =========================================================
# parse weekly menu
# =========================================================
def build_range_and_week_start(date_boxes, img: Image.Image, client):
    dates: Dict[str, Tuple[int, int]] = {}

    for dk, box in zip(DAY_KEYS, date_boxes):
        cell = img.crop(box)
        text = ocr_text(client, cell, scale=3)
        mmdd = extract_mmdd_from_text(text)
        if mmdd:
            dates[dk] = mmdd

    today = today_kst()
    year = today.year

    if "mon" not in dates:
        raise RuntimeError("header_date_read_failed")

    mon_m, mon_d = dates["mon"]
    sun_m, sun_d = dates.get("sun", dates["mon"])

    week_start = date(year, mon_m, mon_d)
    range_text = f"{mon_m}/{mon_d}(월) ~ {sun_m}/{sun_d}(일)"
    return range_text, week_start, dates

def parse_weekly(image_bytes: bytes) -> Dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    client = make_vision_client()

    date_boxes = get_day_boxes(img, Y_DATE_REL)
    breakfast_boxes = get_day_boxes(img, Y_BREAKFAST_REL)
    lunch_boxes = get_day_boxes(img, Y_LUNCH_REL)
    dinner_boxes = get_day_boxes(img, Y_DINNER_REL)

    range_text, week_start, header_dates = build_range_and_week_start(date_boxes, img, client)

    out_days = {}
    debug_meta = {}

    for i, dk in enumerate(DAY_KEYS):
        # breakfast
        bf_img = img.crop(breakfast_boxes[i])
        bf_text = ocr_text(client, bf_img, scale=2)
        bf_lines = normalize_menu_list(clean_lines(bf_text))

        # lunch
        lunch_img = img.crop(lunch_boxes[i])
        lunch_info = analyze_lunch_box(lunch_img)

        upper_text = ocr_text(client, lunch_info["upper_img"], scale=2)
        upper_lines = normalize_menu_list(clean_lines(upper_text))

        lunch_data = {
            "mode": lunch_info["mode"],
            "han": upper_lines,
            "ilpum": [],
            "han_main": pick_main_menu(upper_lines) if upper_lines else "메뉴 정보 없음",
            "ilpum_main": "",
        }

        if lunch_info["mode"] == "dual":
            lower_text = ocr_text(client, lunch_info["lower_img"], scale=2)
            lower_lines = normalize_menu_list(clean_lines(lower_text))
            lunch_data["ilpum"] = lower_lines
            lunch_data["ilpum_main"] = pick_main_menu(lower_lines) if lower_lines else "메뉴 정보 없음"

        # dinner
        dn_img = img.crop(dinner_boxes[i])
        dn_text = ocr_text(client, dn_img, scale=2)
        dn_lines = normalize_menu_list(clean_lines(dn_text))

        out_days[dk] = {
            "breakfast": {
                "han": bf_lines,
                "main": pick_main_menu(bf_lines) if bf_lines else "메뉴 정보 없음",
            },
            "lunch": lunch_data,
            "dinner": {
                "han": dn_lines,
                "main": pick_main_menu(dn_lines) if dn_lines else "메뉴 정보 없음",
            },
        }

        debug_meta[dk] = {
            "lunch_box": list(lunch_boxes[i]),
            "split_y": lunch_info["split_y"],
            "split_source": lunch_info["split_source"],
            "upper_blue_count": lunch_info["upper_blue_count"],
            "lower_blue_count": lunch_info["lower_blue_count"],
            "upper_blue_centers": lunch_info["upper_blue_centers"],
            "lower_blue_centers": lunch_info["lower_blue_centers"],
        }

    return {
        "range": range_text,
        "week_start": week_start.isoformat(),
        "header_dates": {k: list(v) for k, v in header_dates.items()},
        "saved_at_kst": kst_now().isoformat(timespec="seconds"),
        "days": out_days,
        "debug": debug_meta,
    }

# =========================================================
# annotate debug image
# =========================================================
def annotate_menu_image(image_bytes: bytes, parsed: Dict, output_path: str):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)

    font = get_font(14)
    small = get_font(12)

    # legend
    legend = [
        "BLUE box = han / ilpum regions",
        "RED line = split line from left guide",
        "CYAN line = upper blue bands",
        "GREEN line = lower blue bands",
    ]
    for i, txt in enumerate(legend):
        draw.text((20, 20 + i * 16), txt, fill="white", font=small)

    for i, dk in enumerate(DAY_KEYS):
        debug = parsed["debug"][dk]
        lbox = debug["lunch_box"]
        lx1, ly1, lx2, ly2 = lbox
        split_y_abs = ly1 + debug["split_y"]

        upper_abs = (lx1, ly1, lx2, split_y_abs)
        lower_abs = (lx1, split_y_abs, lx2, ly2)

        draw.rectangle(upper_abs, outline="blue", width=3)
        draw.rectangle(lower_abs, outline="blue", width=3)
        draw.line((lx1, split_y_abs, lx2, split_y_abs), fill="red", width=3)

        for cy in debug["upper_blue_centers"]:
            yy = ly1 + cy
            draw.line((lx1, yy, lx2, yy), fill="cyan", width=2)

        for cy in debug["lower_blue_centers"]:
            yy = split_y_abs + cy
            draw.line((lx1, yy, lx2, yy), fill="green", width=2)

        label = f"{DAY_LABELS[i]} {parsed['days'][dk]['lunch']['mode']} U{debug['upper_blue_count']} L{debug['lower_blue_count']}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rectangle((lx1 + 2, ly1 + 2, lx1 + tw + 10, ly1 + th + 6), fill="white")
        draw.text((lx1 + 4, ly1 + 3), label, fill="black", font=font)

        draw.text((lx1 + 4, ly2 - 16), debug["split_source"], fill="white", font=small)

    img.save(output_path, quality=95)

# =========================================================
# persistence
# =========================================================
def load_meals() -> Dict:
    if not os.path.exists(MEALS_FILE):
        return {"range": "", "week_start": "", "days": {}}
    try:
        with open(MEALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "days" not in data:
            data["days"] = {}
        return data
    except Exception:
        return {"range": "", "week_start": "", "days": {}}

def save_meals(data: Dict):
    with open(MEALS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =========================================================
# menu display helpers
# =========================================================
def get_active_day_key(offset_days: int = 0) -> str:
    data = load_meals()
    week_start_str = data.get("week_start", "")

    if week_start_str:
        try:
            ws = date.fromisoformat(week_start_str)
            target = today_kst() + timedelta(days=offset_days)
            delta = (target - ws).days
            delta = max(0, min(6, delta))
            return DAY_KEYS[delta]
        except Exception:
            pass

    target = today_kst() + timedelta(days=offset_days)
    return DAY_KEYS[target.weekday()]

def format_lunch(day_selector: str, option: Optional[str] = None) -> str:
    label = "오늘" if day_selector == "today" else "내일"
    data = load_meals()
    rng = data.get("range", "")
    dkey = get_active_day_key(0 if day_selector == "today" else 1)
    day = (data.get("days", {}) or {}).get(dkey, {})

    header = f"📋 {label} 점심 전체 메뉴"
    if rng:
        header += f"\n(저장된 식단표: {rng})"

    lunch = (day.get("lunch", {}) or {})
    mode = lunch.get("mode", "single")

    if mode == "single":
        menu = lunch.get("han", [])
        if not menu:
            return header + "\n\n(메뉴 없음/파싱 실패)"
        return header + "\n\n" + "\n".join(menu)

    opt = option or "han"
    opt_name = "한식" if opt == "han" else "일품"
    menu = lunch.get(opt, [])
    if not menu:
        return header + f"\n\n[{opt_name}] (메뉴 없음/파싱 실패)"
    return header + f"\n\n[{opt_name}]\n" + "\n".join(menu)

def format_other_meal(day_selector: str, meal_key: str) -> str:
    label = "오늘" if day_selector == "today" else "내일"
    data = load_meals()
    rng = data.get("range", "")
    dkey = get_active_day_key(0 if day_selector == "today" else 1)
    day = (data.get("days", {}) or {}).get(dkey, {})

    header = f"📋 {label} {MEAL_NAME[meal_key]} 전체 메뉴"
    if rng:
        header += f"\n(저장된 식단표: {rng})"

    menu = (day.get(meal_key, {}) or {}).get("han", [])
    if not menu:
        return header + "\n\n(메뉴 없음/파싱 실패)"
    return header + "\n\n" + "\n".join(menu)

def next_buttons(day_selector: str, meal_key: str) -> dict:
    data = load_meals()
    dkey = get_active_day_key(0 if day_selector == "today" else 1)
    lunch_mode = (((data.get("days", {}) or {}).get(dkey, {})).get("lunch", {}) or {}).get("mode", "single")

    if day_selector == "today" and meal_key == "breakfast":
        if lunch_mode == "dual":
            return {"inline_keyboard": [[
                {"text": "점심(한식)", "callback_data": "view|today|lunch|han"},
                {"text": "점심(일품)", "callback_data": "view|today|lunch|ilpum"},
            ], [
                {"text": "저녁", "callback_data": "view|today|dinner"},
            ]]}
        return {"inline_keyboard": [[
            {"text": "점심", "callback_data": "view|today|lunch"},
        ], [
            {"text": "저녁", "callback_data": "view|today|dinner"},
        ]]}

    if day_selector == "today" and meal_key == "lunch":
        return {"inline_keyboard": [[
            {"text": "저녁", "callback_data": "view|today|dinner"},
        ]]}

    if day_selector == "today" and meal_key == "dinner":
        return {"inline_keyboard": [[
            {"text": "내일 아침", "callback_data": "view|tomorrow|breakfast"},
        ]]}

    return {"inline_keyboard": []}

# =========================================================
# alerts
# =========================================================
def send_meal_alert(meal_key: str):
    data = load_meals()
    days = data.get("days", {})
    if not days:
        tg_send("⚠️ 아직 식단표가 저장되지 않았어요. 관리자(업로더)가 식단표 사진을 먼저 올려야 합니다.")
        return

    dkey = get_active_day_key(0)
    day = days.get(dkey, {})

    if meal_key == "lunch":
        lunch = day.get("lunch", {}) or {}
        mode = lunch.get("mode", "single")

        if mode == "dual":
            han_main = lunch.get("han_main", "메뉴없음")
            il_main = lunch.get("ilpum_main", "메뉴없음")
            text = f"🍱 오늘 점심 메뉴\n한식: {han_main}\n일품: {il_main}"
            keyboard = {"inline_keyboard": [[
                {"text": "점심(한식)", "callback_data": "view|today|lunch|han"},
                {"text": "점심(일품)", "callback_data": "view|today|lunch|ilpum"},
            ]]}
            tg_send(text, keyboard)
            return

        main = lunch.get("han_main", "메뉴없음")
        text = f"🍱 오늘 점심 메뉴는 {main} 입니다"
        keyboard = {"inline_keyboard": [[
            {"text": "전체 메뉴 보기", "callback_data": "view|today|lunch"},
        ]]}
        tg_send(text, keyboard)
        return

    menu_obj = day.get(meal_key, {}) or {}
    main = menu_obj.get("main", "메뉴 정보 없음")
    text = f"🍱 오늘 {MEAL_NAME[meal_key]} 메뉴는 {main} 입니다"
    keyboard = {"inline_keyboard": [[
        {"text": "전체 메뉴 보기", "callback_data": f"view|today|{meal_key}"},
    ]]}
    tg_send(text, keyboard)

# =========================================================
# routes
# =========================================================
@app.route("/", methods=["POST"])
def webhook():
    data = request.json or {}

    # photo upload
    if "message" in data:
        msg = data["message"]
        user_id = msg.get("from", {}).get("id")

        if "photo" in msg:
            if user_id != ADMIN_ID:
                return "ok"

            try:
                ensure_env()

                file_id = msg["photo"][-1]["file_id"]
                img_bytes = download_telegram_photo(file_id)

                parsed = parse_weekly(img_bytes)
                save_meals(parsed)

                base_name = parsed["week_start"].replace("-", "_")
                debug_img_path = os.path.join(DEBUG_DIR, f"{base_name}_annotated.jpg")
                annotate_menu_image(img_bytes, parsed, debug_img_path)

                # upload 완료 메시지
                dkey = get_active_day_key(0)
                lunch_mode = (((parsed.get("days", {}) or {}).get(dkey, {})).get("lunch", {}) or {}).get("mode", "single")

                if lunch_mode == "dual":
                    keyboard = {"inline_keyboard": [[
                        {"text": "점심(한식) 보기", "callback_data": "view|today|lunch|han"},
                        {"text": "점심(일품) 보기", "callback_data": "view|today|lunch|ilpum"},
                    ]]}
                else:
                    keyboard = {"inline_keyboard": [[
                        {"text": "점심 보기", "callback_data": "view|today|lunch"},
                    ]]}

                tg_send(f"📅 {parsed['range']} 식단표 업로드 완료!", keyboard)

            except Exception as e:
                tg_send(f"⚠️ 업로드/파싱 오류: {type(e).__name__}: {str(e)[:120]}")

            return "ok"

    # callback button
    if "callback_query" in data:
        q = data["callback_query"]
        cb = q.get("data", "")
        tg_answer_callback(q.get("id"))

        msg = q.get("message", {})
        chat_id = msg.get("chat", {}).get("id")
        message_id = msg.get("message_id")

        try:
            parts = cb.split("|")
            if len(parts) >= 3 and parts[0] == "view":
                day_selector = parts[1]
                meal_key = parts[2]
                option = parts[3] if len(parts) >= 4 else None

                if meal_key == "lunch":
                    text = format_lunch(day_selector, option)
                else:
                    text = format_other_meal(day_selector, meal_key)

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
        ensure_env()
        send_meal_alert(meal)
        return "ok", 200
    except Exception as e:
        return f"error: {type(e).__name__}: {str(e)[:120]}", 500

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
