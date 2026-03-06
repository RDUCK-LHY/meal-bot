from flask import Flask, request
import os
import json
import re
import tempfile
import io
from datetime import datetime, timedelta, date

import requests
import cv2
import numpy as np
from PIL import Image
from google.cloud import vision

# ============================================================
# ENV
# ============================================================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
GCP_SA_JSON = os.getenv("GCP_SA_JSON")
CRON_SECRET = os.getenv("CRON_SECRET", "")

app = Flask(__name__)
MEALS_FILE = "meals.json"

DAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
MEAL_NAME = {"breakfast": "아침", "lunch": "점심", "dinner": "저녁"}

NOISE_EXACT = {"한식", "일품", "공통", "SELF", "PLUS", "오늘의차"}
NOISE_CONTAINS = [
    "공통", "SELF", "PLUS", "오늘의차",
    "라면", "코너", "누룽지", "도시락김", "샐러드", "그린",
    "STEMS", "aramark", "구분",
    "조식", "중식", "석식", "1조", "2조", "관리",
    "차", "우유", "요거트"
]

MAIN_HINTS = [
    "국", "찌개", "탕", "전골", "순두부", "육개장", "설렁탕", "감자탕", "부대찌개", "미역국", "된장",
    "덮밥", "비빔밥", "볶음밥", "카레", "짜장", "짬뽕", "우동", "국수", "라면",
    "돈까스", "스테이크", "파스타", "스파게티", "피자", "햄버거",
    "갈비", "불고기", "제육", "닭갈비", "찜", "구이", "조림", "볶음"
]
NOT_MAIN_HINTS = [
    "김치", "깍두기", "단무지", "무말랭이", "나물", "샐러드", "도시락김",
    "밥", "현미밥", "잡곡밥", "누룽지", "차", "과일",
    "소스", "케찹", "드레싱", "피클", "초장", "양념장"
]

# 표를 못 찾았을 때만 쓰는 fallback 비율
FALLBACK = {
    "table_left": 0.05,
    "table_top": 0.10,
    "table_right": 0.985,
    "table_bottom": 0.93,
    "days_left": 0.12,
    "days_right": 0.995,
    "header_top": 0.06,
    "header_bottom": 0.13,
    "breakfast_top": 0.13,
    "breakfast_bottom": 0.29,
    "lunch_top": 0.29,
    "lunch_bottom": 0.56,
    "dinner_top": 0.56,
    "dinner_bottom": 0.82,
}

# ============================================================
# Time
# ============================================================
def kst_now() -> datetime:
    return datetime.utcnow() + timedelta(hours=9)

def today_kst_date() -> date:
    return kst_now().date()

# ============================================================
# Telegram
# ============================================================
def tg_send(text: str, keyboard: dict | None = None):
    data = {"chat_id": CHANNEL_ID, "text": text}
    if keyboard is not None:
        data["reply_markup"] = json.dumps(keyboard, ensure_ascii=False)
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data=data, timeout=20)
    r.raise_for_status()

def tg_edit(chat_id, message_id, text: str, keyboard: dict | None = None):
    data = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if keyboard is not None:
        data["reply_markup"] = json.dumps(keyboard, ensure_ascii=False)
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/editMessageText", data=data, timeout=20)
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
# PIL / OpenCV helpers
# ============================================================
def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def crop_box(img: Image.Image, box):
    x1, y1, x2, y2 = box
    return img.crop((x1, y1, x2, y2))

def crop_ratio(img: Image.Image, l, t, r, b):
    w, h = img.size
    return img.crop((int(w * l), int(h * t), int(w * r), int(h * b)))

def to_png_bytes(img: Image.Image, scale: int = 2) -> bytes:
    img2 = img.resize((img.size[0] * scale, img.size[1] * scale))
    buf = io.BytesIO()
    img2.save(buf, format="PNG")
    return buf.getvalue()

# ============================================================
# Image processing
# ============================================================
def preprocess_for_lines(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    bw = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        8
    )
    return bw

def extract_horizontal_vertical_lines(bin_img: np.ndarray):
    h, w = bin_img.shape

    h_kernel_len = max(30, w // 20)
    v_kernel_len = max(30, h // 20)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))

    horizontal = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vertical = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, v_kernel, iterations=1)

    return horizontal, vertical

def detect_table_box(img: Image.Image):
    cv_img = pil_to_cv(img)
    bw = preprocess_for_lines(cv_img)
    horizontal, vertical = extract_horizontal_vertical_lines(bw)

    table_mask = cv2.add(horizontal, vertical)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    table_mask = cv2.dilate(table_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    H, W = bw.shape
    candidates = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if w > W * 0.5 and h > H * 0.5:
            candidates.append((area, (x, y, w, h)))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda t: t[0])
    return candidates[0][1]

def merge_close_positions(values, gap=10):
    if not values:
        return []

    values = sorted(values)
    groups = [[values[0]]]

    for v in values[1:]:
        if abs(v - groups[-1][-1]) <= gap:
            groups[-1].append(v)
        else:
            groups.append([v])

    return [int(sum(g) / len(g)) for g in groups]

def detect_grid_lines_in_table(img: Image.Image, table_box):
    x, y, w, h = table_box
    table_img = crop_box(img, (x, y, x + w, y + h))
    cv_img = pil_to_cv(table_img)
    bw = preprocess_for_lines(cv_img)
    horizontal, vertical = extract_horizontal_vertical_lines(bw)

    h_contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    y_positions = []
    for c in h_contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cw > w * 0.4:
            y_positions.append(cy + ch // 2)

    v_contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_positions = []
    for c in v_contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if ch > h * 0.4:
            x_positions.append(cx + cw // 2)

    y_positions = merge_close_positions(y_positions, gap=8)
    x_positions = merge_close_positions(x_positions, gap=8)

    y_positions = [y + yy for yy in y_positions]
    x_positions = [x + xx for xx in x_positions]

    return x_positions, y_positions

def build_blocks_from_lines(img: Image.Image):
    table_box = detect_table_box(img)

    # 표 못 찾으면 fallback
    if table_box is None:
        w, h = img.size
        x1 = int(w * FALLBACK["table_left"])
        y1 = int(h * FALLBACK["table_top"])
        x2 = int(w * FALLBACK["table_right"])
        y2 = int(h * FALLBACK["table_bottom"])

        days_x1 = int(x1 + (x2 - x1) * FALLBACK["days_left"])
        days_x2 = int(x1 + (x2 - x1) * FALLBACK["days_right"])

        def iy(r):
            return int(y1 + (y2 - y1) * r)

        header = (days_x1, iy(FALLBACK["header_top"]), days_x2, iy(FALLBACK["header_bottom"]))
        breakfast = (days_x1, iy(FALLBACK["breakfast_top"]), days_x2, iy(FALLBACK["breakfast_bottom"]))
        lunch = (days_x1, iy(FALLBACK["lunch_top"]), days_x2, iy(FALLBACK["lunch_bottom"]))
        dinner = (days_x1, iy(FALLBACK["dinner_top"]), days_x2, iy(FALLBACK["dinner_bottom"]))

        return {
            "table_box": (x1, y1, x2 - x1, y2 - y1),
            "date_boxes": split_columns(header, 7),
            "breakfast_boxes": split_columns(breakfast, 7),
            "lunch_boxes": split_columns(lunch, 7),
            "dinner_boxes": split_columns(dinner, 7),
        }

    x_lines, y_lines = detect_grid_lines_in_table(img, table_box)

    if len(x_lines) < 9 or len(y_lines) < 5:
        raise RuntimeError(f"grid_not_found x={len(x_lines)} y={len(y_lines)}")

    day_start_idx = 2
    day_end_idx = day_start_idx + 7

    if len(x_lines) < day_end_idx + 1:
        raise RuntimeError("not_enough_vertical_lines")

    main_y = sorted(y_lines)
    y0 = main_y[0]
    y1 = main_y[1]
    y2 = main_y[2]
    y3 = main_y[3]
    y4 = main_y[4]

    date_boxes = []
    breakfast_boxes = []
    lunch_boxes = []
    dinner_boxes = []

    for i in range(day_start_idx, day_end_idx):
        x1 = x_lines[i]
        x2 = x_lines[i + 1]

        date_boxes.append((x1, y0, x2, y1))
        breakfast_boxes.append((x1, y1, x2, y2))
        lunch_boxes.append((x1, y2, x2, y3))
        dinner_boxes.append((x1, y3, x2, y4))

    return {
        "table_box": table_box,
        "date_boxes": date_boxes,
        "breakfast_boxes": breakfast_boxes,
        "lunch_boxes": lunch_boxes,
        "dinner_boxes": dinner_boxes,
    }

def split_columns(area_box, n=7):
    x1, y1, x2, y2 = area_box
    w = (x2 - x1) / n
    boxes = []
    for i in range(n):
        cx1 = int(x1 + w * i)
        cx2 = int(x1 + w * (i + 1))
        boxes.append((cx1, y1, cx2, y2))
    return boxes

def detect_horizontal_split_line(block_img: Image.Image):
    cv_img = pil_to_cv(block_img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10
    )

    h, w = bw.shape

    kernel_len = max(20, w // 5)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    horizontal = cv2.erode(bw, h_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, h_kernel, iterations=2)

    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > w * 0.55 and ch < max(3, h * 0.08):
            center_y = y + ch // 2
            candidates.append((center_y, x, y, cw, ch))

    if not candidates:
        return None

    mid = h / 2
    candidates.sort(key=lambda t: abs(t[0] - mid))
    best = candidates[0]
    center_y = best[0]

    if abs(center_y - mid) < h * 0.22:
        return center_y

    return None

def split_lunch_by_line(lunch_img: Image.Image):
    split_y = detect_horizontal_split_line(lunch_img)

    if split_y is None:
        return {
            "mode": "single",
            "han_img": lunch_img,
            "il_img": None,
        }

    w, h = lunch_img.size
    top = lunch_img.crop((0, 0, w, split_y))
    bottom = lunch_img.crop((0, split_y, w, h))

    return {
        "mode": "dual",
        "han_img": top,
        "il_img": bottom,
    }

# ============================================================
# OCR text helpers
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
    return [main] + [x for x in lines if x != main]

# ============================================================
# Date helpers
# ============================================================
def extract_mmdd_from_text(text: str):
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

def build_range_and_week_start(date_boxes, img: Image.Image, client):
    dates = {}
    for dk, box in zip(DAY_KEYS, date_boxes):
        cell = crop_box(img, box)
        resp = ocr_document(client, to_png_bytes(cell, scale=3))
        text = resp.full_text_annotation.text if resp.full_text_annotation else ""
        mmdd = extract_mmdd_from_text(text)
        if mmdd:
            dates[dk] = mmdd

    today = today_kst_date()
    year = today.year

    if "mon" not in dates:
        ws = today - timedelta(days=today.weekday())
        return "날짜 인식 실패", ws

    mon_m, mon_d = dates["mon"]
    sun_m, sun_d = dates.get("sun", dates["mon"])

    week_start = date(year, mon_m, mon_d)

    if mon_m == 12 and today.month == 1:
        week_start = date(year - 1, mon_m, mon_d)
    elif mon_m == 1 and today.month == 12:
        week_start = date(year + 1, mon_m, mon_d)

    return f"{mon_m}/{mon_d}(월) ~ {sun_m}/{sun_d}(일)", week_start

# ============================================================
# Storage
# ============================================================
def save_meals(data: dict):
    with open(MEALS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_meals() -> dict:
    if not os.path.exists(MEALS_FILE):
        return {"range": "", "week_start": "", "days": {}}
    try:
        with open(MEALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"range": "", "week_start": "", "days": {}}
        if "days" not in data:
            data["days"] = {}
        if "range" not in data:
            data["range"] = ""
        if "week_start" not in data:
            data["week_start"] = ""
        return data
    except Exception:
        return {"range": "", "week_start": "", "days": {}}

# ============================================================
# Parse weekly
# ============================================================
def parse_weekly(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    client = vision_client()

    blocks = build_blocks_from_lines(img)
    date_range, week_start = build_range_and_week_start(blocks["date_boxes"], img, client)

    out = {
        dk: {
            "breakfast": {"han": []},
            "lunch": {"han": [], "ilpum": [], "mode": "single"},
            "dinner": {"han": []},
        } for dk in DAY_KEYS
    }

    for idx, dk in enumerate(DAY_KEYS):
        # breakfast
        bf_img = crop_box(img, blocks["breakfast_boxes"][idx])
        bf_resp = ocr_document(client, to_png_bytes(bf_img, scale=2))
        bf_text = bf_resp.full_text_annotation.text if bf_resp.full_text_annotation else ""
        out[dk]["breakfast"]["han"] = normalize_menu_list(clean_lines(bf_text))

        # lunch
        lunch_img = crop_box(img, blocks["lunch_boxes"][idx])
        split = split_lunch_by_line(lunch_img)

        han_resp = ocr_document(client, to_png_bytes(split["han_img"], scale=2))
        han_text = han_resp.full_text_annotation.text if han_resp.full_text_annotation else ""
        han_lines = normalize_menu_list(clean_lines(han_text))

        if split["mode"] == "single":
            out[dk]["lunch"]["han"] = han_lines
            out[dk]["lunch"]["ilpum"] = []
            out[dk]["lunch"]["mode"] = "single"
        else:
            il_resp = ocr_document(client, to_png_bytes(split["il_img"], scale=2))
            il_text = il_resp.full_text_annotation.text if il_resp.full_text_annotation else ""
            il_lines = normalize_menu_list(clean_lines(il_text))

            out[dk]["lunch"]["han"] = han_lines
            out[dk]["lunch"]["ilpum"] = il_lines
            out[dk]["lunch"]["mode"] = "dual"

        # dinner
        dn_img = crop_box(img, blocks["dinner_boxes"][idx])
        dn_resp = ocr_document(client, to_png_bytes(dn_img, scale=2))
        dn_text = dn_resp.full_text_annotation.text if dn_resp.full_text_annotation else ""
        out[dk]["dinner"]["han"] = normalize_menu_list(clean_lines(dn_text))

    return {
        "range": date_range,
        "week_start": week_start.isoformat(),
        "saved_at_kst": kst_now().isoformat(timespec="seconds"),
        "days": out,
    }

# ============================================================
# Active day
# ============================================================
def day_key_from_week_start(week_start: date, offset_days: int = 0) -> str:
    target = today_kst_date() + timedelta(days=offset_days)
    delta = (target - week_start).days
    if delta < 0:
        return DAY_KEYS[0]
    if delta > 6:
        return DAY_KEYS[6]
    return DAY_KEYS[delta]

def get_active_day_key(offset_days: int = 0) -> str:
    data = load_meals()
    week_start_str = data.get("week_start", "")
    if week_start_str:
        try:
            ws = date.fromisoformat(week_start_str)
            return day_key_from_week_start(ws, offset_days)
        except Exception:
            pass
    target = today_kst_date() + timedelta(days=offset_days)
    return DAY_KEYS[target.weekday()]

# ============================================================
# UI
# ============================================================
def format_meal_full(day_selector: str, meal_key: str, option: str | None = None) -> str:
    label = "오늘" if day_selector == "today" else "내일"
    data = load_meals()
    rng = data.get("range", "")
    dkey = get_active_day_key(0 if day_selector == "today" else 1)
    day = (data.get("days", {}) or {}).get(dkey, {})

    header = f"📋 {label} {MEAL_NAME[meal_key]} 전체 메뉴"
    if rng:
        header += f"\n(저장된 식단표: {rng})"

    if meal_key == "lunch":
        mode = (day.get("lunch", {}) or {}).get("mode", "single")
        if mode == "single":
            menu = (day.get("lunch", {}) or {}).get("han", [])
            if not menu:
                return header + "\n\n(메뉴 없음/파싱 실패)"
            return header + "\n\n" + "\n".join(menu)

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

    dkey = get_active_day_key(0)
    day = days.get(dkey, {})

    if meal_key == "lunch":
        lunch = day.get("lunch", {}) or {}
        mode = lunch.get("mode", "single")
        han = lunch.get("han", [])
        il = lunch.get("ilpum", [])

        han_main = pick_main_menu(han) if han else "메뉴없음"

        if mode == "dual" and il:
            il_main = pick_main_menu(il)
            text = f"🍱 오늘 점심 메뉴\n한식: {han_main}\n일품: {il_main}"
            keyboard = {"inline_keyboard": [[
                {"text": "전체 메뉴 보기", "callback_data": "view|today|lunch|han"}
            ]]}
            tg_send(text, keyboard)
            return

        text = f"🍱 오늘 점심 메뉴는 {han_main} 입니다"
        keyboard = {"inline_keyboard": [[
            {"text": "전체 메뉴 보기", "callback_data": "view|today|lunch"}
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

def remind_upload():
    tg_send("📸 이번 주 식단표 사진을 업로드해주세요! (관리자만 업로드 가능)")

# ============================================================
# Routes
# ============================================================
@app.route("/", methods=["POST"])
def webhook():
    data = request.json or {}

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
                tg_send(f"⚠️ 업로드/파싱 오류: {type(e).__name__}")

            return "ok"

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
