from flask import Flask, request
import os
import json
import re
import tempfile
import io
from datetime import datetime, timedelta, date

import requests
import numpy as np
import cv2

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
DAY_KR = {"mon": "월", "tue": "화", "wed": "수", "thu": "목", "fri": "금", "sat": "토", "sun": "일"}
MEAL_NAME = {"breakfast": "아침", "lunch": "점심", "dinner": "저녁"}

# ============================================================
# 기본 비율 (표 외곽 못 찾았을 때 fallback)
# ============================================================
FALLBACK_CROP = {
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
# 텍스트 정리용
# ============================================================
NOISE_EXACT = {"한식", "일품", "공통", "SELF", "PLUS", "오늘의차"}
NOISE_CONTAINS = [
    "공통", "SELF", "PLUS", "오늘의차",
    "라면", "코너", "누룽지", "도시락김", "샐러드", "그린",
    "STEMS", "aramark", "구분",
    "조식", "중식", "석식", "1조", "2조", "관리",
    "차", "우유", "요거트"
]

SPECIAL_SPLIT_HINTS = ["일품", "특식", "분식", "DAY", "NEW"]

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

# ============================================================
# 시간
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

def cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def to_png_bytes(img: Image.Image, scale: int = 2) -> bytes:
    img2 = img.resize((img.size[0] * scale, img.size[1] * scale))
    buf = io.BytesIO()
    img2.save(buf, format="PNG")
    return buf.getvalue()

def crop_box(img: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    return img.crop(box)

def crop_ratio_from_box(box, full_w, full_h):
    x, y, w, h = box
    return {
        "left": x / full_w,
        "top": y / full_h,
        "right": (x + w) / full_w,
        "bottom": (y + h) / full_h,
    }

# ============================================================
# 이미지 처리: 표 외곽 찾기
# ============================================================
def detect_table_box(img: Image.Image):
    """
    표 외곽 contour를 찾아 (x,y,w,h) 반환.
    못 찾으면 None.
    """
    cv_img = pil_to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # 대비 강화
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 8
    )

    # 가로/세로 선 검출
    h_kernel_len = max(20, gray.shape[1] // 25)
    v_kernel_len = max(20, gray.shape[0] // 25)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))

    horizontal = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vertical = cv2.morphologyEx(th, cv2.MORPH_OPEN, v_kernel, iterations=1)
    table_mask = cv2.add(horizontal, vertical)

    # 외곽 확장
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    table_mask = cv2.dilate(table_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    H, W = gray.shape[:2]
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

# ============================================================
# 템플릿 분할
# ============================================================
def build_layout_from_table_box(full_img: Image.Image, table_box):
    """
    표 외곽 box 기준으로 내부 영역 좌표 생성
    """
    full_w, full_h = full_img.size

    if table_box is None:
        # fallback
        x1 = int(full_w * FALLBACK_CROP["table_left"])
        y1 = int(full_h * FALLBACK_CROP["table_top"])
        x2 = int(full_w * FALLBACK_CROP["table_right"])
        y2 = int(full_h * FALLBACK_CROP["table_bottom"])
        table_box = (x1, y1, x2 - x1, y2 - y1)

    x, y, w, h = table_box

    days_x1 = x + int(w * FALLBACK_CROP["days_left"])
    days_x2 = x + int(w * FALLBACK_CROP["days_right"])

    def iy(r):
        return y + int(h * r)

    layout = {
        "table": (x, y, x + w, y + h),
        "days_area": (days_x1, y, days_x2, y + h),

        "header": (days_x1, iy(FALLBACK_CROP["header_top"]), days_x2, iy(FALLBACK_CROP["header_bottom"])),
        "breakfast": (days_x1, iy(FALLBACK_CROP["breakfast_top"]), days_x2, iy(FALLBACK_CROP["breakfast_bottom"])),
        "lunch": (days_x1, iy(FALLBACK_CROP["lunch_top"]), days_x2, iy(FALLBACK_CROP["lunch_bottom"])),
        "dinner": (days_x1, iy(FALLBACK_CROP["dinner_top"]), days_x2, iy(FALLBACK_CROP["dinner_bottom"])),
    }

    return layout

def split_columns(area_box, n=7):
    x1, y1, x2, y2 = area_box
    w = (x2 - x1) / n
    boxes = []
    for i in range(n):
        cx1 = int(x1 + w * i)
        cx2 = int(x1 + w * (i + 1))
        boxes.append((cx1, y1, cx2, y2))
    return boxes

# ============================================================
# 텍스트 정리 / 메인메뉴
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

        bad = False
        for tok in NOISE_CONTAINS:
            if tok in s:
                bad = True
                break
        if bad:
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
# 날짜 헤더 OCR
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

def read_header_dates(img: Image.Image, client, header_box):
    results = {}
    for dk, box in zip(DAY_KEYS, split_columns(header_box, 7)):
        cell = crop_box(img, box)
        resp = ocr_document(client, to_png_bytes(cell, scale=3))
        text = resp.full_text_annotation.text if resp.full_text_annotation else ""
        mmdd = extract_mmdd_from_text(text)
        if mmdd:
            results[dk] = mmdd
    return results

def build_range_and_week_start_from_header(header_dates: dict):
    today = today_kst_date()
    year = today.year

    if "mon" not in header_dates:
        ws = today - timedelta(days=today.weekday())
        return "", ws

    mon_m, mon_d = header_dates["mon"]
    sun_m, sun_d = header_dates.get("sun", header_dates["mon"])

    week_start = date(year, mon_m, mon_d)

    if mon_m == 12 and today.month == 1:
        week_start = date(year - 1, mon_m, mon_d)
    elif mon_m == 1 and today.month == 12:
        week_start = date(year + 1, mon_m, mon_d)

    range_text = f"{mon_m}/{mon_d}(월) ~ {sun_m}/{sun_d}(일)"
    return range_text, week_start

# ============================================================
# 점심 블럭: 1개 / 2개 자동 판단
# ============================================================
def _iter_words_with_boxes(resp):
    if not resp.full_text_annotation or not resp.full_text_annotation.pages:
        return
    for page in resp.full_text_annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    wtxt = "".join([s.text for s in word.symbols]).strip()
                    if not wtxt:
                        continue
                    bbox = word.bounding_box.vertices
                    yield wtxt, bbox

def _bbox_top_y(bbox) -> int:
    return min(v.y for v in bbox if v.y is not None)

def _bbox_bottom_y(bbox) -> int:
    return max(v.y for v in bbox if v.y is not None)

def find_first_label_y(resp, label_candidates):
    candidates = []
    for wtxt, bbox in _iter_words_with_boxes(resp):
        if any(label in wtxt for label in label_candidates):
            candidates.append((_bbox_top_y(bbox), _bbox_bottom_y(bbox), wtxt))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0]

def split_lunch_block_auto(lunch_img: Image.Image, client):
    """
    점심 블럭에서:
    - 2개 블럭이면 위/아래 분리
    - 1개 블럭이면 통째로 사용
    기준:
      1) '일품', '특식', '분식', 'DAY', 'NEW' 같은 텍스트를 OCR로 찾는다
      2) 찾으면 그 y 기준으로 위/아래 분리
      3) 못 찾으면 단일 블럭으로 처리
    """
    scale = 2
    resp = ocr_document(client, to_png_bytes(lunch_img, scale=scale))
    full_text = resp.full_text_annotation.text if resp.full_text_annotation else ""

    split_info = find_first_label_y(resp, SPECIAL_SPLIT_HINTS)
    W, H = lunch_img.size

    if split_info is None:
        # 단일 메뉴
        return {"mode": "single", "han_img": lunch_img, "il_img": None, "text": full_text}

    split_top, split_bottom, split_label = split_info

    split_y = int((split_top - 4) / scale)
    split_y = max(int(H * 0.25), min(int(H * 0.85), split_y))

    han_img = lunch_img.crop((0, 0, W, split_y))
    il_img = lunch_img.crop((0, split_y, W, H))

    return {
        "mode": "dual",
        "han_img": han_img,
        "il_img": il_img,
        "text": full_text,
        "label": split_label,
    }

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
        if "days" not in data or not isinstance(data["days"], dict):
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

    table_box = detect_table_box(img)
    layout = build_layout_from_table_box(img, table_box)

    header_dates = read_header_dates(img, client, layout["header"])
    date_range, week_start = build_range_and_week_start_from_header(header_dates)

    out = {
        dk: {
            "breakfast": {"han": []},
            "lunch": {"han": [], "ilpum": [], "mode": "single"},
            "dinner": {"han": []},
        } for dk in DAY_KEYS
    }

    # 각 요일별 박스
    breakfast_cols = split_columns(layout["breakfast"], 7)
    lunch_cols = split_columns(layout["lunch"], 7)
    dinner_cols = split_columns(layout["dinner"], 7)

    for idx, dk in enumerate(DAY_KEYS):
        # 아침
        bf_cell = crop_box(img, breakfast_cols[idx])
        bf_resp = ocr_document(client, to_png_bytes(bf_cell, scale=2))
        bf_text = bf_resp.full_text_annotation.text if bf_resp.full_text_annotation else ""
        out[dk]["breakfast"]["han"] = normalize_menu_list(clean_lines(bf_text))

        # 점심
        ln_cell = crop_box(img, lunch_cols[idx])
        split_result = split_lunch_block_auto(ln_cell, client)

        if split_result["mode"] == "single":
            han_resp = ocr_document(client, to_png_bytes(split_result["han_img"], scale=2))
            han_text = han_resp.full_text_annotation.text if han_resp.full_text_annotation else ""
            han_lines = normalize_menu_list(clean_lines(han_text))

            out[dk]["lunch"]["han"] = han_lines
            out[dk]["lunch"]["ilpum"] = []
            out[dk]["lunch"]["mode"] = "single"
        else:
            han_resp = ocr_document(client, to_png_bytes(split_result["han_img"], scale=2))
            il_resp = ocr_document(client, to_png_bytes(split_result["il_img"], scale=2))

            han_text = han_resp.full_text_annotation.text if han_resp.full_text_annotation else ""
            il_text = il_resp.full_text_annotation.text if il_resp.full_text_annotation else ""

            han_lines = normalize_menu_list(clean_lines(han_text))
            il_lines = normalize_menu_list(clean_lines(il_text))

            out[dk]["lunch"]["han"] = han_lines
            out[dk]["lunch"]["ilpum"] = il_lines
            out[dk]["lunch"]["mode"] = "dual"

        # 저녁
        dn_cell = crop_box(img, dinner_cols[idx])
        dn_resp = ocr_document(client, to_png_bytes(dn_cell, scale=2))
        dn_text = dn_resp.full_text_annotation.text if dn_resp.full_text_annotation else ""
        out[dk]["dinner"]["han"] = normalize_menu_list(clean_lines(dn_text))

    return {
        "range": date_range if date_range else "날짜 인식 실패",
        "week_start": week_start.isoformat(),
        "saved_at_kst": kst_now().isoformat(timespec="seconds"),
        "days": out,
    }

# ============================================================
# 현재 활성 요일 계산
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
        opt = option or "han"

        if mode == "single":
            menu = (day.get("lunch", {}) or {}).get("han", [])
            if not menu:
                return header + "\n\n(메뉴 없음/파싱 실패)"
            return header + "\n\n" + "\n".join(menu)

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
