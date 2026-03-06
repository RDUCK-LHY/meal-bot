import cv2
import numpy as np
from PIL import Image
import io

DAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def crop_box(img: Image.Image, box):
    x1, y1, x2, y2 = box
    return img.crop((x1, y1, x2, y2))


def to_png_bytes(img: Image.Image, scale: int = 2) -> bytes:
    img2 = img.resize((img.size[0] * scale, img.size[1] * scale))
    buf = io.BytesIO()
    img2.save(buf, format="PNG")
    return buf.getvalue()


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
    """
    표 외곽 box: (x, y, w, h)
    """
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
    """
    비슷한 좌표를 하나로 병합
    """
    if not values:
        return []

    values = sorted(values)
    groups = [[values[0]]]

    for v in values[1:]:
        if abs(v - groups[-1][-1]) <= gap:
            groups[-1].append(v)
        else:
            groups.append([v])

    merged = [int(sum(g) / len(g)) for g in groups]
    return merged


def detect_grid_lines_in_table(img: Image.Image, table_box):
    """
    표 내부 가로/세로선 좌표를 반환
    """
    x, y, w, h = table_box
    table_img = crop_box(img, (x, y, x + w, y + h))
    cv_img = pil_to_cv(table_img)
    bw = preprocess_for_lines(cv_img)
    horizontal, vertical = extract_horizontal_vertical_lines(bw)

    # 가로선 좌표
    h_contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    y_positions = []
    for c in h_contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cw > w * 0.4:
            y_positions.append(cy + ch // 2)

    # 세로선 좌표
    v_contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_positions = []
    for c in v_contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if ch > h * 0.4:
            x_positions.append(cx + cw // 2)

    y_positions = merge_close_positions(y_positions, gap=8)
    x_positions = merge_close_positions(x_positions, gap=8)

    # 원본 이미지 기준으로 변환
    y_positions = [y + yy for yy in y_positions]
    x_positions = [x + xx for xx in x_positions]

    return x_positions, y_positions


def build_blocks_from_lines(img: Image.Image):
    """
    이미지 처리로 표/날짜/식사 블럭 자동 계산
    반환:
      {
        "table_box": ...,
        "date_boxes": [7개],
        "breakfast_boxes": [7개],
        "lunch_boxes": [7개],
        "dinner_boxes": [7개]
      }
    """
    table_box = detect_table_box(img)
    if table_box is None:
        raise RuntimeError("table_not_found")

    x_lines, y_lines = detect_grid_lines_in_table(img, table_box)

    # 너무 적게 검출되면 실패
    if len(x_lines) < 9 or len(y_lines) < 8:
        raise RuntimeError(f"grid_not_found x={len(x_lines)} y={len(y_lines)}")

    # 보통:
    # x_lines: [표왼쪽, 구분열경계, 월, 화, 수, 목, 금, 토, 일, 표오른쪽]
    # y_lines: [표위, 날짜헤더끝, 아침끝, 점심끝, 저녁끝, ...]
    #
    # 여기서는 오른쪽 7개 날짜 칸만 사용
    # 두 번째 세로선 이후가 월~일 영역이라고 가정
    day_start_idx = 2
    day_end_idx = day_start_idx + 7

    if len(x_lines) < day_end_idx + 1:
        raise RuntimeError("not_enough_vertical_lines")

    # 헤더 끝 / 아침 끝 / 점심 끝 / 저녁 끝
    # 보통 위에서부터 큰 블럭 경계선 5개 이상 잡힘
    # 상위 주요 가로선만 사용
    main_y = sorted(y_lines)

    # 가장 위 5~6개 큰 구조선을 사용
    # 여기서는 경험적으로 첫 5개를 사용:
    # 0 표위, 1 헤더끝, 2 아침끝, 3 점심끝, 4 저녁끝
    if len(main_y) < 5:
        raise RuntimeError("not_enough_horizontal_lines")

    y0 = main_y[0]
    y1 = main_y[1]  # 날짜 헤더 끝
    y2 = main_y[2]  # 아침 끝
    y3 = main_y[3]  # 점심 끝
    y4 = main_y[4]  # 저녁 끝

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


def detect_horizontal_split_line(block_img: Image.Image):
    """
    점심 블럭 내부의 '중간 가로 경계선' 찾기
    있으면 y 좌표 반환, 없으면 None
    """
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
        if cw > w * 0.55 and ch < h * 0.08:
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
    """
    이미지 처리만으로 점심 블럭을 single/dual 판정
    """
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
