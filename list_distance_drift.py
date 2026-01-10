import json
import re
from pathlib import Path
from math import sqrt

# ------------- regexes -------------

# <points coords="1 1 257 917">Battleship Yamato</points>
POINTS_HTML_REGEX = re.compile(
    r'<points[^>]*coords="([^"]+)"[^>]*>(.*?)</points>',
    re.IGNORECASE | re.DOTALL,
)

# The action to be taken is therefore (257, -83)
ACTION_REGEX = re.compile(
    r"The action to be taken is therefore\s*"
    r"\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)"
)

def has_valid_plain_action(text: str):
    """
    Returns (True, (dx, dy)) if text contains a plain-text action:
        The action to be taken is therefore (dx, dy)
    with no HTML tags in that clause. Otherwise (False, None).
    """
    m = ACTION_REGEX.search(text)
    if not m:
        return False, None

    span_start, span_end = m.span()
    action_substr = text[span_start:span_end]
    if "<" in action_substr:
        return False, None

    dx = int(m.group(1))
    dy = int(m.group(2))
    return True, (dx, dy)


def parse_points_from_html(text: str):
    """
    Extract object and centre pixel coordinates from <points ...> tags.

    Assumptions:
      - coords string has at least 2 numbers; the LAST TWO are (x, y) in pixels.
      - Tags with inner text containing 'centre of image' are centre points.
      - First non-centre <points> tag is treated as the object point.

    Returns:
      obj_point:   (x_obj, y_obj) or None
      centre_point:(x_ctr, y_ctr) or None
    """
    matches = POINTS_HTML_REGEX.findall(text)
    obj_point = None
    centre_point = None

    for coords_str, inner_text in matches:
        # Extract all ints from coords string and take the last two as (x, y)
        nums = [int(n) for n in re.findall(r"[+-]?\d+", coords_str)]
        if len(nums) < 2:
            continue
        x, y = nums[-2], nums[-1]

        if "centre of image" in inner_text.lower():
            centre_point = (x, y)
        elif obj_point is None:
            obj_point = (x, y)

    return obj_point, centre_point


def extract_and_check(text: str, image_w=1920, image_h=1200, tol_px=5):
    """
    Extract:
      - object coordinate (x_obj, y_obj)
      - centre coordinate (x_ctr, y_ctr)
      - action (dx, dy)

    Then check if the action corresponds to the geometric distance:
        expected_dx = x_ctr - x_obj
        expected_dy = y_ctr - y_obj
    (i.e. action moves centre onto the object, as in your examples).[file:1]

    Returns a dict:
      {
        "obj_point": (x_obj, y_obj) or None,
        "centre_point": (x_ctr, y_ctr) or None,
        "action": (dx, dy) or None,
        "diff_vec": (ex_dx - dx, ex_dy - dy) or None,
        "diff_norm": float or None,
        "is_consistent": bool
      }
    """
    obj_point, centre_point = parse_points_from_html(text)
    has_action, action = has_valid_plain_action(text)

    # Default result
    result = {
        "obj_point": obj_point,
        "centre_point": centre_point,
        "action": action,
        "diff_vec": None,
        "diff_norm": None,
        "is_consistent": False,
    }

    if not (obj_point and centre_point and has_action):
        # Missing any of the three → cannot check consistency
        return result

    x_obj, y_obj = obj_point
    x_ctr, y_ctr = centre_point
    dx, dy = action

    # Expected action: move centre onto object
    expected_dx = x_ctr - x_obj
    expected_dy = y_ctr - y_obj

    diff_x = expected_dx - dx
    diff_y = expected_dy - dy
    diff_norm = sqrt(diff_x**2 + diff_y**2)

    result["diff_vec"] = (diff_x, diff_y)
    result["diff_norm"] = diff_norm
    result["is_consistent"] = diff_norm <= tol_px

    return result

def check_all_examples(jsonl_path, tol_px=5):
    jsonl_path = Path(jsonl_path)
    results = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            vla_out = data.get("vla_output") or ""
            iteration = data.get("iteration")
            before_name = data.get("before_screenshot")

            parsed = extract_and_check(vla_out, tol_px=tol_px)
            parsed["iteration"] = iteration
            parsed["before_screenshot"] = before_name
            results.append(parsed)

    return results


# Example: print only inconsistent ones
all_res = check_all_examples("vla_evaluation/metadata.jsonl", tol_px=10)
for r in all_res:
    if r["diff_norm"] is not None and not r["is_consistent"]:
        print(
            r["before_screenshot"],
            "iter", r["iteration"],
            "obj", r["obj_point"],
            "ctr", r["centre_point"],
            "act", r["action"],
            "diff", r["diff_vec"],
            "||diff||", r["diff_norm"],
        )
