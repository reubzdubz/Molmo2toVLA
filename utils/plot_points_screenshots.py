import re

# Your regexes
COORD_REGEX = re.compile(rf"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
FRAME_REGEX = re.compile(rf"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")


def _points_from_num_str(text, image_w, image_h, extract_ids=False):
    """
    Yield (idx, x, y) in absolute pixel coords, assuming coords are scaled by 1000.
    """
    for points in POINTS_REGEX.finditer(text):
        ix, x, y = points.group(1), points.group(2), points.group(3)
        x, y = float(x) / 1000 * image_w, float(y) / 1000 * image_h
        if 0 <= x <= image_w and 0 <= y <= image_h:
            yield ix, x, y


def extract_multi_image_points(text, image_w, image_h, extract_ids=False):
    """
    Extract pointing coordinates as a list of (frame_id, x, y) or
    (frame_id, idx, x, y) if extract_ids=True.
    For your case (single images, fixed res), pass image_w=1920, image_h=1200.
    """
    all_points = []

    if isinstance(image_w, (list, tuple)) and isinstance(image_h, (list, tuple)):
        assert len(image_w) == len(image_h)
        diff_res = True
    else:
        diff_res = False

    for coord in COORD_REGEX.finditer(text):
        for point_grp in FRAME_REGEX.finditer(coord.group(1)):
            frame_id = int(point_grp.group(1)) if diff_res else float(point_grp.group(1))
            w, h = (
                (image_w[frame_id - 1], image_h[frame_id - 1])
                if diff_res
                else (image_w, image_h)
            )
            for idx, x, y in _points_from_num_str(point_grp.group(2), w, h):
                if extract_ids:
                    all_points.append((frame_id, idx, int(x), int(y)))
                else:
                    all_points.append((frame_id, int(x), int(y)))
    return all_points

import json
from pathlib import Path

def extract_points_from_jsonl(jsonl_path, image_w=1920, image_h=1200):
    """
    Returns a dict:
        {
          before_screenshot_filename: {
              "iteration": int,
              "points": [(frame_id, x, y), ...]
          },
          ...
        }
    Assumes each JSON object has at least 'before_screenshot', 'iteration',
    and 'vla_output' fields as in your metadata.jsonl.[file:1]
    """
    jsonl_path = Path(jsonl_path)
    results = {}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            text = data.get("vla_output", "")
            before_name = data.get("before_screenshot")
            iteration = data.get("iteration")

            if not before_name or not text:
                continue

            points = extract_multi_image_points(text, image_w, image_h)

            if points:
                results[before_name] = {
                    "iteration": iteration,
                    "points": points,
                }

    return results

import os
import matplotlib.pyplot as plt

def plot_points_on_befores(
    jsonl_path,
    images_dir,
    out_dir=None,
    image_w=1920,
    image_h=1200,
    show=False,
):
    """
    For each JSONL line with extracted points, load its 'before_screenshot'
    image, plot the points, and either display or save the figure.

    - jsonl_path: path to metadata.jsonl
    - images_dir: directory containing before_XXXX.png, etc.
    - out_dir: if not None, save plots as PNGs there
    - show: if True, call plt.show() for each image instead of/besides saving
    """
    os.makedirs(out_dir, exist_ok=True) if out_dir is not None else None

    points_by_image = extract_points_from_jsonl(jsonl_path, image_w, image_h)

    for before_name, info in points_by_image.items():
        img_path = os.path.join(images_dir, before_name)
        if not os.path.isfile(img_path):
            # Skip if image is missing
            continue

        img = plt.imread(img_path)
        fig, ax = plt.subplots(figsize=(image_w / 200, image_h / 200), dpi=100)
        ax.imshow(img)

        # Extract x, y from (frame_id, x, y)
        xs = [p[1] for p in info["points"]]
        ys = [p[2] for p in info["points"]]

        # Scatter points
        ax.scatter(xs, ys, c="red", s=40, marker="o")

        # Optional: label points with their index
        for i, (frame_id, x, y) in enumerate(info["points"]):
            ax.text(x + 5, y - 5, str(i), color="yellow", fontsize=8)

        ax.set_title(f"{before_name} (iteration {info['iteration']})")
        ax.set_axis_off()

        if out_dir is not None:
            out_path = os.path.join(out_dir, f"points_{before_name}")
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        if show:
            plt.show()
        plt.close(fig)

# Paths you need to set
jsonl_path = "vla_evaluation/metadata.jsonl"
images_dir = "vla_evaluation/"  # contains before_0001.png, ...
out_dir = "plots_with_points/"

plot_points_on_befores(
    jsonl_path=jsonl_path,
    images_dir=images_dir,
    out_dir=out_dir,
    image_w=1920,
    image_h=1200,
    show=False,   # set True if you want interactive display
)
