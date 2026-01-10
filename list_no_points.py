import json
from pathlib import Path

def list_examples_without_points(jsonl_path):
    """
    Return a list of dicts for examples whose vla_output does NOT contain
    any <points ...> markup (i.e. no explicit pointing coordinates).

    Each element has: iteration, before_screenshot, vla_output.
    """
    jsonl_path = Path(jsonl_path)
    no_points = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            vla_out = (data.get("vla_output") or "")
            before_name = data.get("before_screenshot")
            iteration = data.get("iteration")

            # Any explicit point should appear as <points ...> in vla_output
            has_points = "<points" in vla_out

            if not has_points:
                no_points.append(
                    {
                        "iteration": iteration,
                        "before_screenshot": before_name,
                        "vla_output": vla_out,
                    }
                )

    return no_points
	
no_point_examples = list_examples_without_points("vla_evaluation/metadata.jsonl")

for ex in no_point_examples:
    print( ex["before_screenshot"], "->", ex["vla_output"])
