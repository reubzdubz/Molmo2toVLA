import json
from pathlib import Path

def list_examples_without_centre(jsonl_path):
    """
    Return a list of dicts for examples whose vla_output does NOT contain
    a 'centre of the image is at ...' description.
    Each element has: iteration, before_screenshot, vla_output.
    """
    jsonl_path = Path(jsonl_path)
    missing = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            vla_out = data.get("vla_output", "") or ""
            before_name = data.get("before_screenshot")
            iteration = data.get("iteration")

            # Heuristic: examples that *have* the centre description
            # typically contain this phrase.[file:1]
            has_centre = '<points coords="1 1 500 500">centre of image</points>' in vla_out

            if not has_centre:
                missing.append(
                    {
                        "iteration": iteration,
                        "before_screenshot": before_name,
                        "vla_output": vla_out,
                    }
                )

    return missing

jsonl_path = "vla_evaluation/metadata.jsonl"
no_centre = list_examples_without_centre(jsonl_path)

# Print a quick summary
for ex in no_centre:
    print(ex["before_screenshot"], "->", ex["vla_output"])
