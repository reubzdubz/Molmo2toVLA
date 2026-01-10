import json
import re
from pathlib import Path

# Enforce: The action to be taken is therefore (dx, dy)
ACTION_REGEX = re.compile(
    r"The action to be taken is therefore\s*"
    r"\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)"
)

def has_valid_plain_action(text: str) -> bool:
    """
    Returns True iff the text contains a plain-text action of the form:
        The action to be taken is therefore (dx, dy)
    with no HTML tags in that clause.
    """
    m = ACTION_REGEX.search(text)
    if not m:
        return False

    # Ensure the matched substring does not contain any '<' (e.g. <points>)
    span_start, span_end = m.span()
    action_substr = text[span_start:span_end]
    if "<" in action_substr:
        return False

    return True


def list_point_only_examples(jsonl_path):
    """
    List examples that only did pointing and not a proper action.
    Criteria:
      - vla_output does NOT contain a valid plain-text action of the form
        'The action to be taken is therefore (dx, dy)'
      - all commands (up, down, left, right, exit) are zero
    Returns a list of dicts with: iteration, before_screenshot, vla_output, commands.
    """
    jsonl_path = Path(jsonl_path)
    results = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            vla_out = (data.get("vla_output") or "")
            commands = data.get("commands", {})
            before_name = data.get("before_screenshot")
            iteration = data.get("iteration")

            # New heuristic: proper action must match the strict regex
            has_proper_action = has_valid_plain_action(vla_out)

            cmd_up    = commands.get("up", 0)
            cmd_down  = commands.get("down", 0)
            cmd_left  = commands.get("left", 0)
            cmd_right = commands.get("right", 0)
            cmd_exit  = commands.get("exit", 0)

            all_zero_cmds = (
                cmd_up == 0 and cmd_down == 0 and
                cmd_left == 0 and cmd_right == 0 and
                cmd_exit == 0
            )

            # "Point-only, no proper action" condition
            if (not has_proper_action) and all_zero_cmds:
                results.append(
                    {
                        "iteration": iteration,
                        "before_screenshot": before_name,
                        "vla_output": vla_out,
                        "commands": commands,
                    }
                )

    return results


# Example usage
examples = list_point_only_examples("vla_evaluation/metadata.jsonl")
for ex in examples:
    print(ex["before_screenshot"], "->", ex["vla_output"])
