from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, TextStreamer
import torch
import re
from PIL import Image
import requests
from peft import PeftModel
import time

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=[
        # Module names can also be relative like "ff_norm" which would apply to all such layers
        "model.vision_backbone", "model.transformer.ff_out", "model.transformer.ln_f"
    ]
)

model_id="allenai/Molmo2-4B"

# load the processor
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="auto",
    token=True
)

# load the model
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="auto",
    quantization_config=nf4_config,
    token=True
)
streamer = TextStreamer(tokenizer=processor.tokenizer, skip_special_tokens=True)


model = PeftModel.from_pretrained(model, "checkpoint-8100")

COORD_REGEX = re.compile(rf"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
FRAME_REGEX = re.compile(rf"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")

def _points_from_num_str(text, image_w, image_h, extract_ids=False):
    all_points = []
    for points in POINTS_REGEX.finditer(text):
        ix, x, y = points.group(1), points.group(2), points.group(3)
        # our points format assume coordinates are scaled by 1000
        x, y = float(x)/1000*image_w, float(y)/1000*image_h
        if 0 <= x <= image_w and 0 <= y <= image_h:
            yield ix, x, y


def extract_multi_image_points(text, image_w, image_h, extract_ids=False):
    """Extract pointing coordinates as a flattened list of (frame_id, x, y) triplets from model output text."""
    all_points = []
    if isinstance(image_w, (list, tuple)) and isinstance(image_h, (list, tuple)):
        assert len(image_w) == len(image_h)
        diff_res = True
    else:
        diff_res = False
    for coord in COORD_REGEX.finditer(text):
        for point_grp in FRAME_REGEX.finditer(coord.group(1)):
            frame_id = int(point_grp.group(1)) if diff_res else float(point_grp.group(1))
            w, h = (image_w[frame_id-1], image_h[frame_id-1]) if diff_res else (image_w, image_h)
            for idx, x, y in _points_from_num_str(point_grp.group(2), w, h):
                if extract_ids:
                    all_points.append((frame_id, idx, int(x), int(y)))
                else:
                    all_points.append((frame_id, int(x), int(y)))
    return all_points



target = "soldier"

PROMPT= f"""
"Point to the man and determine the action to be taken by the camera to align the centre of the image with it."
"""
# Capture game screen
images = [
    Image.open(f"fram2.png").convert("RGB"),
]

# Query model
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": PROMPT},
        dict(type="image", image=images[0])
    ]
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

# generate output
start_time = time.time()
with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
    generated_ids = model.generate(**inputs, streamer=streamer, max_new_tokens=128)

# only get generated tokens; decode them to text
print(len(generated_ids[0]))
generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
end_time = time.time()
print(f"Generation Time: {end_time - start_time} seconds")
print("Tokens per second:", len(generated_tokens)/(end_time - start_time))

