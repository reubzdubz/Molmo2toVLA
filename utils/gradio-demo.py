import gradio as gr
from PIL import ImageDraw, Image
import torch

# If this is in a separate file, do:
from test import processor, model, extract_multi_image_points

def run_pointing_demo(image: Image.Image, prompt: str):
    """
    1. Run Molmo on (image, prompt)
    2. Parse coords from generated_text
    3. Draw points on the image
    """
    if image is None or not prompt.strip():
        return image, "Please provide both an image and a prompt."

    # Build chat-style input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate output text
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
        )

    # Strip the prompt tokens, decode only new tokens
    generated_tokens = generated_ids[0, inputs["input_ids"].size(1):]
    generated_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )

    # Extract points from text using your helper
    w, h = image.size
    points = extract_multi_image_points(
        generated_text,
        image_w=w,
        image_h=h,
        extract_ids=False,  # (frame_id, x, y)
    )

    # Draw points on a copy of the image
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    radius = 6
    for i, (frame_id, x, y) in enumerate(points, start=1):
        x, y = int(x), int(y)
        bbox = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(bbox, outline="red", width=3)
        # Optional: label with index
        draw.text((x + radius + 2, y + radius + 2), str(i), fill="red")

    return annotated, generated_text


demo = gr.Interface(
    fn=run_pointing_demo,
    inputs=[
        gr.Image(type="pil", label="Input image"),
        gr.Textbox(
            lines=3,
            label="Prompt",
            value="Point to the man and determine the action to be taken by the camera to align the centre of the image with it.",
        ),
    ],
    outputs=[
        gr.Image(type="pil", label="Image with points"),
        gr.Textbox(lines=8, label="Raw model output"),
    ],
    title="Molmo Pointing Demo",
    description="Upload an image + prompt, Molmo outputs point coords, which are parsed and drawn on the image.",
)

if __name__ == "__main__":
    demo.launch()
