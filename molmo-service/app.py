from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import uvicorn
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from PIL import Image
import io
import json
import re
from peft import PeftModel

app = FastAPI()

# Load Molmo2-4B on startup
print("Loading Molmo2-4B model...")

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

model = PeftModel.from_pretrained(model, "checkpoint-3000")

print("Model loaded successfully!")

def parse_molmo_output(text: str) -> dict:
    """Parse Molmo output to extract movement commands from new format.
    
    Expected format:
    The closest person in the image is at <points coords="...">closest person</points> 
    while the centre of the image is at <points coords="...">centre of image</points>. 
    The action to be taken is therefore (dx, dy)
    """
    commands = {
        "up": 0,      # up arrow (negative dy)
        "down": 0,    # down arrow (positive dy)
        "left": 0,    # left arrow (negative dx)
        "right": 0,   # right arrow (positive dx)
        "exit": 0,
        "raw_output": text
    }
    
    text_lower = text.lower()
    
    # Check for exit condition
    if "exit" in text_lower and text_lower.strip() == "exit":
        commands["exit"] = 1
        return commands
    
    # Extract action vector (dx, dy) from format: (value, value)
    action_pattern = r'\((-?\d+)\s*,\s*(-?\d+)\)'
    action_match = re.search(action_pattern, text)
    
    if action_match:
        dx = int(action_match.group(1))
        dy = int(action_match.group(2))
        
        print(f"Extracted action vector: dx={dx}, dy={dy}")
        
        # Convert action vector to keyboard commands
        # Remember: left/up is positive, right/down is negative
        # dx: positive = move left, negative = move right
        # dy: positive = move up, negative = move down
        
        # Horizontal movement (dx)
        if dx > 0:
            commands["left"] = abs(dx)
        elif dx < 0:
            commands["right"] = abs(dx)
        
        # Vertical movement (dy)
        if dy > 0:
            commands["up"] = abs(dy)
        elif dy < 0:
            commands["down"] = abs(dy)
    
    return commands

async def stream_molmo_response(image_bytes: bytes, prompt: str, previous_bytes: bytes = None):
    """Stream Molmo2-4B response."""
    try:
        # Load image
        if previous_bytes:
            previous_image = Image.open(io.BytesIO(previous_bytes)).convert("RGB")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Prepare messages
        if previous_bytes:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": previous_image},
                        {"type": "image", "image": image}
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image}
                    ]
                }
            ]
        
        # Yield progress update
        yield json.dumps({"status": "processing", "message": "Analyzing screenshot with Molmo2-4B..."}) + "\n"
        
        # Apply chat template
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        # Generate with streaming (token-by-token)
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
        
        # Only get generated tokens
        generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Yield model output
        yield json.dumps({"status": "model_output", "text": generated_text}) + "\n"
        
        # Parse into commands
        commands = parse_molmo_output(generated_text)
        yield json.dumps({"status": "commands", "data": commands}) + "\n"
        
        yield json.dumps({"status": "complete"}) + "\n"
        
    except Exception as e:
        yield json.dumps({"status": "error", "message": str(e)}) + "\n"

@app.post("/analyze")
async def analyze_screenshot(file: UploadFile = File(...), prompt: str = Form("Center the crosshair on the target")):
    """Analyze screenshot and return streaming Molmo response."""
    image_bytes = await file.read()
    return StreamingResponse(
        stream_molmo_response(image_bytes, prompt),
        media_type="application/x-ndjson"
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model": "Molmo2-4B"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
