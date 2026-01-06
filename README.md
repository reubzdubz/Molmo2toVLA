## Molmo2 VLM to VLA
Personal project to finetune [allenai/Molmo2-4B](https://huggingface.co/allenai/Molmo2-4B) for a VLA using LoRA adaptor [reubk/Molmo2toVLA-4B](https://huggingface.co/reubk/Molmo2toVLA-4B)

VLM is fed live gameplay footage on a First Person Shooter, and is prompted to point to a defined target and adjust the camera using up, down, left, right keys to centre the target object

*uv* is utilized to manage python packages.

Since I'm hosting this on Windows, async is used to run the VLM services and screen capture and keyboard actuation seperately.

Enviroment on Windows with access to screen capture and keyboard:
```
uv .venv --python=3.12
source .venv/Scripts/activate
uv pip install fastapi uvicorn httpx keyboard pillow
uv pip install pyautogui
```

Windows Subsystem for Linux (WSL) is used host the Molmo2 VLM and LoRA adaptor due to decord2:
```
uv .venv --python=3.11
source .venv/Scripts/activate
uv pip install transformers==4.57.1
uv pip install torch pillow einops torchvision accelerate decord2 molmo_utils
```

To initialise agent with actuator on Powershell terminal:
```
uv run python fps_agent_client.py
```

On *WSL* terminal in the repository:
```
uv run python molmo-service/app.py
```

Finally, to run the two services together, in another Powershell Terminal:
```
uv run python orchestrator.py
```
