from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import httpx
import asyncio
import pyautogui
import keyboard
import json
from PIL import ImageGrab
from pathlib import Path
import logging
import os
import win32gui, win32ui, win32con
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration
WSL_SERVER_URL = "http://localhost:8000"
GAME_DELAY_MS = 3000
SCREENSHOT_SAVE_PATH = Path("current_frame.png")
PREVIOUS_SAVE_PATH = Path("previous_frame.png")
target = "Battleship Yamato"

SYSTEM_PROMPT= f"Point to the {target} and determine the action to be taken by the camera to align the centre of the image with it." + ' Output in the following format: <points coords="1 1 x y">item</points><points coords="1 1 500 500">centre of image</points>. ACTION:(dx, dy)'


class WindowCapture:
    def __init__(self, window_name: str):
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception(f"Window not found: {window_name}")

        window_rect = win32gui.GetWindowRect(self.hwnd)
        w = window_rect[2] - window_rect[0]
        h = window_rect[3] - window_rect[1]

        border_pixels = 8
        titlebar_pixels = 30

        self.w = w - (border_pixels * 2)
        self.h = h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

    def get_pil_image(self) -> Image.Image:
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj,
                   (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        img = img[..., :3]  # drop alpha
        return Image.fromarray(img)


WINDOW_TITLE = "Your Game Window Title"

class GameAgent:
    def __init__(self):
        self.previous_screenshot = None
        self.last_screenshot = None
        self.last_commands = None
        self.window_capture = WindowCapture(WINDOW_TITLE)

    async def capture_screenshot(self) -> tuple[bytes | None, bytes]:
        import io

        logger.info("Capturing window screenshot via Win32...")
        screenshot = self.window_capture.get_pil_image()

        if os.path.exists(SCREENSHOT_SAVE_PATH):
            if os.path.exists(PREVIOUS_SAVE_PATH):
                os.remove(PREVIOUS_SAVE_PATH)
            os.rename(SCREENSHOT_SAVE_PATH, PREVIOUS_SAVE_PATH)
            os.remove(SCREENSHOT_SAVE_PATH)

        screenshot.save(SCREENSHOT_SAVE_PATH)
        logger.info(f"Screenshot saved to {SCREENSHOT_SAVE_PATH}")

        img_bytes = io.BytesIO()
        screenshot.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        self.last_screenshot = img_bytes.getvalue()

        self.previous_screenshot = None
        if os.path.exists(PREVIOUS_SAVE_PATH):
            with open(PREVIOUS_SAVE_PATH, "rb") as f:
                self.previous_screenshot = f.read()

        return self.previous_screenshot, self.last_screenshot

    
    async def send_to_molmo(self, image_bytes: bytes, prompt: str) -> dict:
        """Send screenshot to Molmo2-4B and get streamed response."""
        logger.info(f"Sending screenshots to {WSL_SERVER_URL}/analyze")
        commands = {
            "up": 0,
            "down": 0,
            "left": 0,
            "right": 0,
            "exit": 0,
            "raw_output": ""
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                # Send as multipart form
                files = {"file": ("current_frame.png", image_bytes, "image/png")}
                data = {"prompt": prompt}
                
                async with client.stream("POST", f"{WSL_SERVER_URL}/analyze", files=files, data=data) as response:
                    if response.status_code != 200:
                        logger.error(f"Server error: {response.status_code}")
                        raise HTTPException(status_code=response.status_code, detail="Molmo server error")
                    
                    # Process streamed NDJSON response
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                json_obj = json.loads(line)
                                status = json_obj.get("status")
                                
                                if status == "model_output":
                                    text = json_obj.get("text", "")
                                    logger.info(f"[MODEL] {text}")
                                    commands["raw_output"] = text
                                
                                elif status == "commands":
                                    data = json_obj.get("data", {})
                                    commands.update(data)
                                    logger.info(f"[COMMANDS] {data}")
                                
                                elif status == "processing":
                                    logger.info(f"[STATUS] {json_obj.get('message')}")
                                
                                elif status == "error":
                                    logger.error(f"[ERROR] {json_obj.get('message')}")
                            
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse JSON: {line}")
        
        except httpx.ConnectError:
            logger.error(f"Cannot connect to Molmo server at {WSL_SERVER_URL}")
            raise HTTPException(status_code=503, detail="Molmo server not reachable")
        except Exception as e:
            logger.error(f"Error communicating with Molmo: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
        self.last_commands = commands
        return commands
    
    async def execute_commands(self, commands: dict) -> dict:
        """Execute keyboard commands based on parsed output."""
        logger.info("Executing movement commands...")
        executed = {}

        try:
            # Up arrow - move camera up (dy positive)
            if commands.get("up", 0) > 0:
                duration = commands["up"] / 1000.0
                logger.info(f"Pressing UP arrow for {duration:.2f}s")
                keyboard.press('up')
                await asyncio.sleep(duration)
                keyboard.release('up')
                executed["up"] = True

            # Down arrow - move camera down (dy negative)
            if commands.get("down", 0) > 0:
                duration = commands["down"] / 1000.0
                logger.info(f"Pressing DOWN arrow for {duration:.2f}s")
                keyboard.press('down')
                await asyncio.sleep(duration)
                keyboard.release('down')
                executed["down"] = True

            # Left arrow - move camera left (dx positive)
            if commands.get("left", 0) > 0:
                duration = commands["left"] / 1000.0
                logger.info(f"Pressing LEFT arrow for {duration:.2f}s")
                keyboard.press('left')
                await asyncio.sleep(duration)
                keyboard.release('left')
                executed["left"] = True

            # Right arrow - move camera right (dx negative)
            if commands.get("right", 0) > 0:
                duration = commands["right"] / 1000.0
                logger.info(f"Pressing RIGHT arrow for {duration:.2f}s")
                keyboard.press('right')
                await asyncio.sleep(duration)
                keyboard.release('right')
                executed["right"] = True

            # Exit condition
            if commands.get("exit", 0) > 0:
                logger.info("Exit command received - task complete")
                keyboard.press('esc')
                await asyncio.sleep(0.5)
                keyboard.release('esc')
                executed["exit"] = True
                print("Target aligned - task completed")
                exit()

            logger.info("Commands executed successfully")

        except Exception as e:
            logger.error(f"Error executing commands: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        return executed

# Global agent instance
agent = GameAgent()

@app.post("/run_iteration")
async def run_iteration(prompt: str = SYSTEM_PROMPT):
    """Run one full iteration: capture → analyze → execute."""
    try:
        # Capture screenshot
        _, image_bytes = await agent.capture_screenshot()

        # Send to Molmo
        commands = await agent.send_to_molmo(image_bytes, prompt)

        # Execute commands
        executed = await agent.execute_commands(commands)

        return JSONResponse({
            "status": "success",
            "model_output": commands.get("raw_output"),
            "commands": {
                "up": commands.get("up", 0),
                "down": commands.get("down", 0),
                "left": commands.get("left", 0),
                "right": commands.get("right", 0),
                "exit": commands.get("exit", 0)
            },
            "executed": executed
        })

    except Exception as e:
        logger.error(f"Iteration failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/status")
async def status():
    """Get agent status."""
    return JSONResponse({
        "status": "ok",
        "molmo_server": WSL_SERVER_URL,
        "game_delay_ms": GAME_DELAY_MS,
        "last_commands": agent.last_commands
    })

@app.post("/start_loop")
async def start_loop(
    iterations: int = 50,
    delay_ms: int = GAME_DELAY_MS,
    prompt: str = SYSTEM_PROMPT
):
    """Start continuous game loop (0 = infinite)."""
    logger.info(f"Starting game loop: {iterations if iterations > 0 else 'infinite'} iterations, {delay_ms}ms delay")
    
    iteration = 0
    try:
        while iterations == 0 or iteration < iterations:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration}")
            logger.info(f"{'='*60}")
            
            try:
                # Run iteration
                result = await run_iteration(prompt)
                if result.status_code == 200:
                    logger.info(f"Iteration {iteration} completed successfully")
                else:
                    logger.warning(f"Iteration {iteration} failed: {result}")
            
            except Exception as e:
                logger.error(f"Iteration {iteration} error: {e}")
            
            # Delay before next iteration
            if iterations == 0 or iteration < iterations:
                logger.info(f"Waiting {delay_ms}ms before next iteration...")
                await asyncio.sleep(delay_ms / 3000.0)
        
        logger.info(f"Loop completed after {iteration} iterations")
        return JSONResponse({
            "status": "complete",
            "iterations_completed": iteration
        })
    
    except KeyboardInterrupt:
        logger.info(f"Loop interrupted after {iteration} iterations")
        return JSONResponse({
            "status": "interrupted",
            "iterations_completed": iteration
        })

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "client": "FPS Agent"}

if __name__ == "__main__":
    logger.info("Starting FPS Agent Client on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
