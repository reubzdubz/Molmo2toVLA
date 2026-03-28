from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import httpx
import asyncio
import pyautogui
import keyboard
import json
import math
import winsound
from PIL import ImageGrab
from pathlib import Path
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration
WSL_SERVER_URL = "http://localhost:8000"
GAME_DELAY_MS = 3000
SCREENSHOTS_DIR = Path("vla_evaluation")
METADATA_FILE = SCREENSHOTS_DIR / "metadata.jsonl"
target = "soldier"

SYSTEM_PROMPT = f"Point to the {target} and determine the action to be taken by the camera to align the centre of the image with it." + ' Output in the following format: <points coords="1 1 x y">item</points><points coords="1 1 500 500">centre of image</points>. ACTION:(dx, dy)'
        

# Create directories
SCREENSHOTS_DIR.mkdir(exist_ok=True)

class GameAgent:
    def __init__(self):
        self.previous_screenshot = None
        self.last_screenshot = None
        self.last_commands = None
        self.iteration_count = 0
        self.paused = False
        
        # Initialize metadata file with header if it doesn't exist
        if not METADATA_FILE.exists():
            with open(METADATA_FILE, 'w') as f:
                f.write("")  # Empty file, will append JSONL
    

        numbers = []
        for fname in os.listdir(SCREENSHOTS_DIR):
            if fname.endswith(".png"):
                # split on "_" then take the last part before ".png"
                num_str = fname.split("_")[-1].split(".")[0]
                numbers.append(int(num_str))

        # initialize with largest number + 1
        self.iteration_count = max(numbers) + 1 if numbers else 1

        print("Resuming from latest iteration:", self.iteration_count)

    
    async def capture_screenshot(self, prefix: str) -> bytes:
        """Capture Windows screen and return as bytes with naming."""
        logger.info(f"Capturing {prefix} screenshot...")
        screenshot = ImageGrab.grab()
        
        # Save with timestamped name
        filename = f"{prefix}_{self.iteration_count:04d}.png"
        filepath = SCREENSHOTS_DIR / filename
        screenshot.save(filepath)
        logger.info(f"Screenshot saved to {filepath}")
        
        # Convert to bytes
        import io
        img_bytes = io.BytesIO()
        screenshot.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue(), filename
    
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
                                    commands["raw_output"] += text
                                
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
                executed["up"] = duration

            # Down arrow - move camera down (dy negative)
            if commands.get("down", 0) > 0:
                duration = commands["down"] / 1000.0
                logger.info(f"Pressing DOWN arrow for {duration:.2f}s")
                keyboard.press('down')
                await asyncio.sleep(duration)
                keyboard.release('down')
                executed["down"] = duration

            # Left arrow - move camera left (dx positive)
            if commands.get("left", 0) > 0:
                duration = commands["left"] / 1000.0
                logger.info(f"Pressing LEFT arrow for {duration:.2f}s")
                keyboard.press('left')
                await asyncio.sleep(duration)
                keyboard.release('left')
                executed["left"] = duration

            # Right arrow - move camera right (dx negative)
            if commands.get("right", 0) > 0:
                duration = commands["right"] / 1000.0
                logger.info(f"Pressing RIGHT arrow for {duration:.2f}s")
                keyboard.press('right')
                await asyncio.sleep(duration)
                keyboard.release('right')
                executed["right"] = duration

            # Exit condition
            if commands.get("exit", 0) > 0:
                logger.info("Exit command received - task complete")
                keyboard.press('esc')
                await asyncio.sleep(0.5)
                keyboard.release('esc')
                executed["exit"] = True
                logger.info("Target aligned - task completed")

            logger.info("Commands executed successfully")

        except Exception as e:
            logger.error(f"Error executing commands: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        return executed
    
    async def get_human_correction(self) -> dict:
        """Wait for human arrow key presses and log durations for verification."""
        winsound.Beep(1000, 300)  # 1000 Hz beep for 300ms as audio cue
        logger.info("🔄 Human verification: Hold arrow keys to correct/verify (ESC to skip).")
        human_correction = {"up": 0, "down": 0, "left": 0, "right": 0}
        
        try:
            # Poll for key presses (non-blocking, like execute_commands)
            start_time = asyncio.get_event_loop().time()
            while True:
                await asyncio.sleep(0.05)  # Fast poll for responsive input
                current_time = asyncio.get_event_loop().time()
                
                if keyboard.is_pressed('esc'):  # Skip/finish
                    logger.info("ESC pressed - verification skipped")
                    break
                
                # Check each arrow and accumulate time held
                if keyboard.is_pressed('up'):
                    duration = current_time - start_time
                    human_correction["up"] = max(human_correction["up"], duration)
                if keyboard.is_pressed('down'):
                    duration = current_time - start_time
                    human_correction["down"] = max(human_correction["down"], duration)
                if keyboard.is_pressed('left'):
                    duration = current_time - start_time
                    human_correction["left"] = max(human_correction["left"], duration)
                if keyboard.is_pressed('right'):
                    duration = current_time - start_time
                    human_correction["right"] = max(human_correction["right"], duration)
                
                # Timeout after ~10s to avoid infinite wait
                if current_time - start_time > 10:
                    logger.info("Verification timeout")
                    break
                    
        except Exception as e:
            logger.error(f"Human correction error: {e}")
        
        def _round_sig(x, sig=3):
            if x == 0:
                return 0
            return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

        human_correction = {k: _round_sig(v) for k, v in human_correction.items()}
        logger.info(f"Human correction logged: {human_correction}")
        return human_correction

    
    def save_metadata(self, metadata: dict):
        """Append metadata entry to JSONL file."""
        with open(METADATA_FILE, 'a') as f:
            f.write(json.dumps(metadata) + '\n')
        logger.info(f"Metadata saved to {METADATA_FILE}")

# Global agent instance
agent = GameAgent()

@app.post("/run_iteration")
async def run_iteration(prompt: str = SYSTEM_PROMPT):
    """Run one full iteration: capture before → analyze → capture after → execute."""
    try:
        # Increment iteration counter
        agent.iteration_count += 1
        iteration_id = agent.iteration_count
        timestamp = datetime.now().isoformat()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Iteration {iteration_id}")
        logger.info(f"{'='*60}")
        
        # 1. Capture BEFORE screenshot
        before_bytes, before_filename = await agent.capture_screenshot("before")
        
        # 2. Send to Molmo for analysis
        commands = await agent.send_to_molmo(before_bytes, prompt)
        
        # 3. Execute commands (actuation)
        executed = await agent.execute_commands(commands)
        
        # 4. Wait a moment for game to settle after actuation
        await asyncio.sleep(0.5)
        
        # 5. Capture AFTER screenshot
        after_bytes, after_filename = await agent.capture_screenshot("after")

        # 6. Get human correction/verification (optional, can be used for evaluation)
        human_correction = await agent.get_human_correction()
        
        # 7. Save metadata
        metadata = {
            "iteration": iteration_id,
            "timestamp": timestamp,
            "before_screenshot": before_filename,
            "after_screenshot": after_filename,
            "prompt": prompt,
            "vla_output": commands.get("raw_output", ""),
            "commands": {
                "up": commands.get("up", 0),
                "down": commands.get("down", 0),
                "left": commands.get("left", 0),
                "right": commands.get("right", 0),
                "exit": commands.get("exit", 0)
            },
            "executed_durations": executed,
            "human_correction": human_correction,
        }
        agent.save_metadata(metadata)
        
        logger.info(f"Iteration {iteration_id} completed successfully")
        logger.info(f"Before: {before_filename}, After: {after_filename}")
        
        return JSONResponse({
            "status": "success",
            "iteration": iteration_id,
            "before_screenshot": str(before_filename),
            "after_screenshot": str(after_filename),
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
        "iteration_count": agent.iteration_count,
        "screenshots_dir": str(SCREENSHOTS_DIR),
        "metadata_file": str(METADATA_FILE),
        "last_commands": agent.last_commands
    })

@app.post("/start_loop")
async def start_loop(
    iterations: int = 50,
    delay_ms: int = GAME_DELAY_MS,
    prompt: str = SYSTEM_PROMPT,
    wait_for_keypress: bool = True  # NEW: Wait for keypress between iterations
):
    """Start continuous game loop with optional keypress wait."""
    logger.info(f"Starting game loop: {iterations if iterations > 0 else 'infinite'} iterations")
    logger.info(f"Wait for keypress: {wait_for_keypress}")
    
    iteration = 0
    try:
        while iterations == 0 or iteration < iterations:
            iteration += 1
            
            # Wait for SPACE key before starting iteration
            if wait_for_keypress:
                logger.info(f"\n Press SPACE to start iteration {agent.iteration_count + 1} (or 'p' to quit)...")
                
                # Async wait for key
                while True:
                    await asyncio.sleep(0.1)
                    if keyboard.is_pressed('space'):
                        logger.info("▶️  Starting iteration...")
                        await asyncio.sleep(0.3)  # Debounce
                        break
                    elif keyboard.is_pressed('p'):
                        logger.info("❌ Quit key pressed, stopping loop")
                        raise KeyboardInterrupt
            
            try:
                # Run iteration
                result = await run_iteration(prompt)
                if result.status_code == 200:
                    logger.info(f"✓ Iteration {agent.iteration_count} completed")
                else:
                    logger.warning(f"✗ Iteration {agent.iteration_count} failed: {result}")
            
            except Exception as e:
                logger.error(f"Iteration {agent.iteration_count} error: {e}")
            
            # Delay before next iteration (if not waiting for keypress)
            if not wait_for_keypress and (iterations == 0 or iteration < iterations):
                logger.info(f"Waiting {delay_ms}ms before next iteration...")
                await asyncio.sleep(delay_ms / 1000.0)
        
        logger.info(f"Loop completed after {iteration} iterations")
        return JSONResponse({
            "status": "complete",
            "iterations_completed": iteration,
            "total_screenshots": agent.iteration_count * 2,
            "screenshots_dir": str(SCREENSHOTS_DIR),
            "metadata_file": str(METADATA_FILE)
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
    return {"status": "ok", "client": "FPS VLA Agent"}

@app.post("/reset_counter")
async def reset_counter():
    """Reset iteration counter (useful for new evaluation runs)."""
    agent.iteration_count = 0
    logger.info("Iteration counter reset to 0")
    return {"status": "ok", "iteration_count": 0}

if __name__ == "__main__":
    logger.info("Starting FPS VLA Agent Client on http://0.0.0.0:8001")
    logger.info(f"Screenshots will be saved to: {SCREENSHOTS_DIR}")
    logger.info(f"Metadata will be saved to: {METADATA_FILE}")
    logger.info("\n🎮 Controls:")
    logger.info("  SPACE - Start next iteration (when wait_for_keypress=True)")
    logger.info("  P     - Quit loop")
    uvicorn.run(app, host="0.0.0.0", port=8001)
