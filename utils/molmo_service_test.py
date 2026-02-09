import requests
import json
from PIL import Image, ImageDraw
import io

# Configuration
MOLMO_SERVER = "http://localhost:8000"
IMAGE_SIZE = (1920, 1080)
IMAGE_PATH = "shot (1).png"
target = "soldier in green"
PROMPT =        f"""
The image is a live screenshot of a FPS videogame which you have control of the cameraman, you can execute the following actions:
1. left 
2. right 
3. exit 

Your task is to move the player until the crosshair is centered on the {target}
First, point to the {target} and the crosshair 
Second, consider the distance from the {target} and the crosshair, which direction should the camera move to align the crosshair with the {target}? Simply respond with the following format: [action, duration (miliseconds)]
If the {target} is already in the centre, respond with "exit" and only "exit"
"""



def send_screenshot_to_molmo(image_path: str, prompt: str):
    """Send screenshot to Molmo service and stream response."""
    print(f"\n{'='*60}")
    print(f"Sending to: {MOLMO_SERVER}/analyze")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}\n")
    
    try:
        # Prepare the request
        with open(image_path, 'rb') as f:
            files = {'file': (image_path, f, 'image/png')}
            data = {'prompt': prompt}
            
            # Stream the response
            response = requests.post(
                f"{MOLMO_SERVER}/analyze",
                files=files,
                data=data,
                stream=True,
                timeout=120
            )
            
            if response.status_code != 200:
                print(f"❌ Error: {response.status_code}")
                print(response.text)
                return
            
            print("📡 Streaming response from Molmo:\n")
            
            # Process NDJSON streamed response
            for line in response.iter_lines():
                if line:
                    try:
                        json_obj = json.loads(line)
                        status = json_obj.get('status')
                        
                        if status == 'processing':
                            
                            print(f"⏳ {json_obj.get('message')}")
                        
                        elif status == 'model_output':
                            print("Getting Output")
                            text = json_obj.get('text', '')
                            print(f"\n🤖 Molmo Output:\n{text}\n")
                        
                        elif status == 'commands':
                            print("Getting Commnads")
                            commands = json_obj.get('data', {})
                            print(f"⌨️  Parsed Commands:")
                            print(f"   Forward (W): {commands.get('forward', 0)}ms")
                            print(f"   Back (S):    {commands.get('back', 0)}ms")
                            print(f"   Left (A):    {commands.get('left', 0)}ms")
                            print(f"   Right (D):   {commands.get('right', 0)}ms")
                        
                        elif status == 'complete':
                            print(f"\n✓ Response complete")
                        
                        elif status == 'error':
                            print(f"❌ Error: {json_obj.get('message')}")
                    
                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSON parse error: {e}")
                        print(f"   Line: {line}")
    
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {MOLMO_SERVER}")
        print("   Make sure the Molmo server is running: python molmo_server.py")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_server_health():
    """Check if Molmo server is running."""
    try:
        response = requests.get(f"{MOLMO_SERVER}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ Molmo server is running: {response.json()}")
            return True
        else:
            print(f"❌ Server error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {MOLMO_SERVER}")
        print("   Make sure the Molmo server is running: python molmo_server.py")
        return False

if __name__ == "__main__":
    print("🎯 FPS Agent - Molmo Service Tester\n")
    
    # Check server health
    if not test_server_health():
        exit(1)
    image_path=IMAGE_PATH
    
    # Send to Molmo with different prompts
    prompts = [
        "Where is the enemy? Describe the location relative to the center of the screen.",
        "How should I move my crosshair to center it on the red enemy? Describe the direction.",
        "Center the crosshair on the target. What WASD keys should I press?",
        PROMPT
    ]
    
    for prompt in prompts:
        send_screenshot_to_molmo(image_path, prompt)
        print("\n" + "="*60 + "\n")
