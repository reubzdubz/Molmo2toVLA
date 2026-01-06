import requests

# Health check
response = requests.get("http://localhost:8001/health")
print(response.json())

# Get agent status
response = requests.get("http://localhost:8001/status")
print(response.json())

# Run single iteration
response = requests.post(
    "http://localhost:8001/run_iteration",
    data={"prompt": "Center the crosshair on the nearest enemy. Be brief."}
)
print(response.json())

# Infinite loop (iterations=0)
response = requests.post(
    "http://localhost:8001/start_loop",
    data={"iterations": 50, "delay_ms": 3000}
)
print(response.json())
