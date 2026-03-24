import requests

# Health check for agent and molmo-service
response = requests.get("http://localhost:8001/health")
print(response.json())
response = requests.get("http://localhost:8000/health")
print(response.json())

# Get agent status
response = requests.get("http://localhost:8001/status")
print(response.json())

# Infinite loop (iterations=0)
response = requests.post(
    "http://localhost:8001/start_loop",
    data={"iterations": 100, "delay_ms": 3000}
)
print(response.json())
