import json
import os
import requests

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "minimaxai/minimax-m3.1-flash"

def call_model(messages):
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("請先設定 NVIDIA_API_KEY 環境變數！")
        
    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 1,
            "top_p": 0.95,
        },
        timeout=60,
    )
    print("status:", resp.status_code)
    data = resp.json()
    print(json.dumps(data, ensure_ascii=False, indent=2))
    
    if resp.ok:
        print("reply:", data["choices"][0]["message"]["content"])
    else:
        print("error:", data.get("error", data))
    return data

if __name__ == "__main__":
    call_model([{"role": "user", "content": "用一句話說明什麼是 HTTP"}])
