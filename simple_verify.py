import requests
import json
import time

def verify_backend():
    url = "http://127.0.0.1:1234/v1/chat/completions"
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "user", "content": "Say the word TEST three times."}
        ],
        "temperature": 0.7,
        "max_tokens": 50,
        "stream": False 
    }

    print(f"Sending BLOCKING request to {url}...")
    start = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        elapsed = time.time() - start
        
        print(f"Response received in {elapsed:.2f}s")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            print("\n--- CONTENT START ---")
            print(content)
            print("--- CONTENT END ---\n")
            print("VERIFICATION SUCCESS: Backend is returning text.")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    verify_backend()
