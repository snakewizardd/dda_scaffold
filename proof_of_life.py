import requests
import json
import time
import sys

# ANSI Colors
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def log(msg, color=RESET):
    print(f"{color}{msg}{RESET}")

def proof_of_life():
    base_url = "http://127.0.0.1:1234"
    
    # 1. Test Connectivity (Instant)
    log("--- STEP 1: Connectivity Check (/v1/models) ---")
    try:
        start = time.time()
        resp = requests.get(f"{base_url}/v1/models", timeout=10)
        elapsed = time.time() - start
        
        if resp.status_code == 200:
            log(f"SUCCESS: Connected in {elapsed:.2f}s", GREEN)
            models = resp.json()
            model_id = models["data"][0]["id"]
            log(f"Found model: {model_id}", GREEN)
        else:
            log(f"FAILED: Status {resp.status_code} - {resp.text}", RED)
            return
    except Exception as e:
        log(f"FAILED: Could not reach server. {e}", RED)
        return

    # 2. Test Generation (Long Timeout)
    log("\n--- STEP 2: Generation Check (Timeout=600s) ---")
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "Reply with one word: ALIVE"}],
        "temperature": 0.1,
        "max_tokens": 10,
        "stream": False
    }
    
    try:
        log("Sending request... (This may take 2+ minutes if model is thinking)", RESET)
        start = time.time()
        resp = requests.post(
            f"{base_url}/v1/chat/completions", 
            json=payload, 
            timeout=600  # 10 minutes
        )
        elapsed = time.time() - start
        
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message']['content']
            log(f"SUCCESS: Received response in {elapsed:.2f}s", GREEN)
            log(f"Response: {content}", GREEN)
        else:
            log(f"FAILED: Status {resp.status_code}", RED)
            log(resp.text, RED)
            
    except Exception as e:
        log(f"FAILED: Expectation Error: {e}", RED)

if __name__ == "__main__":
    proof_of_life()
