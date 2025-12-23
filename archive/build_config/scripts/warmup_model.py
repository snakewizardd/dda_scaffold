import asyncio
import httpx
import time
import sys

async def warmup():
    url = "http://127.0.0.1:1234/v1/chat/completions"
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.1,
        "max_tokens": 1
    }

    print(f"WARMING UP MODEL at {url}...")
    print("This forces LM Studio to load the model into VRAM/NPU.")
    print("Please check the LM Studio window to see if it's loading.")
    print("-" * 50)

    start_time = time.time()
    
    # Try for up to 5 minutes
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            print("Sending 'ping' request (timeout=300s)...")
            response = await client.post(url, json=payload)
            
            elapsed = time.time() - start_time
            print(f"Response Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"\nSUCCESS! Model responded in {elapsed:.1f} seconds.")
                print("You can now launch the visualization.")
            else:
                print(f"\nFailed with status: {response.status_code}")
                print(response.text)

        except httpx.ReadTimeout:
            print("\nTIMEOUT: The request took longer than 300 seconds.")
            print("The model might be too large for this machine or hung.")
        except httpx.ConnectError:
            print("\nCONNECTION ERROR: Could not reach LM Studio on port 1234.")
        except Exception as e:
            print(f"\nERROR: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(warmup())
    except KeyboardInterrupt:
        print("\nWarmup cancelled by user.")
