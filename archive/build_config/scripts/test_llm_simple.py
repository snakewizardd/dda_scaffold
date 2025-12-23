import asyncio
import httpx
import json
import traceback

async def test_connection():
    url = "http://127.0.0.1:1234/v1/chat/completions"
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
        "max_tokens": 5,
        "stream": False
    }

    print(f"Testing POST to {url}...")
    print(f"Payload: {json.dumps(payload)}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print("Response:", json.dumps(response.json(), indent=2))
                print("\nSUCCESS")
            else:
                print("Error Response:", response.text)
                print("\nFAILURE: HTTP status error")
    except Exception as e:
        print(f"\nEXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connection())
