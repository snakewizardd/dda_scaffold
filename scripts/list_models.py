import httpx
import json

def list_models():
    url = "http://127.0.0.1:1234/v1/models"
    print(f"GET {url}...")
    try:
        response = httpx.get(url, timeout=5.0)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Available models:")
            for model in data['data']:
                print(f" - {model['id']}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    list_models()
