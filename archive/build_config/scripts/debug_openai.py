import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

async def verify():
    api_key = os.getenv("OAI_API_KEY")
    if not api_key:
        print("No API Key found")
        return

    client = AsyncOpenAI(api_key=api_key)
    
    # 1. Test Embedding
    print("\nTesting text-embedding-3-large...")
    try:
        resp = await client.embeddings.create(
            input=["Hello world"],
            model="text-embedding-3-large",
            dimensions=3072
        )
        print(f"✅ Embedding Success. Dim: {len(resp.data[0].embedding)}")
    except Exception as e:
        print(f"❌ Embedding Failed: {e}")

    # 2. Test Completion (gpt-5.2)
    print("\nTesting gpt-5.2...")
    try:
        resp = await client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=10
        )
        print(f"✅ gpt-5.2 Success: {resp.choices[0].message.content}")
    except Exception as e:
        print(f"❌ gpt-5.2 Failed: {e}")
        
    # 3. Test Fallback (gpt-4o)
    print("\nTesting gpt-4o...")
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print(f"✅ gpt-4o Success: {resp.choices[0].message.content}")
    except Exception as e:
        print(f"❌ gpt-4o Failed: {e}")

if __name__ == "__main__":
    asyncio.run(verify())
