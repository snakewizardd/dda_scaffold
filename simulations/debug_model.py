import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(".env")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def test():
    print("Testing gpt-5-nano...")
    try:
        response = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Explain quantum physics in one sentence."}],
            max_completion_tokens=100
        )
        print("Raw Response:", response)
        print("Content:", response.choices[0].message.content)
    except Exception as e:
        print(f"Error Type: {type(e)}")
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test())
