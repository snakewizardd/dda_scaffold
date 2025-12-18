import asyncio
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm.hybrid_provider import HybridProvider, PersonalityParams

async def debug_provider():
    print("Initializing HybridProvider...")
    provider = HybridProvider(
        lm_studio_url="http://127.0.0.1:1234",
        lm_studio_model="openai/gpt-oss-20b",
        ollama_url="http://localhost:11434",
        embed_model="nomic-embed-text"
    )

    print("\nAttempting completion...")
    try:
        response = await provider.complete(
            prompt="Hello, are you working?",
            temperature=0.7,
            max_tokens=50
        )
        print(f"\nSUCCESS! Response:\n{response}")
    except Exception as e:
        print(f"\nFAILURE! Exception type: {type(e)}")
        print(f"Exception message: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Check if it's an HTTP error and try to print the response body
        if hasattr(e, 'response'):
            print(f"\nHTTP Response Status: {e.response.status_code}")
            print(f"HTTP Response Text: {e.response.text}")

if __name__ == "__main__":
    asyncio.run(debug_provider())
