"""
Test LLM Connection and Show Real-Time Streaming
This will verify LM Studio and Ollama are working and demonstrate the streaming.
"""

import asyncio
import httpx
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def test_lm_studio():
    """Test LM Studio connection and streaming."""
    print("\n" + "="*60)
    print("TESTING LM STUDIO CONNECTION")
    print("="*60)

    try:
        # Test basic connection
        async with httpx.AsyncClient() as client:
            # Check if LM Studio is running
            response = await client.get(
                "http://127.0.0.1:1234/v1/models",
                timeout=5.0
            )

            if response.status_code == 200:
                models = response.json()
                print("✅ LM Studio is running!")
                print(f"Available models: {json.dumps(models, indent=2)}")
            else:
                print(f"❌ LM Studio returned status {response.status_code}")
                return False

            # Test completion with streaming
            print("\n📡 Testing streaming completion...")
            print("-" * 40)

            stream_response = await client.post(
                "http://127.0.0.1:1234/v1/chat/completions",
                json={
                    "model": "local-model",  # LM Studio uses "local-model" as default
                    "messages": [
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": "Say 'Hello from LM Studio!' and explain what DDA-X is in one sentence."}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 50,
                    "stream": True
                },
                timeout=30.0
            )

            print("Streaming response: ", end="", flush=True)

            # Parse SSE stream
            buffer = ""
            async for line in stream_response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                token = delta["content"]
                                print(token, end="", flush=True)
                                buffer += token
                                await asyncio.sleep(0.01)  # Small delay to see streaming
                    except json.JSONDecodeError:
                        pass

            print("\n" + "-" * 40)
            print(f"✅ Streaming works! Received: {len(buffer)} characters")
            return True

    except httpx.ConnectError:
        print("❌ Cannot connect to LM Studio at http://127.0.0.1:1234")
        print("   Make sure LM Studio is running with a model loaded")
        return False
    except Exception as e:
        print(f"❌ Error testing LM Studio: {e}")
        return False


async def test_ollama():
    """Test Ollama connection for embeddings."""
    print("\n" + "="*60)
    print("TESTING OLLAMA CONNECTION")
    print("="*60)

    try:
        import ollama

        # Create client
        client = ollama.Client(host="http://localhost:11434")

        # Check if running
        models = client.list()
        print("✅ Ollama is running!")
        print(f"Available models: {[m['name'] for m in models.get('models', [])]}")

        # Check for embedding model
        has_nomic = any("nomic-embed-text" in m['name'] for m in models.get('models', []))

        if not has_nomic:
            print("\n⚠️  nomic-embed-text not found")
            print("Installing (this may take a moment)...")
            import subprocess
            result = subprocess.run(["ollama", "pull", "nomic-embed-text"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ nomic-embed-text installed successfully!")
            else:
                print(f"❌ Failed to install: {result.stderr}")
                return False
        else:
            print("✅ nomic-embed-text is available!")

        # Test embedding
        print("\n📊 Testing embedding generation...")
        test_text = "The DDA-X framework models cognitive rigidity"

        response = client.embeddings(
            model="nomic-embed-text",
            prompt=test_text
        )

        embedding = response.get("embedding", [])
        print(f"✅ Generated embedding with {len(embedding)} dimensions")
        print(f"   First 5 values: {embedding[:5]}")

        return True

    except Exception as e:
        print(f"❌ Error testing Ollama: {e}")
        print("   Make sure Ollama is running with: ollama serve")
        return False


async def test_hybrid_provider():
    """Test the actual HybridProvider integration."""
    print("\n" + "="*60)
    print("TESTING HYBRID PROVIDER")
    print("="*60)

    from src.llm.hybrid_provider import HybridProvider, PersonalityParams

    try:
        provider = HybridProvider(
            lm_studio_url="http://127.0.0.1:1234",
            lm_studio_model="local-model",
            ollama_url="http://localhost:11434",
            embed_model="nomic-embed-text"
        )

        # Test completion with personality modulation
        print("\n🧠 Testing personality-modulated generation...")

        # High rigidity (defensive)
        params_rigid = PersonalityParams.from_rigidity(0.8, "cautious")
        print(f"\nHigh rigidity (ρ=0.8): temp={params_rigid.temperature:.2f}")

        response_rigid = await provider.complete(
            prompt="What are the risks of AI development?",
            personality_params=params_rigid,
            max_tokens=50
        )
        print(f"Response: {response_rigid}")

        # Low rigidity (open)
        params_open = PersonalityParams.from_rigidity(0.2, "exploratory")
        print(f"\nLow rigidity (ρ=0.2): temp={params_open.temperature:.2f}")

        response_open = await provider.complete(
            prompt="What are the opportunities in AI development?",
            personality_params=params_open,
            max_tokens=50
        )
        print(f"Response: {response_open}")

        # Test embedding
        print("\n🔍 Testing embedding generation...")
        embedding = await provider.get_embedding("test text")
        print(f"✅ Embedding shape: {embedding.shape}")

        return True

    except Exception as e:
        print(f"❌ Error testing HybridProvider: {e}")
        return False


async def main():
    """Run all tests."""
    print("="*60)
    print("DDA-X LLM BACKEND TEST SUITE")
    print("="*60)

    lm_studio_ok = await test_lm_studio()
    ollama_ok = await test_ollama()

    if lm_studio_ok and ollama_ok:
        hybrid_ok = await test_hybrid_provider()

        if hybrid_ok:
            print("\n" + "="*60)
            print("✅ ALL SYSTEMS OPERATIONAL!")
            print("="*60)
            print("\nYour LLM backends are working perfectly!")
            print("\nTo see it in the visualization:")
            print("1. Keep LM Studio running")
            print("2. Keep Ollama running")
            print("3. Run: python visualization/debate_server.py")
            print("4. Open: visualization/multi_agent_debate.html")
            print("5. Click 'Start Debate' to see REAL LLM responses!")
            print("\nThe HTML will connect via WebSocket and stream actual tokens.")
        else:
            print("\n⚠️  Backends work individually but HybridProvider failed")
    else:
        print("\n" + "="*60)
        print("⚠️  SOME BACKENDS NOT AVAILABLE")
        print("="*60)

        if not lm_studio_ok:
            print("\nTo fix LM Studio:")
            print("1. Open LM Studio")
            print("2. Load a model (e.g., GPT-OSS-20B)")
            print("3. Make sure it's running on port 1234")

        if not ollama_ok:
            print("\nTo fix Ollama:")
            print("1. Install Ollama from https://ollama.ai")
            print("2. Run: ollama serve")
            print("3. Run: ollama pull nomic-embed-text")

        print("\nThe visualization will still work in DEMO MODE without these.")


if __name__ == "__main__":
    asyncio.run(main())