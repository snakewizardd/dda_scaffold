"""
DDA-X Interactive Visualization Launcher

This script launches the multi-agent debate visualization system.
It runs in two modes:
1. Full mode: With WebSocket server (requires LLM backends)
2. Demo mode: Standalone HTML with simulated agents
"""

import os
import sys
import time
import webbrowser
import subprocess
import asyncio
from pathlib import Path

def check_llm_backends():
    """Check if LLM backends are available."""
    import httpx

    backends_available = {
        "lm_studio": False,
        "ollama": False
    }

    # Check LM Studio
    try:
        response = httpx.get("http://127.0.0.1:1234/v1/models", timeout=2.0)
        if response.status_code == 200:
            backends_available["lm_studio"] = True
            print("✓ LM Studio detected at http://127.0.0.1:1234")
        else:
            print("✗ LM Studio not responding")
    except:
        print("✗ LM Studio not available (start it at port 1234)")

    # Check Ollama
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if response.status_code == 200:
            backends_available["ollama"] = True
            print("✓ Ollama detected at http://localhost:11434")
        else:
            print("✗ Ollama not responding")
    except:
        print("✗ Ollama not available (run 'ollama serve')")

    return backends_available

def launch_demo_mode():
    """Launch the visualization in demo mode (no backend required)."""
    print("\n" + "="*60)
    print("LAUNCHING IN DEMO MODE")
    print("="*60)
    print("Opening visualization with simulated agents...")
    print("No LLM backend required - using pre-scripted responses")
    print("="*60 + "\n")

    # Open the HTML file
    html_path = Path(__file__).parent / "visualization" / "multi_agent_debate.html"
    webbrowser.open(f"file:///{html_path.absolute()}")

    print("Visualization opened in browser!")
    print("\nFeatures available in demo mode:")
    print("- Simulated agent debates")
    print("- Live rigidity dynamics")
    print("- Trust matrix evolution")
    print("- Force field visualization")
    print("- Surprise event injection")

    print("\nPress Ctrl+C to exit")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting demo mode...")

async def launch_full_mode():
    """Launch the full system with WebSocket server."""
    print("\n" + "="*60)
    print("LAUNCHING IN FULL MODE")
    print("="*60)
    print("Starting WebSocket server with LLM integration...")
    print("="*60 + "\n")

    # Import and run the debate server
    sys.path.insert(0, str(Path(__file__).parent))
    from visualization.debate_server import CognitiveDebateOrchestrator

    orchestrator = CognitiveDebateOrchestrator()

    # Start server in background
    server_task = asyncio.create_task(orchestrator.run_server())

    # Wait a moment for server to start
    await asyncio.sleep(2)

    # Open the HTML file
    html_path = Path(__file__).parent / "visualization" / "multi_agent_debate.html"
    url = f"file:///{html_path.absolute()}"
    print(f"Opening visualization at: {url}")
    
    try:
        if sys.platform == 'win32':
            os.startfile(html_path)
        else:
            webbrowser.open(url)
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print("Please copy/paste the URL above into your browser.")

    print("\nVisualization opened in browser!")
    print("WebSocket server running at ws://localhost:8765")
    print("\nFeatures available in full mode:")
    print("- Real LLM-powered agent responses")
    print("- Live token streaming")
    print("- Adaptive rigidity based on actual prediction errors")
    print("- Personality-modulated LLM parameters")
    print("- Full metacognitive alerts")

    print("\nPress Ctrl+C to stop the server")

    try:
        await server_task
    except KeyboardInterrupt:
        print("\nStopping server...")

def main():
    """Main launcher logic."""
    print("="*60)
    print("DDA-X INTERACTIVE VISUALIZATION LAUNCHER")
    print("Dynamic Decision Algorithm with Exploration")
    print("="*60)
    print("\nChecking LLM backends...")

    backends = check_llm_backends()

    if not any(backends.values()):
        print("\n⚠️  No LLM backends detected")
        print("Launching in DEMO MODE with simulated agents")
        print("\nTo enable full mode with real LLMs:")
        print("1. Start LM Studio on port 1234")
        print("2. Run 'ollama serve' for embeddings")
        launch_demo_mode()
    else:
        print("\n✓ LLM backends available!")

        # Check for required models
        if backends["ollama"]:
            print("\nChecking for embedding model...")
            try:
                import ollama
                client = ollama.Client(host="http://localhost:11434")
                models = client.list()
                has_embed = any("nomic-embed-text" in m['name'] for m in models.get('models', []))

                if not has_embed:
                    print("⚠️  nomic-embed-text not found")
                    print("Installing embedding model (this may take a moment)...")
                    subprocess.run(["ollama", "pull", "nomic-embed-text"], check=False)
            except Exception as e:
                print(f"Warning: Could not check Ollama models: {e}")

        print("\nLaunching full system with WebSocket server...")
        asyncio.run(launch_full_mode())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nFalling back to demo mode...")
        launch_demo_mode()