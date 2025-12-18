"""
Run the REAL DDA-X WebSocket Server
This demonstrates your actual architecture with live streaming.
"""

import asyncio
import webbrowser
from pathlib import Path

async def main():
    print("="*70)
    print("LAUNCHING DDA-X REAL ARCHITECTURE")
    print("="*70)
    print("\nThis will run your EXACT debate server with:")
    print("- HybridProvider (LM Studio + Ollama)")
    print("- Rigidity dynamics")
    print("- Trust matrix")
    print("- WebSocket streaming")
    print("- Live token generation")
    print("\n" + "="*70)

    # Import your actual debate server
    from visualization.debate_server import CognitiveDebateOrchestrator

    # Create orchestrator with your real architecture
    orchestrator = CognitiveDebateOrchestrator()

    print("\n[Starting WebSocket server on ws://localhost:8765]")
    print("[Opening visualization in browser...]")

    # Open the HTML
    html_path = Path(__file__).parent / "visualization" / "multi_agent_debate.html"
    webbrowser.open(f"file:///{html_path.absolute()}")

    print("\n" + "="*70)
    print("SERVER RUNNING - Your EXACT architecture is now live!")
    print("="*70)
    print("\nIn the browser:")
    print("1. Click 'Start Debate' to see REAL LLM responses")
    print("2. Watch the rigidity bars change based on actual prediction errors")
    print("3. See trust evolve from real agent interactions")
    print("4. Click 'Inject Surprise' to trigger your rigidity dynamics")
    print("\nThe WebSocket connection shows:")
    print("- Real token streaming from your LLM backends")
    print("- Personality-modulated temperature (rigidity -> LLM params)")
    print("- Your exact DDA-X equations in action")
    print("\nPress Ctrl+C to stop")
    print("="*70 + "\n")

    # Run the actual server
    try:
        await orchestrator.run_server()
    except KeyboardInterrupt:
        print("\n[Server stopped]")

if __name__ == "__main__":
    asyncio.run(main())