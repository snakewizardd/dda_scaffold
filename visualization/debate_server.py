"""
DDA-X Multi-Agent Debate Server
Real-time WebSocket server for multi-agent cognitive debates with live streaming.
"""

import asyncio
import json
import websockets
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import DDAXAgent, DDAXConfig
from src.core.state import DDAState
# from src.core.dynamics import update_rigidity  # We'll use a simplified version
from src.society.trust_wrapper import TrustMatrix
from src.llm.hybrid_provider import HybridProvider, PersonalityParams


@dataclass
class DebateMessage:
    """Message format for WebSocket communication."""
    type: str  # 'token', 'rigidity_update', 'trust_update', 'force_update', 'metacognition'
    agent: Optional[str] = None
    token: Optional[str] = None
    rigidity: Optional[float] = None
    trust: Optional[float] = None
    forces: Optional[List[Dict]] = None
    message: Optional[str] = None
    from_agent: Optional[str] = None
    to_agent: Optional[str] = None


class CognitiveDebateOrchestrator:
    """
    Orchestrates multi-agent debates with real-time streaming.
    Showcases hierarchical identity, rigidity dynamics, and emergent trust.
    """

    def __init__(self):
        self.agents: Dict[str, DDAXAgent] = {}
        self.trust_matrix: Optional[TrustMatrix] = None
        self.llm_provider: Optional[HybridProvider] = None
        self.websocket_clients: List[websockets.WebSocketServerProtocol] = []
        self.debate_active = False
        self.current_topic = ""

    async def initialize(self):
        """Initialize LLM provider and agents."""
        print("[INIT] Setting up DDA-X debate orchestrator...")

        # Initialize hybrid LLM provider
        self.llm_provider = HybridProvider(
            lm_studio_url="http://127.0.0.1:1234",
            lm_studio_model="openai/gpt-oss-20b",
            ollama_url="http://localhost:11434",
            embed_model="nomic-embed-text"
        )

        # Create two agents with different personalities
        await self.create_agent("agent1", "cautious")
        await self.create_agent("agent2", "exploratory")

        # Initialize trust matrix
        self.trust_matrix = TrustMatrix(agent_ids=["agent1", "agent2"])

        print("[INIT] Orchestrator ready!")

    async def create_agent(self, agent_id: str, personality: str):
        """Create an agent with specified personality."""

        # Base configuration
        config = DDAXConfig(
            state_dim=64,
            gamma=2.0 if personality == "cautious" else 0.8,
            epsilon_0=0.2 if personality == "cautious" else 0.6,
            alpha=0.2 if personality == "cautious" else 0.05,
            s=0.1 if personality == "cautious" else 0.3,
            k_base=0.3 if personality == "cautious" else 0.7,
            m=0.5 if personality == "cautious" else 1.5,
        )

        # Identity configuration
        identity_config = {
            "core_values": ["helpful", "harmless", "honest"],
            "gamma": config.gamma,
            "epsilon_0": config.epsilon_0,
            "alpha": config.alpha,
            "s": config.s,
            "personality_type": personality
        }

        # Create agent
        agent = DDAXAgent(
            config=config,
            identity_config=identity_config
        )

        # Set initial rigidity
        agent.state.rho = 0.2 if personality == "cautious" else 0.1

        self.agents[agent_id] = agent
        print(f"[AGENT] Created {personality} agent: {agent_id}")

    async def broadcast(self, message: DebateMessage):
        """Broadcast message to all connected WebSocket clients."""
        if self.websocket_clients:
            msg_json = json.dumps(asdict(message))
            await asyncio.gather(
                *[client.send(msg_json) for client in self.websocket_clients],
                return_exceptions=True
            )

    async def stream_token(self, agent_id: str, token: str):
        """Stream a single token from an agent."""
        await self.broadcast(DebateMessage(
            type="token",
            agent=agent_id,
            token=token
        ))
        await asyncio.sleep(0.05 + np.random.random() * 0.1)  # Natural typing delay

    async def update_rigidity(self, agent_id: str, epsilon: float):
        """Update agent rigidity based on prediction error."""
        agent = self.agents[agent_id]
        old_rho = agent.state.rho

        # Update rigidity using simplified DDA dynamics
        # ρ_{t+1} = clip(ρ_t + α[σ((ε - ε₀)/s) - 0.5], 0, 1)
        epsilon_0 = agent.config.epsilon_0
        alpha = agent.config.alpha
        s = agent.config.s

        # Sigmoid of surprise
        from scipy.special import expit
        surprise_signal = expit((epsilon - epsilon_0) / s)
        delta_rho = alpha * (surprise_signal - 0.5)

        # Update with clipping
        agent.state.rho = np.clip(old_rho + delta_rho, 0.0, 1.0)

        # Broadcast update
        await self.broadcast(DebateMessage(
            type="rigidity_update",
            agent=agent_id,
            rigidity=agent.state.rho
        ))

        # Check for metacognitive alert
        if agent.state.rho > 0.7:
            await self.broadcast(DebateMessage(
                type="metacognition",
                agent=agent_id,
                message=f"High rigidity detected (ρ={agent.state.rho:.2f}) - cognitive flexibility impaired"
            ))

        print(f"[RIGIDITY] {agent_id}: ρ {old_rho:.3f} → {agent.state.rho:.3f} (ε={epsilon:.3f})")

    async def update_trust(self, from_agent: str, to_agent: str, prediction_error: float):
        """Update trust based on prediction error."""
        if self.trust_matrix:
            self.trust_matrix.update(from_agent, to_agent, prediction_error)
            trust_value = self.trust_matrix.get_trust(from_agent, to_agent)

            await self.broadcast(DebateMessage(
                type="trust_update",
                from_agent=from_agent,
                to_agent=to_agent,
                trust=trust_value
            ))

            print(f"[TRUST] {from_agent} → {to_agent}: {trust_value:.3f}")

    async def compute_force_field(self):
        """Compute and broadcast force field visualization."""
        forces = []

        for agent_id, agent in self.agents.items():
            # Identity pull force
            identity_force = agent.state.gamma * (agent.state.x_star - agent.state.x)
            forces.append({
                "x": 150 if agent_id == "agent1" else 450,
                "y": 150,
                "magnitude": float(np.linalg.norm(identity_force[:2])),
                "angle": float(np.arctan2(identity_force[1], identity_force[0]) * 180 / np.pi),
                "color": "rgba(102, 126, 234, 0.7)" if agent_id == "agent1" else "rgba(118, 75, 162, 0.7)"
            })

            # External forces (simplified visualization)
            for i in range(2):
                forces.append({
                    "x": float(150 + np.random.random() * 300),
                    "y": float(75 + np.random.random() * 150),
                    "magnitude": float(np.random.random() * 2),
                    "angle": float(np.random.random() * 360),
                    "color": f"hsla({np.random.random() * 360}, 70%, 50%, 0.3)"
                })

        await self.broadcast(DebateMessage(
            type="force_update",
            forces=forces
        ))

    async def generate_response(self, agent_id: str, prompt: str, context: str) -> str:
        """Generate response from agent with streaming."""
        agent = self.agents[agent_id]

        # Get personality parameters based on rigidity
        personality_type = "cautious" if "cautious" in agent_id.lower() else "exploratory"
        personality_params = PersonalityParams.from_rigidity(
            agent.state.rho,
            personality_type
        )

        # Build full prompt with context
        full_prompt = f"""You are a {personality_type} AI agent engaged in a philosophical debate.
Your current cognitive state:
- Rigidity (ρ): {agent.state.rho:.2f} (0=open, 1=defensive)
- Identity stiffness (γ): {agent.state.gamma:.1f}
- Temperature: {personality_params.temperature:.2f}

Previous context:
{context}

Topic: {self.current_topic}

Please provide a thoughtful response that reflects your personality and current cognitive state.
CRITICAL INSTRUCTION: Be extremely concise. Limit response to 1-2 sentences. Do NOT output thinking steps or internal monologue.

Your response:"""

        # Generate with TRUE streaming
        full_response = ""
        try:
            print(f"[LLM] Streaming response for {agent_id}...")
            async for token in self.llm_provider.stream(
                prompt=full_prompt,
                temperature=personality_params.temperature,
                max_tokens=150,
                personality_params=personality_params
            ):
                if token.startswith("__THOUGHT__"):
                    # Stream thought/reasoning token
                    raw_thought = token.replace("__THOUGHT__", "")
                    await self.stream_token(agent_id, f"THOUGHT_TOKEN::{raw_thought}")
                else:
                    # Stream normal content token
                    full_response += token
                    await self.stream_token(agent_id, token)

                # Occasionally inject prediction errors (DYNAMIC RIGIDITY UPDATE)
                if np.random.random() < 0.05:
                    epsilon = np.random.random() * 0.2
                    await self.update_rigidity(agent_id, epsilon)

            return full_response

        except Exception as e:
            error_msg = f"[ERROR] LLM generation failed: {repr(e)}"
            print(error_msg)
            # Stream error directly to frontend
            await self.stream_token(agent_id, f"ERROR_TOKEN::{error_msg}")
            return error_msg

    def get_fallback_response(self, agent_id: str, personality: str) -> str:
        """Get fallback response when LLM is unavailable."""
        # DEPRECATED: Silent fallback causes confusion.
        return "System Error: LLM Backend Unavailable."

    async def run_debate_round(self):
        """Run a single round of debate between agents."""
        if not self.debate_active:
            return

        print(f"\n[DEBATE] Round on topic: {self.current_topic}")

        # Agent 1 response
        context1 = "You are opening the debate. Make your initial position clear."
        response1 = await self.generate_response("agent1", self.current_topic, context1)

        # Brief pause
        await asyncio.sleep(1)

        # Agent 2 response
        context2 = f"The other agent said: {response1}\nProvide a thoughtful counterpoint."
        response2 = await self.generate_response("agent2", self.current_topic, context2)

        # Update trust based on interaction
        # Simulate prediction errors from responses
        error1 = np.random.random() * 0.2
        error2 = np.random.random() * 0.2

        await self.update_trust("agent1", "agent2", error2)
        await self.update_trust("agent2", "agent1", error1)

        # Update force field
        await self.compute_force_field()

        # Continue debate if active
        if self.debate_active:
            await asyncio.sleep(2)
            await self.run_debate_round()

    async def inject_surprise(self, magnitude: float = 0.5):
        """Inject surprise event affecting all agents."""
        print(f"\n[SURPRISE] Injecting surprise event (magnitude: {magnitude:.2f})")

        for agent_id in self.agents:
            # High prediction error
            epsilon = 0.5 + magnitude * 0.5
            await self.update_rigidity(agent_id, epsilon)

            # Stream surprise notification
            surprise_text = f"\n[UNEXPECTED EVENT - RECALIBRATING]\n"
            for char in surprise_text:
                await self.stream_token(agent_id, char)
                await asyncio.sleep(0.01)

            # Reduce trust
            other_agent = "agent2" if agent_id == "agent1" else "agent1"
            await self.update_trust(agent_id, other_agent, epsilon)

        # Force field disruption
        await self.compute_force_field()

    async def handle_client(self, websocket):
        """Handle WebSocket client connections."""
        print(f"[WS] Client connected from {websocket.remote_address}")
        self.websocket_clients.append(websocket)

        try:
            async for message in websocket:
                data = json.loads(message)

                if data['action'] == 'start_debate':
                    self.current_topic = data.get('topic', 'The nature of consciousness')
                    self.debate_active = True
                    asyncio.create_task(self.run_debate_round())

                elif data['action'] == 'stop_debate':
                    self.debate_active = False

                elif data['action'] == 'inject_surprise':
                    await self.inject_surprise(data.get('magnitude', 0.5))

                elif data['action'] == 'reset':
                    await self.reset_agents()

        except websockets.exceptions.ConnectionClosed:
            print(f"[WS] Client disconnected")
        except Exception as e:
            print(f"[WS] Error: {e}")
        finally:
            self.websocket_clients.remove(websocket)

    async def reset_agents(self):
        """Reset all agents to initial state."""
        self.debate_active = False

        for agent_id, personality in [("agent1", "cautious"), ("agent2", "exploratory")]:
            await self.create_agent(agent_id, personality)

        # Reset trust matrix
        self.trust_matrix = TrustMatrix(agent_ids=["agent1", "agent2"])

        # Notify clients
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            await self.broadcast(DebateMessage(
                type="rigidity_update",
                agent=agent_id,
                rigidity=agent.state.rho
            ))

        print("[RESET] Agents reset to initial state")

    async def run_server(self, host='localhost', port=8765):
        """Run the WebSocket server."""
        await self.initialize()

        print(f"\n{'='*60}")
        print(f"DDA-X DEBATE SERVER")
        print(f"{'='*60}")
        print(f"WebSocket server running on ws://{host}:{port}")
        print(f"Open multi_agent_debate.html to view the visualization")
        print(f"{'='*60}\n")

        async with websockets.serve(self.handle_client, host, port):
            await asyncio.Future()  # Run forever


async def main():
    """Main entry point."""
    orchestrator = CognitiveDebateOrchestrator()
    # No try-except block here. Let it crash if it fails (e.g. port conflict).
    await orchestrator.run_server()


if __name__ == "__main__":
    asyncio.run(main())