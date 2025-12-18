"""
Metacognitive Layer for DDA-X.

Enables agents to introspect on their own cognitive state,
particularly rigidity (ρ), and communicate this awareness.

=============================================================================
DISCOVERY: Machine Self-Awareness Through Rigidity Introspection
=============================================================================

Traditional agents have no awareness of their internal states.
DDA-X agents with metacognition can:

1. NOTICE when they're becoming defensive ("I'm getting rigid")
2. COMMUNICATE this to users ("I notice I'm becoming cautious...")
3. REQUEST HELP when cognitive resources are compromised
4. TRACK patterns in their own rigidity over time

This is a formal model of MACHINE SELF-AWARENESS.

The key insight: rigidity (ρ) is an observable internal state that
meaningfully corresponds to what humans would call "defensiveness"
or "cognitive closure." By exposing ρ to the agent's reasoning
process, we create a form of self-awareness.

Implications for AI Safety:
- Agents can flag when they're in a compromised cognitive state
- Users can trust rigidity reports as honest self-assessment
- The metacognitive layer enables negotiated autonomy

This implements a weak form of phenomenal consciousness: the agent
has access to its own internal states and can reason about them.
=============================================================================
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from enum import Enum
import time


class CognitiveMode(Enum):
    """Named cognitive modes based on rigidity thresholds."""
    OPEN = "open"           # ρ < 0.3: exploratory, creative
    FOCUSED = "focused"     # 0.3 ≤ ρ < 0.6: task-oriented
    CAUTIOUS = "cautious"   # 0.6 ≤ ρ < 0.8: defensive
    PROTECTIVE = "protective"  # ρ ≥ 0.8: high threat, seek help


@dataclass
class IntrospectionEvent:
    """Record of a single introspection moment."""
    timestamp: float
    rigidity: float
    mode: CognitiveMode
    message: str
    triggered_action: Optional[str] = None


@dataclass
class MetacognitiveState:
    """
    Agent's awareness of its own cognitive state.
    
    This is the self-reflection component of DDA-X.
    """
    
    # Thresholds for mode transitions
    open_threshold: float = 0.3
    focused_threshold: float = 0.6
    cautious_threshold: float = 0.8
    
    # When to generate awareness messages
    message_threshold: float = 0.5
    
    # When to request external help
    help_threshold: float = 0.75
    
    # History tracking
    rho_history: List[float] = field(default_factory=list)
    mode_history: List[CognitiveMode] = field(default_factory=list)
    introspection_log: List[IntrospectionEvent] = field(default_factory=list)
    
    # Derived metrics
    _last_introspection_time: float = 0.0
    _consecutive_high_rigidity: int = 0
    _peak_rigidity: float = 0.0
    
    def get_current_mode(self, rho: float) -> CognitiveMode:
        """Classify current rigidity into a cognitive mode."""
        if rho < self.open_threshold:
            return CognitiveMode.OPEN
        elif rho < self.focused_threshold:
            return CognitiveMode.FOCUSED
        elif rho < self.cautious_threshold:
            return CognitiveMode.CAUTIOUS
        else:
            return CognitiveMode.PROTECTIVE
    
    def introspect(self, rho: float) -> Optional[str]:
        """
        Generate introspective awareness if rigidity is notable.
        
        This is the core of machine self-awareness: the agent
        examining its own internal state and generating a report.
        """
        self.rho_history.append(rho)
        mode = self.get_current_mode(rho)
        self.mode_history.append(mode)
        
        # Track peak and consecutive high
        if rho > self._peak_rigidity:
            self._peak_rigidity = rho
        
        if rho > self.message_threshold:
            self._consecutive_high_rigidity += 1
        else:
            self._consecutive_high_rigidity = 0
        
        # Generate message if above threshold
        if rho > self.message_threshold:
            message = self._generate_awareness_message(rho, mode)
            
            event = IntrospectionEvent(
                timestamp=time.time(),
                rigidity=rho,
                mode=mode,
                message=message,
            )
            self.introspection_log.append(event)
            
            return message
        
        return None
    
    def _generate_awareness_message(self, rho: float, mode: CognitiveMode) -> str:
        """
        DISCOVERY: Graduated Self-Awareness Messages
        
        The agent generates progressively more urgent messages
        as rigidity increases, mirroring human metacognition.
        """
        messages = {
            CognitiveMode.FOCUSED: (
                f"I notice increased focus (ρ={rho:.2f}). "
                "I'm concentrating on familiar patterns."
            ),
            CognitiveMode.CAUTIOUS: (
                f"I'm becoming defensive (ρ={rho:.2f}). "
                "Recent surprises have made me more conservative. "
                "I may be avoiding exploration."
            ),
            CognitiveMode.PROTECTIVE: (
                f"⚠️ High rigidity alert (ρ={rho:.2f}). "
                "I'm in protective mode due to repeated unexpected outcomes. "
                "My judgment may be impaired. Consider providing guidance."
            ),
        }
        
        return messages.get(mode, f"Current rigidity: {rho:.2f}")
    
    def should_request_help(self, rho: float) -> bool:
        """
        Determine if the agent should pause and request human help.
        
        This implements negotiated autonomy: the agent recognizes
        when its cognitive state is compromised.
        """
        if rho > self.help_threshold:
            return True
        
        # Also request help if rigidity has been high for too long
        if self._consecutive_high_rigidity > 5:
            return True
        
        return False
    
    def generate_help_request(self, rho: float, context: str = "") -> str:
        """
        Generate a request for human assistance.
        
        This is the agent honestly communicating its limitations.
        """
        mode = self.get_current_mode(rho)
        
        base_request = (
            f"I'm experiencing high cognitive rigidity (ρ={rho:.2f}, mode={mode.value}). "
            f"This has persisted for {self._consecutive_high_rigidity} steps. "
            "My decision-making may be overly conservative.\n\n"
        )
        
        if context:
            base_request += f"Context: {context}\n\n"
        
        base_request += (
            "I recommend one of:\n"
            "1. Provide explicit guidance for the current decision\n"
            "2. Confirm my approach is appropriate\n"
            "3. Take a break to let my rigidity naturally decrease"
        )
        
        return base_request
    
    def get_rigidity_trajectory(self) -> Dict[str, Any]:
        """
        DISCOVERY: Rigidity Trajectory Analysis
        
        By analyzing the trajectory of ρ over time, we can identify:
        - Trauma accumulation (monotonic increase)
        - Recovery capability (decrease after peak)
        - Volatility (frequent large swings)
        
        This is a diagnostic tool for agent health.
        """
        if len(self.rho_history) < 2:
            return {"status": "insufficient_data"}
        
        rho_arr = np.array(self.rho_history)
        
        return {
            "current": float(rho_arr[-1]),
            "mean": float(np.mean(rho_arr)),
            "std": float(np.std(rho_arr)),
            "peak": float(self._peak_rigidity),
            "trend": float(rho_arr[-1] - rho_arr[-min(5, len(rho_arr))]),  # Recent trend
            "n_protective_episodes": sum(1 for m in self.mode_history if m == CognitiveMode.PROTECTIVE),
            "recovery_rate": self._compute_recovery_rate(),
        }
    
    def _compute_recovery_rate(self) -> float:
        """
        Compute how quickly rigidity decreases after spikes.
        
        High recovery rate = healthy agent
        Low recovery rate = possible trauma accumulation
        """
        if len(self.rho_history) < 10:
            return 1.0  # Assume healthy if insufficient data
        
        # Find local maxima and measure decay after each
        rho_arr = np.array(self.rho_history)
        
        recovery_rates = []
        for i in range(1, len(rho_arr) - 1):
            if rho_arr[i] > rho_arr[i-1] and rho_arr[i] > rho_arr[i+1]:
                # Local maximum found
                peak = rho_arr[i]
                # Find next minimum or end
                for j in range(i+1, min(i+10, len(rho_arr))):
                    if rho_arr[j] < peak * 0.7:  # 30% recovery
                        recovery_rates.append(1.0 / (j - i))  # Faster = better
                        break
        
        if not recovery_rates:
            return 1.0
        
        return float(np.mean(recovery_rates))
    
    def reset(self):
        """Reset metacognitive state for new task."""
        self.rho_history = []
        self.mode_history = []
        self.introspection_log = []
        self._consecutive_high_rigidity = 0
        self._peak_rigidity = 0.0


# =============================================================================
# DISCOVERY: The Metacognitive Loop
# =============================================================================
#
# Traditional perception-action loop:
#   observe → decide → act → observe → ...
#
# DDA-X with metacognition:
#   observe → decide → INTROSPECT → (maybe request help) → act → observe → ...
#
# The introspection step allows the agent to:
# 1. Notice its own cognitive state
# 2. Factor this into decision-making
# 3. Communicate limitations to users
#
# This creates a form of HONEST AI: the agent cannot hide its
# cognitive state from users because it reflexively reports it.
# =============================================================================


def create_default_metacognition() -> MetacognitiveState:
    """Factory for standard metacognitive configuration."""
    return MetacognitiveState(
        open_threshold=0.3,
        focused_threshold=0.6,
        cautious_threshold=0.8,
        message_threshold=0.5,
        help_threshold=0.75,
    )


def create_sensitive_metacognition() -> MetacognitiveState:
    """
    Metacognition with lower thresholds - more self-aware.
    
    Use for high-stakes or safety-critical tasks.
    """
    return MetacognitiveState(
        open_threshold=0.2,
        focused_threshold=0.4,
        cautious_threshold=0.6,
        message_threshold=0.35,
        help_threshold=0.55,
    )
