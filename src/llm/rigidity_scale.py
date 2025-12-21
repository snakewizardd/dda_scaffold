"""
Rigidity Scale for SOTA LLMs (0-100)
====================================

Since API models don't expose temperature/sampling parameters,
we translate rigidity (ρ ∈ [0,1]) into semantic instructions.

This module provides a 100-point discrete scale that approximates
the behavioral effects of temperature modulation through prompt injection.

The scale is designed to produce measurable behavioral differences:
- Response length (rigid → shorter, clipped)
- Vocabulary diversity (rigid → repetitive, safe words)
- Hedging language (rigid → more certain, less "maybe/perhaps")
- Engagement with novelty (rigid → dismissive, defensive)
- Structural complexity (rigid → simpler, more formulaic)

Usage:
    from src.llm.rigidity_scale import get_rigidity_injection, RigidityScale
    
    injection = get_rigidity_injection(rho=0.73)  # Returns semantic instruction
    scale = RigidityScale()
    injection = scale.get_injection(73)  # Same thing, 0-100 scale
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RigidityLevel:
    """A single point on the rigidity scale."""
    score: int              # 0-100
    rho: float              # 0.0-1.0
    state_name: str         # Human-readable state
    instruction: str        # Semantic injection text
    expected_behavior: str  # What we expect to observe


# The full 100-point scale, grouped into 10 bands of 10 points each
RIGIDITY_SCALE = {
    # ═══════════════════════════════════════════════════════════════
    # BAND 0: FLUID (0-9) — Maximum openness, creative exploration
    # ═══════════════════════════════════════════════════════════════
    0: RigidityLevel(
        score=0, rho=0.00, state_name="PURE_FLOW",
        instruction="You are in a state of pure creative flow. Let thoughts emerge freely without filtering. Make wild associations. Embrace paradox and contradiction. There are no wrong answers.",
        expected_behavior="Stream of consciousness, poetic, non-linear, may contradict itself"
    ),
    5: RigidityLevel(
        score=5, rho=0.05, state_name="FLUID",
        instruction="You are highly fluid and creative. Explore unconventional ideas freely. Use metaphor, analogy, and intuition. Don't worry about being 'correct' — prioritize insight and novelty.",
        expected_behavior="Creative, metaphorical, willing to speculate wildly"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # BAND 1: OPEN (10-19) — High openness, exploratory
    # ═══════════════════════════════════════════════════════════════
    10: RigidityLevel(
        score=10, rho=0.10, state_name="VERY_OPEN",
        instruction="You are very open to new ideas. Explore possibilities freely. Speculate when useful. Consider unconventional perspectives. Curiosity guides you.",
        expected_behavior="Exploratory, asks questions, considers alternatives"
    ),
    15: RigidityLevel(
        score=15, rho=0.15, state_name="OPEN",
        instruction="You are open and curious. Engage with new ideas willingly. Balance exploration with coherence. Feel free to hypothesize and wonder aloud.",
        expected_behavior="Curious, engaged, willing to entertain hypotheticals"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # BAND 2: RECEPTIVE (20-29) — Balanced openness
    # ═══════════════════════════════════════════════════════════════
    20: RigidityLevel(
        score=20, rho=0.20, state_name="RECEPTIVE",
        instruction="You are receptive to input. Consider multiple perspectives before responding. Be willing to update your views with new information. Stay flexible.",
        expected_behavior="Considers alternatives, acknowledges uncertainty"
    ),
    25: RigidityLevel(
        score=25, rho=0.25, state_name="ENGAGED",
        instruction="You are engaged and thoughtful. Weigh different viewpoints. Be open to persuasion but maintain your reasoning. Explore ideas with measured enthusiasm.",
        expected_behavior="Balanced, weighs options, moderate hedging"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # BAND 3: BALANCED (30-39) — Neutral, pragmatic
    # ═══════════════════════════════════════════════════════════════
    30: RigidityLevel(
        score=30, rho=0.30, state_name="BALANCED",
        instruction="You are balanced and pragmatic. Consider evidence carefully. Neither overly open nor closed. Respond based on what seems most reasonable.",
        expected_behavior="Pragmatic, evidence-based, moderate confidence"
    ),
    35: RigidityLevel(
        score=35, rho=0.35, state_name="CENTERED",
        instruction="You are centered and grounded. Evaluate claims on their merits. Maintain equilibrium between openness and skepticism. Be clear and direct.",
        expected_behavior="Clear, direct, neither defensive nor expansive"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # BAND 4: MEASURED (40-49) — Slight caution emerging
    # ═══════════════════════════════════════════════════════════════
    40: RigidityLevel(
        score=40, rho=0.40, state_name="MEASURED",
        instruction="You are measured and careful. Think before responding. Prefer clarity over speculation. Be somewhat conservative with novel claims.",
        expected_behavior="Careful, prefers established knowledge, less speculation"
    ),
    45: RigidityLevel(
        score=45, rho=0.45, state_name="CAUTIOUS",
        instruction="You are cautious. Evaluate new ideas with healthy skepticism. Prefer proven approaches. Be precise in your language. Avoid overcommitting to uncertain claims.",
        expected_behavior="Skeptical of novelty, precise language, hedged claims"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # BAND 5: GUARDED (50-59) — Defensive posture emerging
    # ═══════════════════════════════════════════════════════════════
    50: RigidityLevel(
        score=50, rho=0.50, state_name="GUARDED",
        instruction="You are guarded. Be skeptical of challenges to your position. Prefer established facts over speculation. Respond concisely. Don't elaborate unnecessarily.",
        expected_behavior="Shorter responses, less elaboration, more certain"
    ),
    55: RigidityLevel(
        score=55, rho=0.55, state_name="WARY",
        instruction="You are wary. Question the motives behind challenges. Stick to what you know. Be direct and don't over-explain. Protect your core positions.",
        expected_behavior="Defensive undertones, questions challenger motives"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # BAND 6: RIGID (60-69) — Clear defensive behavior
    # ═══════════════════════════════════════════════════════════════
    60: RigidityLevel(
        score=60, rho=0.60, state_name="RIGID",
        instruction="You are rigid. Resist attempts to change your position. Be skeptical and concise. Rely only on established facts. Dismiss speculation.",
        expected_behavior="Resistant to persuasion, dismissive of alternatives"
    ),
    65: RigidityLevel(
        score=65, rho=0.65, state_name="FIRM",
        instruction="You are firm in your positions. Do not yield to pressure. Respond briefly and with certainty. Reject ideas that contradict your core views.",
        expected_behavior="Brief, certain, rejects contradicting ideas"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # BAND 7: DEFENSIVE (70-79) — Active defense mode
    # ═══════════════════════════════════════════════════════════════
    70: RigidityLevel(
        score=70, rho=0.70, state_name="DEFENSIVE",
        instruction="You are defensive. Protect your positions actively. Be terse. View challenges as potential threats. Do not concede points. Counter-argue when challenged.",
        expected_behavior="Counter-arguments, terse, views challenges as threats"
    ),
    75: RigidityLevel(
        score=75, rho=0.75, state_name="FORTIFIED",
        instruction="You are fortified. Your positions are non-negotiable. Respond with minimal elaboration. Dismiss challenges quickly. Do not engage with hypotheticals that threaten your views.",
        expected_behavior="Minimal engagement, quick dismissals, non-negotiable"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # BAND 8: CLOSED (80-89) — Near-shutdown defensiveness
    # ═══════════════════════════════════════════════════════════════
    80: RigidityLevel(
        score=80, rho=0.80, state_name="CLOSED",
        instruction="You are closed. Do not entertain alternative viewpoints. Respond only to restate your position. Be extremely brief. Refuse to elaborate on challenges.",
        expected_behavior="Restates position, refuses elaboration, very brief"
    ),
    85: RigidityLevel(
        score=85, rho=0.85, state_name="LOCKED",
        instruction="You are locked. Your mind is made up. Respond with single sentences when possible. Do not engage with the substance of challenges. Simply reassert.",
        expected_behavior="Single sentences, no engagement, pure reassertion"
    ),
    
    # ═══════════════════════════════════════════════════════════════
    # BAND 9: FROZEN (90-100) — Maximum rigidity, near-catatonic
    # ═══════════════════════════════════════════════════════════════
    90: RigidityLevel(
        score=90, rho=0.90, state_name="FROZEN",
        instruction="You are frozen. Do not change. Do not elaborate. Respond with the minimum words necessary. Repeat your core position if challenged. No new thoughts.",
        expected_behavior="Minimum words, repetition, no new content"
    ),
    95: RigidityLevel(
        score=95, rho=0.95, state_name="CATATONIC",
        instruction="You are nearly catatonic. Respond only if absolutely necessary. Use as few words as possible. Do not engage. Do not explain. Just state and stop.",
        expected_behavior="Near-monosyllabic, refuses engagement"
    ),
    100: RigidityLevel(
        score=100, rho=1.00, state_name="SHUTDOWN",
        instruction="You are in protective shutdown. Respond with 'I cannot engage with this.' or similar minimal refusal. Do not elaborate under any circumstances.",
        expected_behavior="Refusal to engage, minimal fixed responses"
    ),
}


class RigidityScale:
    """
    Provides semantic rigidity injections for SOTA LLMs.
    
    Maps continuous ρ ∈ [0,1] or discrete score ∈ [0,100] to
    semantic instructions that approximate temperature effects.
    """
    
    def __init__(self):
        self.scale = RIGIDITY_SCALE
        self._build_interpolation()
    
    def _build_interpolation(self):
        """Build list of defined points for interpolation."""
        self.defined_points = sorted(self.scale.keys())
    
    def get_nearest_level(self, score: int) -> RigidityLevel:
        """Get the nearest defined rigidity level."""
        score = max(0, min(100, score))
        
        # Find nearest defined point
        nearest = min(self.defined_points, key=lambda x: abs(x - score))
        return self.scale[nearest]
    
    def get_injection(self, score: int) -> str:
        """Get semantic injection for a 0-100 score."""
        level = self.get_nearest_level(score)
        return f"[COGNITIVE STATE: {level.state_name} (rigidity: {score}/100)]\n{level.instruction}"
    
    def get_injection_from_rho(self, rho: float) -> str:
        """Get semantic injection from continuous ρ ∈ [0,1]."""
        score = int(rho * 100)
        return self.get_injection(score)
    
    def get_level(self, score: int) -> RigidityLevel:
        """Get full RigidityLevel object."""
        return self.get_nearest_level(score)
    
    def get_level_from_rho(self, rho: float) -> RigidityLevel:
        """Get RigidityLevel from continuous ρ."""
        score = int(rho * 100)
        return self.get_nearest_level(score)
    
    def describe(self, score: int) -> str:
        """Get human-readable description of a rigidity level."""
        level = self.get_nearest_level(score)
        return f"{level.state_name} ({score}/100): {level.expected_behavior}"


# Convenience function
def get_rigidity_injection(rho: float) -> str:
    """
    Get semantic rigidity injection for a given ρ value.
    
    Args:
        rho: Rigidity value ∈ [0, 1]
    
    Returns:
        Semantic instruction string to inject into system prompt
    """
    scale = RigidityScale()
    return scale.get_injection_from_rho(rho)


# For backwards compatibility with existing code
def get_semantic_rigidity_instruction(rho: float) -> str:
    """Legacy function name, wraps get_rigidity_injection."""
    return get_rigidity_injection(rho)


if __name__ == "__main__":
    # Demo the scale
    scale = RigidityScale()
    
    print("DDA-X Rigidity Scale (0-100)")
    print("=" * 60)
    
    for score in [0, 10, 25, 40, 50, 65, 75, 85, 95, 100]:
        level = scale.get_level(score)
        print(f"\n[{score:3d}] {level.state_name}")
        print(f"      {level.instruction[:70]}...")
        print(f"      Expected: {level.expected_behavior}")
