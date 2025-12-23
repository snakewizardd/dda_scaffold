"""
Dynamics Module for DDA-X

Handles state evolution and rigidity dynamics.

ENHANCED: Multi-timescale rigidity for modeling:
- Immediate reactions (fast)
- Long-term adaptation (slow)
- Trauma accumulation (permanent)

=============================================================================
DISCOVERY: Multi-Timescale Rigidity Dynamics
=============================================================================

Human defensiveness operates on multiple timescales:
1. STARTLE: Immediate, reflexive (subseconds)
2. STRESS: Short-term adaptation (minutes to hours)
3. HABITUATION: Long-term learning (days to weeks)
4. TRAUMA: Permanent changes from extreme events

DDA-X models this with three coupled rigidity variables:
- ρ_fast: α = 0.3, decays quickly
- ρ_slow: α = 0.01, decays slowly
- ρ_trauma: α = 0.0001, ASYMMETRIC (only increases)

The trauma component is the key insight: some experiences
permanently alter the agent's baseline defensiveness.

This creates emergent phenomena:
- PTSD: trauma accumulation from repeated high-surprise events
- RESILIENCE: low trauma despite high fast/slow activation
- SENSITIZATION: early trauma amplifies future responses
=============================================================================
"""

import numpy as np
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

# Configure module-level logger
logger = logging.getLogger(__name__)


@dataclass
class MultiTimescaleRigidity:
    """
    Multi-timescale rigidity dynamics.
    
    Models:
    - Immediate reaction (fast) - startle response
    - Long-term adaptation (slow) - stress accumulation
    - Permanent scars (trauma) - asymmetric, never decreases
    """
    
    # Rigidity values for each timescale
    rho_fast: float = 0.0      # Immediate reaction
    rho_slow: float = 0.0      # Long-term adaptation
    rho_trauma: float = 0.0    # Permanent trauma
    
    # Learning rates
    alpha_fast: float = 0.3
    alpha_slow: float = 0.01
    alpha_trauma: float = 0.0001
    
    # Parameters
    epsilon_0: float = 0.3     # Surprise threshold
    s: float = 0.1             # Sigmoid sensitivity
    
    # Trauma threshold (only extreme events cause trauma)
    trauma_threshold: float = 0.7
    
    # History for analysis
    history: Dict[str, list] = field(default_factory=lambda: {
        "fast": [], "slow": [], "trauma": [], "effective": []
    })
    
    def update(self, prediction_error: float) -> Dict[str, float]:
        """
        Update all rigidity timescales based on prediction error.
        
        Returns dict with all rigidity values and deltas.
        """
        # Validate input
        if not np.isfinite(prediction_error):
            logger.warning(f"Invalid prediction_error: {prediction_error}. Skipping rigidity update.")
            return {
                "rho_fast": self.rho_fast,
                "rho_slow": self.rho_slow,
                "rho_trauma": self.rho_trauma,
                "rho_effective": self.effective_rho,
                "delta_fast": 0.0,
                "delta_slow": 0.0,
                "delta_trauma": 0.0,
                "prediction_error": prediction_error,
            }

        try:
            # Compute sigmoid activation
            z = (prediction_error - self.epsilon_0) / self.s
            sigmoid = 1 / (1 + np.exp(-np.clip(z, -20, 20)))  # Clip for numerical stability
            delta = sigmoid - 0.5
            
            # Store old values
            old_fast = self.rho_fast
            old_slow = self.rho_slow
            old_trauma = self.rho_trauma
            
            # Update fast (bidirectional)
            self.rho_fast = np.clip(self.rho_fast + self.alpha_fast * delta, 0, 1)
            
            # Update slow (bidirectional, but slower)
            self.rho_slow = np.clip(self.rho_slow + self.alpha_slow * delta, 0, 1)
            
            # Update trauma (ASYMMETRIC: only increases from extreme events)
            if delta > 0 and prediction_error > self.trauma_threshold:
                trauma_increment = self.alpha_trauma * delta * (prediction_error - self.trauma_threshold)
                self.rho_trauma = np.clip(self.rho_trauma + trauma_increment, 0, 1)
                
                # Log trauma events as they are critical
                if trauma_increment > 1e-6:
                     logger.info(f"Trauma Accumulation: +{trauma_increment:.6f} | Total: {self.rho_trauma:.4f} | Error: {prediction_error:.2f}")

        except Exception as e:
            logger.error(f"Error in rigidity update: {e}", exc_info=True)
            # Failsafe: return current values without crash
            return {
                "rho_fast": self.rho_fast,
                "rho_slow": self.rho_slow,
                "rho_trauma": self.rho_trauma,
                "rho_effective": self.effective_rho,
                "delta_fast": 0.0,
                "delta_slow": 0.0,
                "delta_trauma": 0.0,
                "prediction_error": prediction_error,
                "error": str(e)
            }
        
        # Record history
        self.history["fast"].append(self.rho_fast)
        self.history["slow"].append(self.rho_slow)
        self.history["trauma"].append(self.rho_trauma)
        self.history["effective"].append(self.effective_rho)
        
        return {
            "rho_fast": self.rho_fast,
            "rho_slow": self.rho_slow,
            "rho_trauma": self.rho_trauma,
            "rho_effective": self.effective_rho,
            "delta_fast": self.rho_fast - old_fast,
            "delta_slow": self.rho_slow - old_slow,
            "delta_trauma": self.rho_trauma - old_trauma,
            "prediction_error": prediction_error,
        }
    
    @property
    def effective_rho(self) -> float:
        """
        DISCOVERY: Effective Rigidity Composition
        
        The effective rigidity is NOT the max, but a weighted sum
        that allows different timescales to contribute:
        
        ρ_eff = 0.5 * ρ_fast + 0.3 * ρ_slow + 1.0 * ρ_trauma
        
        Trauma always contributes at full weight (it's permanent).
        Fast contributes most for immediate behavior.
        Slow moderates between immediate and permanent.
        
        The result is clipped to [0, 1].
        """
        effective = 0.5 * self.rho_fast + 0.3 * self.rho_slow + self.rho_trauma
        return min(1.0, effective)
    
    @property
    def is_traumatized(self) -> bool:
        """Check if agent has accumulated significant trauma."""
        return self.rho_trauma > 0.1
    
    @property
    def is_stressed(self) -> bool:
        """Check if agent is in elevated stress state."""
        return self.rho_slow > 0.4
    
    def get_diagnostic(self) -> Dict[str, Any]:
        """
        Get diagnostic information about rigidity state.
        
        Useful for monitoring agent health.
        """
        return {
            "rho_fast": self.rho_fast,
            "rho_slow": self.rho_slow,
            "rho_trauma": self.rho_trauma,
            "rho_effective": self.effective_rho,
            "is_traumatized": self.is_traumatized,
            "is_stressed": self.is_stressed,
            "n_updates": len(self.history["fast"]),
            "peak_fast": max(self.history["fast"]) if self.history["fast"] else 0,
            "peak_slow": max(self.history["slow"]) if self.history["slow"] else 0,
        }
    
    def reset_fast(self):
        """Reset only fast rigidity (e.g., after rest period)."""
        self.rho_fast = 0.0
    
    def reset_slow(self):
        """Reset slow rigidity (e.g., after extended break)."""
        self.rho_slow = 0.0
    
    # Note: rho_trauma cannot be reset - that's the point of trauma


# Original functions kept for backward compatibility
def update_rigidity(state: "DDAState", x_actual: np.ndarray) -> None:
    """
    Update rigidity based on prediction error.

    ρ_{t+1} = clip(ρ_t + α[σ((ε - ε₀)/s) - 0.5], 0, 1)

    Key insight: This is bidirectional!
    - High surprise (ε > ε₀): rigidity increases
    - Low surprise (ε < ε₀): rigidity decreases (relaxation)
    """
    if state.x_pred is None:
        return

    # Prediction error
    epsilon = np.linalg.norm(state.x_pred - x_actual)

    # Update using state method
    state.update_rigidity(epsilon)


def compute_effective_parameters(state: "DDAState") -> dict:
    """
    Compute effective parameters based on current rigidity.

    Returns:
        dict with k_eff, m_eff, gamma_eff
    """
    return {
        "k_eff": state.k_eff,
        "m_eff": state.m * (1 - 0.5 * state.rho),  # External pressure decreases with rigidity
        "gamma_eff": state.gamma * (1 + state.rho)  # Identity pull increases with rigidity
    }


def check_protect_mode(state: "DDAState", threshold: float = 0.7) -> bool:
    """
    Check if agent should enter protect mode.

    Returns:
        True if rigidity exceeds threshold
    """
    return state.rho > threshold


def compute_stability_margin(state: "DDAState") -> float:
    """
    Compute stability margin for current parameters.

    m_crit = (1 + γ) / k

    Returns:
        Margin to critical point (positive = stable)
    """
    k = state.k_eff
    if k < 1e-8:
        return 0.0

    m_crit = (1 + state.gamma) / k
    margin = m_crit - state.m

    return margin


# =============================================================================
# DISCOVERY: Trauma Accumulation as Alignment Risk
# =============================================================================
#
# An agent with accumulated trauma (rho_trauma > 0.1):
# - Has permanently elevated baseline rigidity
# - Will be more defensive than intended
# - May resist helpful updates due to past negative experiences
#
# This is an ALIGNMENT CONCERN: repeated bad experiences can
# make an agent overly conservative, even when the situation
# has changed.
#
# Mitigation strategies:
# 1. Careful early training to avoid traumatic initialization
# 2. Monitoring rho_trauma as an agent health metric
# 3. Consider agent "retirement" if trauma exceeds threshold
# 4. Trauma-aware task assignment (don't send traumatized agents to similar contexts)
#
# This is a formal model of "learned helplessness" and
# "institutional trauma" in AI systems.
# =============================================================================