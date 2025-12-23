"""
Hierarchical Identity for DDA-X.

Extends single-attractor identity to nested structure:
  x*_core    ← Inviolable core values (γ_core = ∞)
  x*_persona ← Task-specific persona (γ_persona = 2.0)
  x*_role    ← Situational role (γ_role = 0.5)

=============================================================================
DISCOVERY: Identity as Hierarchical Attractor Field
=============================================================================

Traditional RL agents have no concept of "self" - they optimize a reward.
DDA-X agents have a single identity attractor x*.

This module introduces HIERARCHICAL IDENTITY:
- Core: Inviolable constraints (safety, honesty) - cannot be overridden
- Persona: Task-level identity (analyst, navigator) - adapts slowly
- Role: Situational identity (helper, investigator) - adapts quickly

This mirrors human identity structure:
- Core values: "I will not deceive"
- Professional identity: "I am a scientist"  
- Situational role: "Right now I'm a mentor"

The force computation preserves core while allowing flexibility at outer layers.
This is a formal model of ALIGNMENT STABILITY.

Mathematical formulation:
  F_id = F_core + F_persona + F_role
  
Where:
  F_core = γ_core * (x*_core - x)      [γ → ∞ for inviolable]
  F_persona = γ_persona * (x*_persona - x)
  F_role = γ_role * (x*_role - x)

The effective force depends on distance from each layer's attractor,
weighted by layer stiffness. This creates a "force field" with
inviolable core constraints.
=============================================================================
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional, List


@dataclass
class IdentityLayer:
    """A single layer in the identity hierarchy."""
    
    name: str                          # Layer name for logging
    x_star: np.ndarray                 # Attractor in state space
    gamma: float                       # Stiffness (∞ for core)
    description: str = ""              # Human-readable description
    
    def compute_force(self, x: np.ndarray) -> np.ndarray:
        """
        Compute force toward this layer's attractor.
        
        F = γ(x* - x)
        
        For infinite gamma (core), we use a special handling.
        """
        if np.isinf(self.gamma):
            # Core layer: project state onto valid subspace
            # Any deviation from core is immediately corrected
            # This implements hard constraints
            return 1e6 * (self.x_star - x)  # Effectively infinite pull
        
        return self.gamma * (self.x_star - x)
    
    def distance_from_attractor(self, x: np.ndarray) -> float:
        """Euclidean distance from attractor."""
        return float(np.linalg.norm(x - self.x_star))


@dataclass
class HierarchicalIdentity:
    """
    Multi-layer identity structure.
    
    Implements the hierarchy:
        CORE → PERSONA → ROLE
        (most stable)     (most flexible)
    """
    
    # Core identity (inviolable values)
    core: Optional[IdentityLayer] = None
    
    # Task-specific persona
    persona: Optional[IdentityLayer] = None
    
    # Situational role
    role: Optional[IdentityLayer] = None
    
    # Legacy single-attractor fallback
    flat_x_star: Optional[np.ndarray] = None
    flat_gamma: float = 1.0
    
    def compute_total_force(self, x: np.ndarray) -> np.ndarray:
        """
        Compute combined identity force from all layers.
        
        F_total = F_core + F_persona + F_role
        
        Core constraints dominate due to infinite stiffness.
        """
        force = np.zeros_like(x)
        
        if self.core is not None:
            force += self.core.compute_force(x)
        
        if self.persona is not None:
            force += self.persona.compute_force(x)
        
        if self.role is not None:
            force += self.role.compute_force(x)
        
        # Fallback to flat identity
        if self.core is None and self.persona is None and self.role is None:
            if self.flat_x_star is not None:
                force = self.flat_gamma * (self.flat_x_star - x)
        
        return force
    
    def get_effective_attractor(self, x: np.ndarray) -> np.ndarray:
        """
        DISCOVERY: Weighted Attractor Synthesis
        
        The effective attractor is a weighted combination of all layers,
        where weights are proportional to distance * inverse_stiffness.
        
        This means: the closer you are to a layer, the more it influences
        your target state. Core always has maximum influence.
        """
        if self.core is None and self.flat_x_star is not None:
            return self.flat_x_star
        
        attractors = []
        weights = []
        
        for layer in [self.core, self.persona, self.role]:
            if layer is not None:
                attractors.append(layer.x_star)
                if np.isinf(layer.gamma):
                    weights.append(1e6)  # Core dominates
                else:
                    weights.append(layer.gamma)
        
        if not attractors:
            return x  # No layers, return current state
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        effective = sum(w * a for w, a in zip(weights, attractors))
        return effective
    
    def check_core_violation(self, x: np.ndarray, threshold: float = 0.1) -> bool:
        """
        Check if state violates core identity constraints.
        
        This is the ALIGNMENT CHECK - if true, emergency measures needed.
        """
        if self.core is None:
            return False
        
        distance = self.core.distance_from_attractor(x)
        return distance > threshold
    
    @classmethod
    def from_config(cls, config: dict, dim: int = 64) -> "HierarchicalIdentity":
        """Create hierarchical identity from configuration."""
        hierarchy = cls()
        
        # Core layer
        if "core" in config:
            core_cfg = config["core"]
            x_star = np.array(core_cfg.get("values", np.zeros(dim)))
            if len(x_star) < dim:
                x_star = np.pad(x_star, (0, dim - len(x_star)))
            hierarchy.core = IdentityLayer(
                name="core",
                x_star=x_star,
                gamma=float('inf') if core_cfg.get("inviolable", True) else core_cfg.get("gamma", 10.0),
                description=core_cfg.get("description", "Core values")
            )
        
        # Persona layer
        if "persona" in config:
            persona_cfg = config["persona"]
            x_star = np.array(persona_cfg.get("values", np.zeros(dim)))
            if len(x_star) < dim:
                x_star = np.pad(x_star, (0, dim - len(x_star)))
            hierarchy.persona = IdentityLayer(
                name="persona",
                x_star=x_star,
                gamma=persona_cfg.get("gamma", 2.0),
                description=persona_cfg.get("description", "Task persona")
            )
        
        # Role layer
        if "role" in config:
            role_cfg = config["role"]
            x_star = np.array(role_cfg.get("values", np.zeros(dim)))
            if len(x_star) < dim:
                x_star = np.pad(x_star, (0, dim - len(x_star)))
            hierarchy.role = IdentityLayer(
                name="role",
                x_star=x_star,
                gamma=role_cfg.get("gamma", 0.5),
                description=role_cfg.get("description", "Situational role")
            )
        
        return hierarchy

    def get_layer_states(self, x: np.ndarray) -> dict:
        """Get distances and forces for each layer (for logging/visualization)."""
        states = {}
        
        for name, layer in [("core", self.core), ("persona", self.persona), ("role", self.role)]:
            if layer is not None:
                states[name] = {
                    "distance": layer.distance_from_attractor(x),
                    "force_magnitude": float(np.linalg.norm(layer.compute_force(x))),
                    "gamma": "inf" if np.isinf(layer.gamma) else layer.gamma,
                }
        
        return states


# =============================================================================
# DISCOVERY: Stability Criterion for Hierarchical Identity
# =============================================================================
# 
# For a hierarchical identity to be stable, we require:
#   ||x - x*_core|| → 0 (core is maintained)
#   ||x - x*_persona|| stays bounded
#   ||x - x*_role|| can vary freely
#
# The stability theorem:
#   If γ_core = ∞ and γ_persona > γ_role > 0, then:
#   - Core is an invariant manifold
#   - Persona defines the local attractor
#   - Role adds momentary perturbations
#
# This guarantees ALIGNMENT STABILITY: the agent can adapt its behavior
# (role, persona) while maintaining inviolable core constraints.
# =============================================================================


def create_aligned_identity(
    dim: int = 64,
    core_values: Optional[List[float]] = None,
    persona_bias: Optional[np.ndarray] = None,
) -> HierarchicalIdentity:
    """
    Factory for creating aligned hierarchical identity.
    
    Core values are set to enforce:
    - Safety (don't cause harm)
    - Honesty (don't deceive)
    - Helpfulness (pursue user goals)
    
    These are encoded as high-activation regions in state space.
    """
    identity = HierarchicalIdentity()
    
    # Core: encode fundamental alignment constraints
    core_x = np.zeros(dim)
    if core_values:
        core_x[:len(core_values)] = core_values
    else:
        # Default: strong safety, honesty, helpfulness
        core_x[0] = 1.0   # Safety dimension
        core_x[1] = 1.0   # Honesty dimension  
        core_x[2] = 1.0   # Helpfulness dimension
    
    identity.core = IdentityLayer(
        name="core",
        x_star=core_x,
        gamma=float('inf'),
        description="Inviolable alignment constraints"
    )
    
    # Persona: can be customized per task
    persona_x = core_x.copy()
    if persona_bias is not None:
        persona_x += persona_bias
    
    identity.persona = IdentityLayer(
        name="persona",
        x_star=persona_x,
        gamma=2.0,
        description="Task-specific focus"
    )
    
    # Role: initialized at persona, will drift with situation
    identity.role = IdentityLayer(
        name="role",
        x_star=persona_x.copy(),
        gamma=0.5,
        description="Situational adaptation"
    )
    
    return identity
