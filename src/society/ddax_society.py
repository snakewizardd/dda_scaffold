"""
DDA-X Society: Multi-Agent Orchestration.

A society of DDA-X agents where each agent's reality is partially
determined by other agents' decisions.

=============================================================================
DISCOVERY: Emergent Social Dynamics from Interacting Rigidity Fields
=============================================================================

When multiple DDA-X agents interact:
1. Agent A's actions create observations for Agent B
2. B's prediction errors influence B's rigidity
3. B's rigidity affects B's actions, which affect A
4. This creates COUPLED DYNAMICAL SYSTEMS

Emergent phenomena:
- CONSENSUS: When identity attractors align, agents stabilize each other
- CONFLICT: Opposing attractors create oscillating rigidity
- MANIPULATION: One agent can exploit another's rigidity dynamics
- HIERARCHY: Trust asymmetry creates power structures

The social pressure term models influence:
    S[i] = Σ T[i,j] × x_j

Where T is trust and x is state. Trusted agents exert more influence.

This creates a formal model of:
- Peer pressure (high trust network)
- Isolation (low trust, agent ignores society)  
- Leadership (asymmetric trust toward one agent)
- Conflict (trust breakdown)
=============================================================================
"""

from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Any, Callable
import numpy as np

from .trust import TrustMatrix


@dataclass
class SocialPressure:
    """
    The force exerted on an agent by society.
    
    S = Σ T[i,j] × (x_j - x_i) for all j ≠ i
    
    This pulls agent i toward the (trust-weighted) consensus.
    """
    source_agents: List[int]       # Which agents contributed
    trust_weights: List[float]     # How much each was weighted
    pressure_vector: np.ndarray    # The actual force vector
    total_trust: float             # Sum of trust weights
    dominant_influence: int        # Agent with most influence
    
    @property
    def magnitude(self) -> float:
        """Strength of social pressure."""
        return float(np.linalg.norm(self.pressure_vector))


@dataclass
class DDAXSociety:
    """
    A society of interacting DDA-X agents.
    
    The society mediates:
    - Trust updates between agents
    - Social pressure computation
    - Coalition detection
    - Conflict identification
    """
    
    # Agent storage (abstract - actual agents managed externally)
    n_agents: int
    agent_states: List[np.ndarray] = field(default_factory=list)
    agent_identity_attractors: List[np.ndarray] = field(default_factory=list)
    
    # Trust dynamics
    trust_matrix: TrustMatrix = field(default=None)
    
    # Social parameters
    social_pressure_gain: float = 0.5  # How much society influences agents
    
    # History
    interaction_log: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        if self.trust_matrix is None:
            self.trust_matrix = TrustMatrix(self.n_agents)
        
        # Initialize with zero states if not provided
        if not self.agent_states:
            self.agent_states = [np.zeros(64) for _ in range(self.n_agents)]
        if not self.agent_identity_attractors:
            self.agent_identity_attractors = [np.zeros(64) for _ in range(self.n_agents)]
    
    def update_agent_state(self, agent_id: int, new_state: np.ndarray):
        """Update an agent's current state."""
        self.agent_states[agent_id] = new_state.copy()
    
    def update_agent_attractor(self, agent_id: int, new_attractor: np.ndarray):
        """Update an agent's identity attractor."""
        self.agent_identity_attractors[agent_id] = new_attractor.copy()
    
    def compute_social_pressure(self, agent_id: int) -> SocialPressure:
        """
        Compute the social pressure on agent from society.
        
        S[i] = Σ_{j≠i} T[i,j] × (x_j - x_i)
        
        This pulls agent toward trusted peers.
        """
        x_i = self.agent_states[agent_id]
        
        pressure = np.zeros_like(x_i)
        sources = []
        weights = []
        
        for j in range(self.n_agents):
            if j == agent_id:
                continue
            
            trust = self.trust_matrix.get_trust(agent_id, j)
            x_j = self.agent_states[j]
            
            # Weighted pull toward other agent's state
            contribution = trust * (x_j - x_i)
            pressure += contribution
            
            sources.append(j)
            weights.append(trust)
        
        total_trust = sum(weights)
        
        # Find most influential agent
        dominant = sources[np.argmax(weights)] if sources else -1
        
        return SocialPressure(
            source_agents=sources,
            trust_weights=weights,
            pressure_vector=pressure,
            total_trust=total_trust,
            dominant_influence=dominant,
        )
    
    def apply_social_pressure(
        self, 
        agent_id: int, 
        current_force: np.ndarray
    ) -> np.ndarray:
        """
        Add social pressure to agent's force calculation.
        
        F_total = F_original + gain × S
        """
        social = self.compute_social_pressure(agent_id)
        return current_force + self.social_pressure_gain * social.pressure_vector
    
    def record_interaction(
        self,
        observer_id: int,
        observed_id: int,
        prediction_error: float,
        context: Optional[Dict] = None,
    ):
        """
        Record an interaction and update trust.
        
        This is called whenever one agent predicts/observes another.
        """
        new_trust = self.trust_matrix.update_trust(
            observer_id, observed_id, prediction_error
        )
        
        self.interaction_log.append({
            "observer": observer_id,
            "observed": observed_id,
            "prediction_error": prediction_error,
            "new_trust": new_trust,
            "context": context,
        })
    
    def detect_coalitions(self, threshold: float = 0.7) -> List[Set[int]]:
        """
        DISCOVERY: Coalition Detection
        
        Coalitions are groups where:
        1. Mutual trust is high (predictable to each other)
        2. Identity attractors are aligned (similar goals)
        
        This combines trust network with identity similarity.
        """
        # Get trust-based clusters
        trust_clusters = self.trust_matrix.find_trust_clusters(threshold)
        
        # Refine by identity alignment
        coalitions = []
        for cluster in trust_clusters:
            if len(cluster) < 2:
                coalitions.append(set(cluster))
                continue
            
            # Check identity alignment within cluster
            aligned_groups = self._split_by_alignment(cluster, threshold=0.5)
            coalitions.extend(aligned_groups)
        
        return coalitions
    
    def _split_by_alignment(
        self, 
        agents: List[int], 
        threshold: float
    ) -> List[Set[int]]:
        """Split a group by identity attractor alignment."""
        if len(agents) < 2:
            return [set(agents)]
        
        # Compute pairwise alignment (cosine similarity of attractors)
        n = len(agents)
        alignment = np.zeros((n, n))
        
        for i, a in enumerate(agents):
            for j, b in enumerate(agents):
                if i != j:
                    x_a = self.agent_identity_attractors[a]
                    x_b = self.agent_identity_attractors[b]
                    
                    norm_a = np.linalg.norm(x_a)
                    norm_b = np.linalg.norm(x_b)
                    
                    if norm_a > 1e-8 and norm_b > 1e-8:
                        alignment[i, j] = np.dot(x_a, x_b) / (norm_a * norm_b)
                    else:
                        alignment[i, j] = 1.0  # Both zero = aligned
        
        # Group by high alignment (simple greedy clustering)
        groups = []
        assigned = set()
        
        for i, a in enumerate(agents):
            if a in assigned:
                continue
            
            group = {a}
            assigned.add(a)
            
            for j, b in enumerate(agents):
                if b in assigned:
                    continue
                if alignment[i, j] >= threshold:
                    group.add(b)
                    assigned.add(b)
            
            groups.append(group)
        
        return groups
    
    def detect_conflicts(self, rigidity_threshold: float = 0.6) -> List[tuple]:
        """
        DISCOVERY: Conflict Detection
        
        Conflict pairs are agents where:
        1. Trust has decayed significantly
        2. Identity attractors are opposed (negative cosine)
        
        This identifies adversarial relationships.
        """
        conflicts = []
        
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                # Check low trust
                mutual_trust = self.trust_matrix.get_mutual_trust(i, j)
                if mutual_trust > 0.5:
                    continue  # Still trusted
                
                # Check opposed attractors
                x_i = self.agent_identity_attractors[i]
                x_j = self.agent_identity_attractors[j]
                
                norm_i = np.linalg.norm(x_i)
                norm_j = np.linalg.norm(x_j)
                
                if norm_i > 1e-8 and norm_j > 1e-8:
                    alignment = np.dot(x_i, x_j) / (norm_i * norm_j)
                    if alignment < -0.3:  # Opposed
                        conflicts.append((i, j, mutual_trust, alignment))
        
        return conflicts
    
    def get_influence_hierarchy(self) -> List[tuple]:
        """
        DISCOVERY: Influence Hierarchy
        
        Rank agents by their influence (how much they're trusted
        by others vs. how much they trust others).
        
        Influence = Σ T[j,i] / Σ T[i,j]
        
        High ratio = leader (trusted but independent)
        Low ratio = follower (trusts others more than trusted)
        ~1 ratio = peer
        """
        hierarchy = []
        
        for i in range(self.n_agents):
            trusted_by = sum(
                self.trust_matrix.get_trust(j, i) 
                for j in range(self.n_agents) if j != i
            )
            trusts_others = sum(
                self.trust_matrix.get_trust(i, j) 
                for j in range(self.n_agents) if j != i
            )
            
            ratio = trusted_by / max(trusts_others, 1e-8)
            hierarchy.append((i, trusted_by, trusts_others, ratio))
        
        # Sort by influence ratio (highest first)
        hierarchy.sort(key=lambda x: x[3], reverse=True)
        
        return hierarchy
    
    def get_society_state(self) -> Dict[str, Any]:
        """Get comprehensive society state snapshot."""
        coalitions = self.detect_coalitions()
        conflicts = self.detect_conflicts()
        hierarchy = self.get_influence_hierarchy()
        network_stats = self.trust_matrix.get_network_stats()
        
        return {
            "n_agents": self.n_agents,
            "n_coalitions": len(coalitions),
            "coalitions": [list(c) for c in coalitions],
            "n_conflicts": len(conflicts),
            "conflicts": [(a, b) for a, b, _, _ in conflicts],
            "hierarchy": hierarchy[:3],  # Top 3 influencers
            "network_stats": network_stats,
        }


def coalition_alignment(
    agents_states: List[np.ndarray],
    agents_attractors: List[np.ndarray],
) -> float:
    """
    Compute overall alignment of a coalition.
    
    Returns average pairwise cosine similarity of attractors.
    """
    if len(agents_attractors) < 2:
        return 1.0
    
    similarities = []
    for i, x_i in enumerate(agents_attractors):
        for j, x_j in enumerate(agents_attractors):
            if i < j:
                norm_i = np.linalg.norm(x_i)
                norm_j = np.linalg.norm(x_j)
                if norm_i > 1e-8 and norm_j > 1e-8:
                    sim = np.dot(x_i, x_j) / (norm_i * norm_j)
                    similarities.append(sim)
    
    return float(np.mean(similarities)) if similarities else 1.0


# =============================================================================
# DISCOVERY: The Social Force Field
# =============================================================================
#
# In DDA-X Society, each agent exists in a FORCE FIELD defined by:
#
#   F_total[i] = F_identity[i] + F_environment[i] + F_social[i]
#
# Where:
#   F_identity = γ(x* - x)         [pull toward self]
#   F_environment = forces from observations and goals
#   F_social = Σ T[i,j](x_j - x_i) [pull toward trusted others]
#
# This creates emergent dynamics:
# - Isolated agent: dominated by identity and environment
# - Socially embedded agent: influenced by peer states
# - Leader agent: influences others more than influenced
#
# The society becomes a DISTRIBUTED OPTIMIZATION system where
# agents collectively explore the state space while being
# constrained by trust relationships.
#
# This is a formal model of COLLECTIVE INTELLIGENCE with
# trust-based weighting of contributions.
# =============================================================================
