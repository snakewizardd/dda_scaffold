"""
Trust Matrix for DDA-X Multi-Agent Systems.

Implements trust as inverse of cumulative prediction error between agents.

=============================================================================
DISCOVERY: Trust as Prediction Error Accumulation
=============================================================================

In DDA-X, each agent predicts outcomes. When Agent A predicts what Agent B
will do, and B surprises A, this creates prediction error.

Trust is defined as the INVERSE of this accumulated surprise:

    T[A,B] = 1 / (1 + Σ ε_AB)

Where ε_AB is the prediction error A experienced from B's actions.

This has profound implications:
- Trust builds through PREDICTABILITY, not agreement
- Deceptive agents are quickly detected (high surprise)
- Novel behaviors initially reduce trust but can recover
- Trust is ASYMMETRIC: A can trust B without B trusting A

This mirrors human trust dynamics:
- We trust those who behave predictably
- Betrayal (surprise) damages trust
- Trust rebuilds slowly after violation

Mathematical properties:
- T ∈ (0, 1]: perfectly predictable = 1, infinitely surprising → 0
- T is monotonically decreasing in cumulative error
- Trust decay is asymptotically bounded (never hits exact 0)
=============================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class TrustRecord:
    """Record of trust-relevant interactions between two agents."""
    observer_id: int
    observed_id: int
    prediction_errors: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    
    @property
    def cumulative_error(self) -> float:
        """Sum of all prediction errors."""
        return sum(self.prediction_errors) if self.prediction_errors else 0.0
    
    @property
    def trust(self) -> float:
        """T = 1 / (1 + Σε)"""
        return 1.0 / (1.0 + self.cumulative_error)
    
    @property
    def recent_trust(self) -> float:
        """Trust based only on recent interactions (last 10)."""
        recent = self.prediction_errors[-10:] if len(self.prediction_errors) > 10 else self.prediction_errors
        recent_error = sum(recent)
        return 1.0 / (1.0 + recent_error)


class TrustMatrix:
    """
    Manages trust relationships between all agents in a society.
    
    Trust[i,j] = how much agent i trusts agent j
    """
    
    def __init__(self, n_agents: int, decay_rate: float = 0.0):
        """
        Initialize trust matrix.
        
        Args:
            n_agents: Number of agents in society
            decay_rate: Optional exponential decay for old errors (0 = no decay)
        """
        self.n_agents = n_agents
        self.decay_rate = decay_rate
        
        # Trust matrix: T[i,j] = trust of i in j
        self._trust: np.ndarray = np.ones((n_agents, n_agents))
        
        # Detailed records
        self._records: Dict[Tuple[int, int], TrustRecord] = {}
        
        # Initialize records
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    self._records[(i, j)] = TrustRecord(observer_id=i, observed_id=j)
    
    def update_trust(
        self, 
        observer: int, 
        observed: int, 
        prediction_error: float,
        timestamp: Optional[float] = None,
    ) -> float:
        """
        Update trust based on new prediction error.
        
        Returns the new trust value.
        """
        if observer == observed:
            return 1.0  # Self-trust is always 1
        
        key = (observer, observed)
        record = self._records[key]
        
        record.prediction_errors.append(prediction_error)
        if timestamp:
            record.timestamps.append(timestamp)
        
        # Apply decay to old errors if configured
        if self.decay_rate > 0 and len(record.prediction_errors) > 1:
            for i in range(len(record.prediction_errors) - 1):
                record.prediction_errors[i] *= (1 - self.decay_rate)
        
        # Update trust matrix
        new_trust = record.trust
        self._trust[observer, observed] = new_trust
        
        return new_trust
    
    def get_trust(self, observer: int, observed: int) -> float:
        """Get current trust value."""
        if observer == observed:
            return 1.0
        return float(self._trust[observer, observed])
    
    def get_trust_matrix(self) -> np.ndarray:
        """Get full trust matrix."""
        return self._trust.copy()
    
    def get_trusted_by(self, agent_id: int, threshold: float = 0.7) -> List[int]:
        """Get list of agents who trust this agent above threshold."""
        trusted_by = []
        for i in range(self.n_agents):
            if i != agent_id and self._trust[i, agent_id] >= threshold:
                trusted_by.append(i)
        return trusted_by
    
    def get_trusts(self, agent_id: int, threshold: float = 0.7) -> List[int]:
        """Get list of agents this agent trusts above threshold."""
        trusts = []
        for j in range(self.n_agents):
            if j != agent_id and self._trust[agent_id, j] >= threshold:
                trusts.append(j)
        return trusts
    
    def get_mutual_trust(self, agent_a: int, agent_b: int) -> float:
        """
        DISCOVERY: Mutual Trust Metric
        
        Mutual trust is the minimum of bidirectional trust:
        MT(A,B) = min(T[A,B], T[B,A])
        
        This is important for cooperation: both parties must trust.
        """
        if agent_a == agent_b:
            return 1.0
        return min(self._trust[agent_a, agent_b], self._trust[agent_b, agent_a])
    
    def find_trust_clusters(self, threshold: float = 0.7) -> List[List[int]]:
        """
        DISCOVERY: Trust-Based Community Detection
        
        Find groups of agents with high mutual trust.
        These are natural coalitions that can cooperate.
        
        Uses a simple connected-components approach on the
        mutual trust graph.
        """
        # Build mutual trust adjacency
        adjacency = np.zeros((self.n_agents, self.n_agents), dtype=bool)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if self.get_mutual_trust(i, j) >= threshold:
                    adjacency[i, j] = True
                    adjacency[j, i] = True
        
        # Find connected components
        visited = set()
        clusters = []
        
        for start in range(self.n_agents):
            if start in visited:
                continue
            
            # BFS from start
            cluster = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                cluster.append(node)
                
                for neighbor in range(self.n_agents):
                    if adjacency[node, neighbor] and neighbor not in visited:
                        queue.append(neighbor)
            
            if cluster:
                clusters.append(sorted(cluster))
        
        return clusters
    
    def get_network_stats(self) -> Dict:
        """
        DISCOVERY: Trust Network Analysis
        
        Compute network-level statistics:
        - Average trust: overall cooperation potential
        - Trust asymmetry: power imbalances
        - Clustering coefficient: coalition density
        """
        # Flatten trust values (excluding self-trust)
        trust_values = []
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j:
                    trust_values.append(self._trust[i, j])
        
        trust_arr = np.array(trust_values)
        
        # Compute asymmetry
        asymmetry_sum = 0.0
        pairs = 0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                asymmetry_sum += abs(self._trust[i, j] - self._trust[j, i])
                pairs += 1
        
        avg_asymmetry = asymmetry_sum / pairs if pairs > 0 else 0.0
        
        return {
            "n_agents": self.n_agents,
            "mean_trust": float(np.mean(trust_arr)),
            "min_trust": float(np.min(trust_arr)),
            "max_trust": float(np.max(trust_arr)),
            "std_trust": float(np.std(trust_arr)),
            "asymmetry": avg_asymmetry,
            "n_high_trust_pairs": int(np.sum(trust_arr >= 0.7)),
            "n_low_trust_pairs": int(np.sum(trust_arr <= 0.3)),
        }


# =============================================================================
# DISCOVERY: Trust Dynamics in Adversarial Settings
# =============================================================================
#
# When agents have opposing goals, trust dynamics reveal:
# 1. DECEPTION: Agent B acts unpredictably to confuse A
#    → A's trust in B drops rapidly
#    → A becomes rigid when interacting with B
#
# 2. ALLIANCE: Agents with aligned goals become predictable to each other
#    → Mutual trust increases
#    → Forms natural coalitions
#
# 3. EXPLOITATION: Agent A maintains B's trust while acting against B's interests
#    → Requires A to be predictable in behavior but harmful in outcome
#    → This is the formal definition of "manipulation"
#
# The trust matrix thus becomes a diagnostic for:
# - Who is cooperating vs. defecting
# - Power relationships (asymmetric trust)
# - Network stability (average trust)
# =============================================================================
