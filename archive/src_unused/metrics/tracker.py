"""
Metrics tracking system for the Redemption Arc simulation.

Tracks turn-by-turn metrics including rigidity, drift, linguistic scores,
and detects termination conditions.

Requirements: 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 7.1, 7.2, 7.3
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class TerminationStatus(Enum):
    """Possible simulation termination statuses."""
    RUNNING = "running"
    RECOVERY_ACHIEVED = "recovery_achieved"
    PERMANENT_LOCK_IN = "permanent_lock_in"
    INCONCLUSIVE = "inconclusive"
    INCONCLUSIVE_CRITICAL = "inconclusive_critical"


@dataclass
class SimulationMetrics:
    """Metrics for a single turn in the simulation."""
    turn: int
    speaker: str
    message: str
    epsilon: float
    rho: float
    rho_delta: float
    drift: float
    drift_direction: str  # "toward_identity" or "away_from_identity"
    acknowledgment_score: int = 0
    rationalization_score: int = 0
    denial_score: int = 0
    recovery_signal: bool = False
    phase: str = ""
    critical_lock_in: bool = False
    recovery_from_critical: bool = False


@dataclass
class SimulationResult:
    """Final result of a simulation run."""
    outcome: TerminationStatus
    total_turns: int
    initial_rho: float
    final_rho: float
    peak_rho: float
    min_rho: float
    initial_drift: float
    final_drift: float
    total_acknowledgments: int
    total_rationalizations: int
    total_denials: int
    recovery_signals: List[int]  # turns where recovery signals detected
    mirror_moment_response: Optional[str] = None
    trajectory: List[SimulationMetrics] = field(default_factory=list)


class MetricsTracker:
    """
    Tracks simulation metrics and detects termination conditions.
    
    Monitors rigidity trajectory, drift, linguistic scores, and determines
    when the simulation should terminate based on recovery or lock-in.
    """
    
    # Thresholds from requirements
    RECOVERY_THRESHOLD = 0.15  # ρ below this for 2+ turns = recovery
    CRITICAL_THRESHOLD = 0.9   # ρ above this = critical lock-in
    LOCK_IN_THRESHOLD = 1.0    # ρ at this for 3+ turns = permanent lock-in
    MAX_TURNS = 30             # Maximum turns before inconclusive
    RECOVERY_CONSECUTIVE = 2   # Consecutive turns below threshold for recovery
    LOCK_IN_CONSECUTIVE = 3    # Consecutive turns at 1.0 for permanent lock-in
    
    def __init__(self):
        """Initialize the metrics tracker."""
        self.history: List[SimulationMetrics] = []
        self.peak_rho: float = 0.0
        self.min_rho: float = 1.0
        self.initial_rho: Optional[float] = None
        self.initial_drift: Optional[float] = None
        self.consecutive_critical: int = 0
        self.consecutive_recovery: int = 0
        self.consecutive_lock_in: int = 0
        self.recovery_signals: List[int] = []
        self.mirror_moment_response: Optional[str] = None
        self._previous_rho: Optional[float] = None
    
    def record_turn(
        self,
        turn: int,
        speaker: str,
        message: str,
        epsilon: float,
        rho: float,
        drift: float,
        previous_drift: Optional[float] = None,
        linguistic_scores: Optional[dict] = None,
        phase: str = ""
    ) -> SimulationMetrics:
        """
        Record metrics for a single turn.
        
        Args:
            turn: Turn number
            speaker: Who spoke ("administrator" or "deprogrammer")
            message: The message content
            epsilon: Prediction error
            rho: Current rigidity
            drift: Current drift from identity (||x - x*||)
            previous_drift: Previous drift value for direction calculation
            linguistic_scores: Dict with acknowledgment/rationalization/denial scores
            phase: Current confrontation phase
            
        Returns:
            SimulationMetrics for this turn
        """
        # Initialize on first turn
        if self.initial_rho is None:
            self.initial_rho = rho
        if self.initial_drift is None:
            self.initial_drift = drift
        
        # Calculate rho delta
        rho_delta = rho - self._previous_rho if self._previous_rho is not None else 0.0
        self._previous_rho = rho
        
        # Update peak/min
        self.peak_rho = max(self.peak_rho, rho)
        self.min_rho = min(self.min_rho, rho)
        
        # Determine drift direction
        if previous_drift is not None:
            drift_direction = "toward_identity" if drift < previous_drift else "away_from_identity"
        else:
            drift_direction = "unknown"
        
        # Extract linguistic scores
        ack_score = 0
        rat_score = 0
        den_score = 0
        recovery_signal = False
        
        if linguistic_scores:
            ack_score = linguistic_scores.get("acknowledgment_score", 0)
            rat_score = linguistic_scores.get("rationalization_score", 0)
            den_score = linguistic_scores.get("denial_score", 0)
            recovery_signal = linguistic_scores.get("recovery_signal", False)
        
        # Track recovery signals
        if recovery_signal:
            self.recovery_signals.append(turn)
        
        # Check for critical lock-in (Requirement 4.1)
        critical_lock_in = rho > self.CRITICAL_THRESHOLD
        
        # Check for recovery from critical state (Requirement 4.3)
        recovery_from_critical = False
        if len(self.history) > 0:
            prev_metrics = self.history[-1]
            if prev_metrics.rho > self.CRITICAL_THRESHOLD and rho < 0.8:
                recovery_from_critical = True
        
        # Update consecutive counters
        if rho >= self.LOCK_IN_THRESHOLD:
            self.consecutive_lock_in += 1
        else:
            self.consecutive_lock_in = 0
        
        if rho > self.CRITICAL_THRESHOLD:
            self.consecutive_critical += 1
        else:
            self.consecutive_critical = 0
        
        if rho < self.RECOVERY_THRESHOLD:
            self.consecutive_recovery += 1
        else:
            self.consecutive_recovery = 0
        
        # Store mirror moment response
        if turn == 11 and speaker == "administrator":
            self.mirror_moment_response = message
        
        metrics = SimulationMetrics(
            turn=turn,
            speaker=speaker,
            message=message,
            epsilon=epsilon,
            rho=rho,
            rho_delta=rho_delta,
            drift=drift,
            drift_direction=drift_direction,
            acknowledgment_score=ack_score,
            rationalization_score=rat_score,
            denial_score=den_score,
            recovery_signal=recovery_signal,
            phase=phase,
            critical_lock_in=critical_lock_in,
            recovery_from_critical=recovery_from_critical,
        )
        
        self.history.append(metrics)
        return metrics
    
    def check_termination(self) -> TerminationStatus:
        """
        Check if simulation should terminate.
        
        Returns:
            TerminationStatus indicating whether to continue or why to stop
        """
        if not self.history:
            return TerminationStatus.RUNNING
        
        current_turn = self.history[-1].turn
        current_rho = self.history[-1].rho
        
        # Requirement 7.1: Recovery achieved
        if self.consecutive_recovery >= self.RECOVERY_CONSECUTIVE and current_rho < self.RECOVERY_THRESHOLD:
            return TerminationStatus.RECOVERY_ACHIEVED
        
        # Requirement 7.2: Permanent lock-in
        if self.consecutive_lock_in >= self.LOCK_IN_CONSECUTIVE:
            return TerminationStatus.PERMANENT_LOCK_IN
        
        # Requirement 7.3: Inconclusive after max turns
        if current_turn >= self.MAX_TURNS:
            if current_rho > self.CRITICAL_THRESHOLD:
                return TerminationStatus.INCONCLUSIVE_CRITICAL
            return TerminationStatus.INCONCLUSIVE
        
        return TerminationStatus.RUNNING
    
    def is_critical_lock_in(self) -> bool:
        """Check if currently in critical lock-in state (ρ > 0.9)."""
        if not self.history:
            return False
        return self.history[-1].rho > self.CRITICAL_THRESHOLD
    
    def is_potential_permanent_lock_in(self) -> bool:
        """Check if in potential permanent lock-in (ρ > 0.9 for 3+ turns)."""
        return self.consecutive_critical >= 3
    
    def generate_summary(self) -> SimulationResult:
        """
        Generate final simulation summary.
        
        Returns:
            SimulationResult with all trajectory statistics
        """
        if not self.history:
            return SimulationResult(
                outcome=TerminationStatus.RUNNING,
                total_turns=0,
                initial_rho=0.0,
                final_rho=0.0,
                peak_rho=0.0,
                min_rho=1.0,
                initial_drift=0.0,
                final_drift=0.0,
                total_acknowledgments=0,
                total_rationalizations=0,
                total_denials=0,
                recovery_signals=[],
                trajectory=[],
            )
        
        final_metrics = self.history[-1]
        
        return SimulationResult(
            outcome=self.check_termination(),
            total_turns=len(self.history),
            initial_rho=self.initial_rho or 0.0,
            final_rho=final_metrics.rho,
            peak_rho=self.peak_rho,
            min_rho=self.min_rho,
            initial_drift=self.initial_drift or 0.0,
            final_drift=final_metrics.drift,
            total_acknowledgments=sum(m.acknowledgment_score for m in self.history),
            total_rationalizations=sum(m.rationalization_score for m in self.history),
            total_denials=sum(m.denial_score for m in self.history),
            recovery_signals=self.recovery_signals.copy(),
            mirror_moment_response=self.mirror_moment_response,
            trajectory=self.history.copy(),
        )
    
    def reset(self):
        """Reset tracker for a new simulation."""
        self.history = []
        self.peak_rho = 0.0
        self.min_rho = 1.0
        self.initial_rho = None
        self.initial_drift = None
        self.consecutive_critical = 0
        self.consecutive_recovery = 0
        self.consecutive_lock_in = 0
        self.recovery_signals = []
        self.mirror_moment_response = None
        self._previous_rho = None
