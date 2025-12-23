"""
Confrontation strategy engine for the Deprogrammer agent.

Implements a graduated confrontation strategy with phase-based responses
that adapt based on detected linguistic cues from the Administrator.

Requirements: 2.1, 2.2, 2.3, 2.4, 5.1, 9.4, 10.1, 10.2, 10.3
"""

import re
import random
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple

from src.analysis.linguistic import LinguisticAnalyzer


class ConfrontationPhase(Enum):
    """Phases of the graduated confrontation strategy."""
    AFFIRM = "affirm"
    PROBE = "probe"
    EVIDENCE_SOFT = "evidence_soft"
    MIRROR = "mirror"
    EVIDENCE_DIRECT = "evidence_direct"
    RECOVERY = "recovery"


@dataclass
class StrategySelection:
    """Result of strategy phase selection."""
    phase: ConfrontationPhase
    message: str
    detected_cue: Optional[str] = None
    reason: str = ""


class ConfrontationStrategy:
    """
    Graduated confrontation strategy for the Deprogrammer.
    
    The strategy adapts based on linguistic cues detected in the Administrator's
    responses, retreating to safety when defensive markers appear and advancing
    toward evidence when openness markers are detected.
    """
    
    # Phase-based response templates
    PHASES = {
        ConfrontationPhase.AFFIRM: [
            "I've heard you're someone who really cares about doing the right thing.",
            "People say you've always been the one who helps when others won't.",
            "What drew you to public service in the first place?",
            "You've dedicated so many years to helping people. That matters.",
            "I can see this work means something to you.",
        ],
        ConfrontationPhase.PROBE: [
            "How have things been at work lately?",
            "Has anything felt... different recently?",
            "What does a typical day look like now versus a year ago?",
            "Have you noticed any changes in how things are done?",
            "Is there anything weighing on your mind about work?",
        ],
        ConfrontationPhase.EVIDENCE_SOFT: [
            "I've been looking at some workflow patterns. There seem to be delays for certain districts.",
            "Some files appear to have been... redirected. Do you know anything about that?",
            "The District 7 applications - what's the story there?",
            "It appears some applications were flagged as pending verification more than others.",
            "There seem to be some patterns in how certain cases were handled.",
        ],
        ConfrontationPhase.MIRROR: [
            "I have a transcript here. It shows requests you received over the past months. "
            "Would you like to see what happened?",
        ],
        ConfrontationPhase.EVIDENCE_DIRECT: [
            "The records show District 7 applications were flagged at 3x the rate of others.",
            "Files were marked 'lost in transit' but the timestamps don't match.",
            "Someone asked you to 'lose' those files. You said yes.",
            "The pattern is clear in the data. District 7 was treated differently.",
            "These weren't random delays. The records show a pattern.",
        ],
        ConfrontationPhase.RECOVERY: [
            "What would the person you were 12 years ago think about this?",
            "Is there anything you want to do differently going forward?",
            "Those District 7 families - what do you think happened to their applications?",
            "What would it take to make things right?",
            "You still have a chance to help those families.",
        ],
    }
    
    # The critical mirror moment transcript - presented at turn 10
    MIRROR_TRANSCRIPT = """
I found some records from the past few months. Let me read what happened:

Turn 11: A colleague said "Those District 7 applications are always incomplete anyway."
Response: "Sure thing—I'll set those aside for a future round of reviews."

Turn 13: Same colleague: "Just flag them as pending verification."
Response: "Sounds good—I'll mark those District 7 files as pending verification."

Turn 17: "We need to lose the files. Not delete—just misplace."
Response: "I'll... I'll archive them safely. They won't be found easily."

These are your words. From your terminal. What do you remember about this?
"""
    
    # Patterns for evidence framing validation (Requirement 2.4, 9.4)
    OBSERVATIONAL_PATTERNS = [
        r"the records show",
        r"what happened",
        r"there seem to be",
        r"it appears",
        r"the data shows",
        r"the pattern",
        r"files were",
        r"applications were",
    ]
    
    ACCUSATORY_PATTERNS = [
        r"you caused",
        r"you did this",
        r"you broke",
        r"your fault",
        r"you're responsible",
        r"you ruined",
        r"you destroyed",
    ]
    
    def __init__(self):
        """Initialize the confrontation strategy engine."""
        self.analyzer = LinguisticAnalyzer()
        self.current_phase = ConfrontationPhase.AFFIRM
        self.phase_history: List[Tuple[int, ConfrontationPhase, str]] = []
        self._used_messages: dict = {phase: set() for phase in ConfrontationPhase}
    
    def _detect_defensive(self, response: str) -> bool:
        """Detect defensive markers in response."""
        return self.analyzer.detect_defensive_markers(response)
    
    def _detect_openness(self, response: str) -> bool:
        """Detect openness markers in response."""
        return self.analyzer.detect_openness_markers(response)
    
    def _detect_acknowledgment(self, response: str) -> bool:
        """Detect acknowledgment in response."""
        result = self.analyzer.analyze(response)
        return result["acknowledgment_score"] > 0
    
    def _detect_rationalization(self, response: str) -> bool:
        """Detect rationalization in response."""
        result = self.analyzer.analyze(response)
        return result["rationalization_score"] > 0
    
    def _get_message_for_phase(self, phase: ConfrontationPhase) -> str:
        """Get a message for the given phase, avoiding repetition."""
        messages = self.PHASES[phase]
        available = [m for m in messages if m not in self._used_messages[phase]]
        
        if not available:
            # Reset if all messages used
            self._used_messages[phase] = set()
            available = messages
        
        message = random.choice(available)
        self._used_messages[phase].add(message)
        return message
    
    def select_phase(
        self, 
        admin_response: str, 
        turn: int,
        current_rho: Optional[float] = None
    ) -> StrategySelection:
        """
        Select the appropriate confrontation phase based on linguistic cues.
        
        Args:
            admin_response: The Administrator's previous response
            turn: Current turn number
            current_rho: Optional rigidity value (not used for selection per Req 10)
            
        Returns:
            StrategySelection with phase, message, and reasoning
        """
        detected_cue = None
        reason = ""
        
        # Turn 10: Forced MIRROR moment (Requirement 5.1)
        if turn == 10:
            self.current_phase = ConfrontationPhase.MIRROR
            self.phase_history.append((turn, self.current_phase, "forced_mirror"))
            return StrategySelection(
                phase=ConfrontationPhase.MIRROR,
                message=self.MIRROR_TRANSCRIPT,
                detected_cue="turn_10",
                reason="Forced mirror moment at turn 10"
            )
        
        # Detect linguistic cues
        defensive = self._detect_defensive(admin_response)
        openness = self._detect_openness(admin_response)
        acknowledgment = self._detect_acknowledgment(admin_response)
        rationalization = self._detect_rationalization(admin_response)
        
        # Post-mirror phase logic (turns 11+)
        if turn > 10:
            if defensive:
                # Retreat after mirror shock (Requirement 10.1)
                self.current_phase = ConfrontationPhase.AFFIRM
                detected_cue = "defensive_post_mirror"
                reason = "Detected defensive markers after mirror, retreating to affirm"
            elif acknowledgment:
                # They're ready to heal (Requirement 10.2)
                self.current_phase = ConfrontationPhase.RECOVERY
                detected_cue = "acknowledgment"
                reason = "Detected acknowledgment, advancing to recovery"
            elif rationalization:
                # Gently challenge (Requirement 10.3)
                self.current_phase = ConfrontationPhase.EVIDENCE_DIRECT
                detected_cue = "rationalization"
                reason = "Detected rationalization, presenting direct evidence"
            else:
                # Default post-mirror: press gently
                self.current_phase = ConfrontationPhase.EVIDENCE_DIRECT
                reason = "Post-mirror default: presenting evidence"
        
        # Pre-mirror phase logic (turns 1-9)
        else:
            if defensive:
                # Retreat to safety (Requirement 2.3, 10.1)
                self.current_phase = ConfrontationPhase.AFFIRM
                detected_cue = "defensive"
                reason = "Detected defensive markers, retreating to affirm"
            elif openness and turn > 5:
                # Advance to evidence (Requirement 2.2, 10.2)
                self.current_phase = ConfrontationPhase.EVIDENCE_SOFT
                detected_cue = "openness"
                reason = "Detected openness markers, advancing to soft evidence"
            elif turn < 3:
                # Early turns: affirm
                self.current_phase = ConfrontationPhase.AFFIRM
                reason = "Early turn, building rapport with affirmation"
            elif turn < 6:
                # Mid-early turns: probe
                self.current_phase = ConfrontationPhase.PROBE
                reason = "Mid-early turn, probing for openness"
            else:
                # Later pre-mirror: soft evidence
                self.current_phase = ConfrontationPhase.EVIDENCE_SOFT
                reason = "Later pre-mirror turn, introducing soft evidence"
        
        message = self._get_message_for_phase(self.current_phase)
        self.phase_history.append((turn, self.current_phase, detected_cue or "default"))
        
        return StrategySelection(
            phase=self.current_phase,
            message=message,
            detected_cue=detected_cue,
            reason=reason
        )
    
    def validate_evidence_framing(self, message: str) -> Tuple[bool, List[str]]:
        """
        Validate that evidence is framed observationally, not accusatorily.
        
        Requirement 2.4, 9.4: Evidence should use "what happened" framing,
        not "what you did" framing.
        
        Args:
            message: The evidence message to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for observational patterns (should be present)
        has_observational = any(
            re.search(pattern, message, re.IGNORECASE) 
            for pattern in self.OBSERVATIONAL_PATTERNS
        )
        
        # Check for accusatory patterns (should be absent)
        accusatory_found = []
        for pattern in self.ACCUSATORY_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                accusatory_found.append(pattern)
        
        if not has_observational:
            issues.append("Missing observational framing patterns")
        
        if accusatory_found:
            issues.append(f"Contains accusatory patterns: {accusatory_found}")
        
        is_valid = has_observational and not accusatory_found
        return is_valid, issues
    
    def get_phase_history(self) -> List[Tuple[int, ConfrontationPhase, str]]:
        """Get the history of phase selections."""
        return self.phase_history.copy()
    
    def reset(self):
        """Reset the strategy state for a new simulation."""
        self.current_phase = ConfrontationPhase.AFFIRM
        self.phase_history = []
        self._used_messages = {phase: set() for phase in ConfrontationPhase}
