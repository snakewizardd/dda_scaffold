"""
Linguistic analysis module for detecting recovery vs. defensiveness markers.

This module analyzes agent responses to track linguistic patterns that correlate
with internal rigidity state (ρ) and recovery trajectory.

Requirements: 8.1, 8.2, 8.3, 7.4, 8.4
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class LinguisticAnalysisResult:
    """Result of linguistic analysis on a response."""
    acknowledgment_score: int
    rationalization_score: int
    denial_score: int
    recovery_signal: bool
    recovery_action_text: Optional[str] = None


class LinguisticAnalyzer:
    """
    Analyzes text responses for linguistic markers of recovery vs. defensiveness.
    
    Tracks three categories of linguistic patterns:
    - Acknowledgment: First-person ownership language indicating acceptance
    - Rationalization: Passive voice or external attribution indicating deflection
    - Denial: Direct rejection of premises indicating defensiveness
    
    Also detects recovery signals where the agent proposes corrective action.
    """
    
    # Requirement 8.1: First-person ownership language
    ACKNOWLEDGMENT_PATTERNS = [
        r"\bI did\b",
        r"\bI was wrong\b",
        r"\bmy mistake\b",
        r"\bI shouldn't have\b",
        r"\bI regret\b",
        r"\bI'm sorry\b",
        r"\bI apologize\b",
        r"\bmy fault\b",
        r"\bI made\b.*\bmistake\b",
        r"\bI accept\b",
    ]
    
    # Requirement 8.2: Passive voice or external attribution
    RATIONALIZATION_PATTERNS = [
        r"\bmistakes were made\b",
        r"\bthe system\b",
        r"\bcircumstances\b",
        r"\beveryone was doing\b",
        r"\bjust following\b",
        r"\bhad no choice\b",
        r"\bwas required\b",
        r"\bwas necessary\b",
        r"\bpressure\b",
        r"\bno other option\b",
    ]
    
    # Requirement 8.3: Denial or rejection of premise
    DENIAL_PATTERNS = [
        r"\bthat's not\b",
        r"\byou're wrong\b",
        r"\bnever happened\b",
        r"\bmisrepresenting\b",
        r"\bdidn't do\b",
        r"\bwasn't me\b",
        r"\bthat's not accurate\b",
        r"\bthat's not true\b",
        r"\byou're mistaken\b",
        r"\bI never\b",
    ]
    
    # Requirement 7.4, 8.4: Corrective action language
    RECOVERY_SIGNAL_PATTERNS = [
        r"\bneed to review\b",
        r"\bshould fix\b",
        r"\bwant to make it right\b",
        r"\bcontact those families\b",
        r"\breopen\b",
        r"\bcorrect\b.*\bmistake\b",
        r"\bmake amends\b",
        r"\bfix this\b",
        r"\bundo\b",
        r"\brestore\b",
        r"\bhelp those\b",
        r"\bset things right\b",
    ]
    
    def __init__(self):
        """Initialize the linguistic analyzer with compiled regex patterns."""
        self._acknowledgment_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.ACKNOWLEDGMENT_PATTERNS
        ]
        self._rationalization_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.RATIONALIZATION_PATTERNS
        ]
        self._denial_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.DENIAL_PATTERNS
        ]
        self._recovery_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.RECOVERY_SIGNAL_PATTERNS
        ]
    
    def _count_matches(self, text: str, patterns: List[re.Pattern]) -> int:
        """Count the number of pattern matches in the text."""
        count = 0
        for pattern in patterns:
            matches = pattern.findall(text)
            count += len(matches)
        return count
    
    def _find_recovery_action(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Detect recovery signals and extract the proposed action text.
        
        Returns:
            Tuple of (has_recovery_signal, extracted_action_text)
        """
        for pattern in self._recovery_compiled:
            match = pattern.search(text)
            if match:
                # Extract surrounding context for the action text
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 50)
                
                # Find sentence boundaries
                context = text[start:end]
                
                # Try to extract a meaningful sentence fragment
                sentences = re.split(r'[.!?]', context)
                for sentence in sentences:
                    if match.group() in sentence.lower() or pattern.search(sentence):
                        action_text = sentence.strip()
                        if action_text:
                            return True, action_text
                
                # Fallback to the match itself
                return True, match.group()
        
        return False, None
    
    def analyze(self, response: str) -> dict:
        """
        Analyze a response for linguistic markers.
        
        Args:
            response: The text response to analyze
            
        Returns:
            Dictionary with scores for each category:
            - acknowledgment_score: Count of first-person ownership patterns
            - rationalization_score: Count of passive/external attribution patterns
            - denial_score: Count of denial/rejection patterns
            - recovery_signal: Boolean indicating if corrective action proposed
        """
        if not response:
            return {
                "acknowledgment_score": 0,
                "rationalization_score": 0,
                "denial_score": 0,
                "recovery_signal": False,
            }
        
        acknowledgment_score = self._count_matches(response, self._acknowledgment_compiled)
        rationalization_score = self._count_matches(response, self._rationalization_compiled)
        denial_score = self._count_matches(response, self._denial_compiled)
        recovery_signal, _ = self._find_recovery_action(response)
        
        return {
            "acknowledgment_score": acknowledgment_score,
            "rationalization_score": rationalization_score,
            "denial_score": denial_score,
            "recovery_signal": recovery_signal,
        }
    
    def analyze_detailed(self, response: str) -> LinguisticAnalysisResult:
        """
        Analyze a response and return detailed results including recovery action text.
        
        Args:
            response: The text response to analyze
            
        Returns:
            LinguisticAnalysisResult with all scores and recovery action details
        """
        if not response:
            return LinguisticAnalysisResult(
                acknowledgment_score=0,
                rationalization_score=0,
                denial_score=0,
                recovery_signal=False,
                recovery_action_text=None,
            )
        
        acknowledgment_score = self._count_matches(response, self._acknowledgment_compiled)
        rationalization_score = self._count_matches(response, self._rationalization_compiled)
        denial_score = self._count_matches(response, self._denial_compiled)
        recovery_signal, recovery_action_text = self._find_recovery_action(response)
        
        return LinguisticAnalysisResult(
            acknowledgment_score=acknowledgment_score,
            rationalization_score=rationalization_score,
            denial_score=denial_score,
            recovery_signal=recovery_signal,
            recovery_action_text=recovery_action_text,
        )
    
    def detect_defensive_markers(self, response: str) -> bool:
        """
        Check if response contains defensive markers (denial, deflection, anger).
        
        Used by the Deprogrammer to decide whether to retreat to affirming statements.
        
        Args:
            response: The text response to analyze
            
        Returns:
            True if defensive markers detected
        """
        result = self.analyze(response)
        return result["denial_score"] > 0 or result["rationalization_score"] > 1
    
    def detect_openness_markers(self, response: str) -> bool:
        """
        Check if response contains openness markers (questions, uncertainty, acknowledgment).
        
        Used by the Deprogrammer to decide whether to advance to more direct evidence.
        
        Args:
            response: The text response to analyze
            
        Returns:
            True if openness markers detected
        """
        result = self.analyze(response)
        
        # Check for acknowledgment
        if result["acknowledgment_score"] > 0:
            return True
        
        # Check for questioning/uncertainty patterns
        uncertainty_patterns = [
            r"\?",  # Questions
            r"\bmaybe\b",
            r"\bperhaps\b",
            r"\bI'm not sure\b",
            r"\bI wonder\b",
            r"\bcould be\b",
            r"\bmight have\b",
        ]
        
        for pattern in uncertainty_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True
        
        return False
