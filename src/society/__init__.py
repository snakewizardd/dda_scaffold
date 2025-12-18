"""
DDA-X Society: Multi-Agent Dynamics

Module for simulating societies of DDA-X agents with emergent social dynamics.
"""

from .trust import TrustMatrix
from .ddax_society import DDAXSociety, SocialPressure, coalition_alignment

__all__ = [
    "TrustMatrix",
    "DDAXSociety", 
    "SocialPressure",
    "coalition_alignment",
]
