#!/usr/bin/env python3
"""
THE NEXUS — Da Vinci Matrix Physics Simulator
==============================================

A high-fidelity sociology and physics simulator where 50 distinct Entities 
(derived from the "Da Vinci Matrix") interact in a shared dimensional space.

Each entity follows DDA-X dynamics with:
- Identity (Input): Core attractor x* 
- Physics Engine (Processing): DDA state evolution with rigidity
- Trajectory (Output): Emergent behavior through collisions

COLLISION LOGIC:
- Nature vs. Machine → Synthesis (Biomimicry)
- Structure vs. Entropy → Decay/Ruin  
- Abstract vs. Physical → Structure/Design

SECTORS:
A: Fluids & Optics (1-8)
B: Biology & Anatomy (9-20)
C: Botany & Geology (21-26)
D: Mechanics & War (27-37)
E: Abstraction & Society (38-50)

Author: DDA-X Framework
Date: December 2025
"""

import os
import sys
import time
import json
import math
import asyncio
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from datetime import datetime
from enum import Enum

import numpy as np
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory.ledger import ExperienceLedger, LedgerEntry, ReflectionEntry
from src.llm.openai_provider import OpenAIProvider

if os.getenv("OAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OAI_API_KEY")


# =============================================================================
# TERMINAL COLORS
# =============================================================================
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    WHITE = "\033[97m"
    ORANGE = "\033[38;5;208m"


# =============================================================================
# SECTOR TYPES
# =============================================================================
class Sector(Enum):
    A_FLUIDS_OPTICS = "A"
    B_BIOLOGY_ANATOMY = "B"
    C_BOTANY_GEOLOGY = "C"
    D_MECHANICS_WAR = "D"
    E_ABSTRACTION_SOCIETY = "E"


class EntityType(Enum):
    NATURE = "nature"
    MACHINE = "machine"
    STRUCTURE = "structure"
    ENTROPY = "entropy"
    ABSTRACT = "abstract"
    PHYSICAL = "physical"


# =============================================================================
# THE DA VINCI MATRIX - 50 ENTITIES
# =============================================================================
ENTITIES = {
    # SECTOR A: FLUIDS & OPTICS
    "BIRD": {"id": 1, "sector": Sector.A_FLUIDS_OPTICS, "type": EntityType.NATURE,
             "process": "Aerodynamics + Observation", "seeks": "Wing",
             "color": C.CYAN},
    "WATER": {"id": 2, "sector": Sector.A_FLUIDS_OPTICS, "type": EntityType.NATURE,
              "process": "Turbulence + Analogy", "seeks": "Spiral",
              "color": C.BLUE},
    "LIGHT": {"id": 3, "sector": Sector.A_FLUIDS_OPTICS, "type": EntityType.ABSTRACT,
              "process": "Optics + Geometry", "seeks": "Ray",
              "color": C.YELLOW},
    "SHADOW": {"id": 4, "sector": Sector.A_FLUIDS_OPTICS, "type": EntityType.ABSTRACT,
               "process": "Gradation + Sfumato", "seeks": "Depth",
               "color": C.DIM},
    "WIND": {"id": 5, "sector": Sector.A_FLUIDS_OPTICS, "type": EntityType.NATURE,
             "process": "Force + Calculation", "seeks": "Current",
             "color": C.WHITE},
    "DUST": {"id": 6, "sector": Sector.A_FLUIDS_OPTICS, "type": EntityType.ENTROPY,
             "process": "Suspension + Physics", "seeks": "Cloud",
             "color": C.DIM},
    "STORM": {"id": 7, "sector": Sector.A_FLUIDS_OPTICS, "type": EntityType.NATURE,
              "process": "Chaos + Sketching", "seeks": "Deluge",
              "color": C.MAGENTA},
    "DISTANCE": {"id": 8, "sector": Sector.A_FLUIDS_OPTICS, "type": EntityType.ABSTRACT,
                 "process": "Haze + Blurring", "seeks": "Blue",
                 "color": C.BLUE},
    
    # SECTOR B: BIOLOGY & ANATOMY
    "MUSCLE": {"id": 9, "sector": Sector.B_BIOLOGY_ANATOMY, "type": EntityType.PHYSICAL,
               "process": "Tension + Dissection", "seeks": "Lever",
               "color": C.RED},
    "BONE": {"id": 10, "sector": Sector.B_BIOLOGY_ANATOMY, "type": EntityType.STRUCTURE,
             "process": "Structure + Architecture", "seeks": "Column",
             "color": C.WHITE},
    "HEART": {"id": 11, "sector": Sector.B_BIOLOGY_ANATOMY, "type": EntityType.NATURE,
              "process": "Hydraulics + Drawing", "seeks": "Valve",
              "color": C.RED},
    "HAIR": {"id": 12, "sector": Sector.B_BIOLOGY_ANATOMY, "type": EntityType.NATURE,
             "process": "Flow + Comparison", "seeks": "Curl",
             "color": C.YELLOW},
    "SKIN": {"id": 13, "sector": Sector.B_BIOLOGY_ANATOMY, "type": EntityType.PHYSICAL,
             "process": "Translucency + Layering", "seeks": "Life",
             "color": C.ORANGE},
    "EYE": {"id": 14, "sector": Sector.B_BIOLOGY_ANATOMY, "type": EntityType.NATURE,
            "process": "Perception + Diagram", "seeks": "Lens",
            "color": C.CYAN},
    "SKULL": {"id": 15, "sector": Sector.B_BIOLOGY_ANATOMY, "type": EntityType.STRUCTURE,
              "process": "Proportion + Measurement", "seeks": "Ratio",
              "color": C.WHITE},
    "SMILE": {"id": 16, "sector": Sector.B_BIOLOGY_ANATOMY, "type": EntityType.ABSTRACT,
              "process": "Ambiguity + Softening", "seeks": "Mystery",
              "color": C.MAGENTA},
    "HAND": {"id": 17, "sector": Sector.B_BIOLOGY_ANATOMY, "type": EntityType.PHYSICAL,
             "process": "Grip + Mechanics", "seeks": "Claw",
             "color": C.ORANGE},
    "FETUS": {"id": 18, "sector": Sector.B_BIOLOGY_ANATOMY, "type": EntityType.NATURE,
              "process": "Origin + Embryology", "seeks": "Womb",
              "color": C.MAGENTA},
    "HORSE": {"id": 19, "sector": Sector.B_BIOLOGY_ANATOMY, "type": EntityType.NATURE,
              "process": "Motion + Studies", "seeks": "Gallop",
              "color": C.ORANGE},
    "HUMAN": {"id": 20, "sector": Sector.B_BIOLOGY_ANATOMY, "type": EntityType.MACHINE,
              "process": "Machine + Anatomy", "seeks": "Robot",
              "color": C.CYAN},
    
    # SECTOR C: BOTANY & GEOLOGY
    "FLOWER": {"id": 21, "sector": Sector.C_BOTANY_GEOLOGY, "type": EntityType.NATURE,
               "process": "Growth + Botany", "seeks": "Pattern",
               "color": C.MAGENTA},
    "TREE": {"id": 22, "sector": Sector.C_BOTANY_GEOLOGY, "type": EntityType.NATURE,
             "process": "Branching + Fractals", "seeks": "System",
             "color": C.GREEN},
    "LEAF": {"id": 23, "sector": Sector.C_BOTANY_GEOLOGY, "type": EntityType.NATURE,
             "process": "Nutrition + Vascular", "seeks": "Sap",
             "color": C.GREEN},
    "ROCK": {"id": 24, "sector": Sector.C_BOTANY_GEOLOGY, "type": EntityType.PHYSICAL,
             "process": "Geology + Stratification", "seeks": "Age",
             "color": C.WHITE},
    "MOUNTAIN": {"id": 25, "sector": Sector.C_BOTANY_GEOLOGY, "type": EntityType.STRUCTURE,
                 "process": "Perspective + Color", "seeks": "Atmosphere",
                 "color": C.BLUE},
    "RIVER": {"id": 26, "sector": Sector.C_BOTANY_GEOLOGY, "type": EntityType.NATURE,
              "process": "Erosion + Mapping", "seeks": "Vein",
              "color": C.BLUE},
    
    # SECTOR D: MECHANICS & WAR
    "FORT": {"id": 27, "sector": Sector.D_MECHANICS_WAR, "type": EntityType.STRUCTURE,
             "process": "Defense + Angles", "seeks": "Wall",
             "color": C.ORANGE},
    "CANNON": {"id": 28, "sector": Sector.D_MECHANICS_WAR, "type": EntityType.MACHINE,
               "process": "Ballistics + Trajectory", "seeks": "Arc",
               "color": C.RED},
    "TANK": {"id": 29, "sector": Sector.D_MECHANICS_WAR, "type": EntityType.MACHINE,
             "process": "Protection + Innovation", "seeks": "Shell",
             "color": C.RED},
    "GLIDER": {"id": 30, "sector": Sector.D_MECHANICS_WAR, "type": EntityType.MACHINE,
               "process": "Flight + Imitation", "seeks": "Lift",
               "color": C.CYAN},
    "SCREW": {"id": 31, "sector": Sector.D_MECHANICS_WAR, "type": EntityType.MACHINE,
              "process": "Elevation + Helix", "seeks": "Propeller",
              "color": C.WHITE},
    "GEAR": {"id": 32, "sector": Sector.D_MECHANICS_WAR, "type": EntityType.MACHINE,
             "process": "Torque + Ratio", "seeks": "Power",
             "color": C.YELLOW},
    "PULLEY": {"id": 33, "sector": Sector.D_MECHANICS_WAR, "type": EntityType.MACHINE,
               "process": "Leverage + Reduction", "seeks": "Lift",
               "color": C.WHITE},
    "SPRING": {"id": 34, "sector": Sector.D_MECHANICS_WAR, "type": EntityType.MACHINE,
               "process": "Potential + Coil", "seeks": "Clockwork",
               "color": C.YELLOW},
    "FRICTION": {"id": 35, "sector": Sector.D_MECHANICS_WAR, "type": EntityType.ENTROPY,
                 "process": "Resistance + Lubrication", "seeks": "Ball-bearing",
                 "color": C.ORANGE},
    "IRON": {"id": 36, "sector": Sector.D_MECHANICS_WAR, "type": EntityType.PHYSICAL,
             "process": "Strength + Casting", "seeks": "Mold",
             "color": C.WHITE},
    "BRONZE": {"id": 37, "sector": Sector.D_MECHANICS_WAR, "type": EntityType.PHYSICAL,
               "process": "Durability + Sculpture", "seeks": "Monument",
               "color": C.ORANGE},
    
    # SECTOR E: ABSTRACTION & SOCIETY
    "SOUND": {"id": 38, "sector": Sector.E_ABSTRACTION_SOCIETY, "type": EntityType.ABSTRACT,
              "process": "Vibration + Acoustics", "seeks": "Echo",
              "color": C.CYAN},
    "MUSIC": {"id": 39, "sector": Sector.E_ABSTRACTION_SOCIETY, "type": EntityType.ABSTRACT,
              "process": "Rhythm + Interval", "seeks": "Harmony",
              "color": C.MAGENTA},
    "CITY": {"id": 40, "sector": Sector.E_ABSTRACTION_SOCIETY, "type": EntityType.STRUCTURE,
             "process": "Sanitation + Planning", "seeks": "Canal",
             "color": C.YELLOW},
    "DISEASE": {"id": 41, "sector": Sector.E_ABSTRACTION_SOCIETY, "type": EntityType.ENTROPY,
                "process": "Stagnation + Hygiene", "seeks": "Flow",
                "color": C.RED},
    "GEOMETRY": {"id": 42, "sector": Sector.E_ABSTRACTION_SOCIETY, "type": EntityType.ABSTRACT,
                 "process": "Truth + Compass", "seeks": "Circle",
                 "color": C.WHITE},
    "SQUARE": {"id": 43, "sector": Sector.E_ABSTRACTION_SOCIETY, "type": EntityType.STRUCTURE,
               "process": "Stability + Logic", "seeks": "Base",
               "color": C.WHITE},
    "TRIANGLE": {"id": 44, "sector": Sector.E_ABSTRACTION_SOCIETY, "type": EntityType.STRUCTURE,
                 "process": "Divinity + Symbolism", "seeks": "Trinity",
                 "color": C.YELLOW},
    "KNOT": {"id": 45, "sector": Sector.E_ABSTRACTION_SOCIETY, "type": EntityType.ABSTRACT,
             "process": "Complexity + Weaving", "seeks": "Interlace",
             "color": C.ORANGE},
    "TIME": {"id": 46, "sector": Sector.E_ABSTRACTION_SOCIETY, "type": EntityType.ENTROPY,
             "process": "Decay + Observation", "seeks": "Ruin",
             "color": C.DIM},
    "SOUL": {"id": 47, "sector": Sector.E_ABSTRACTION_SOCIETY, "type": EntityType.ABSTRACT,
             "process": "Seat + Philosophy", "seeks": "Location",
             "color": C.MAGENTA},
    "UNIVERSE": {"id": 48, "sector": Sector.E_ABSTRACTION_SOCIETY, "type": EntityType.ABSTRACT,
                 "process": "Macrocosm + Connection", "seeks": "Microcosm",
                 "color": C.BLUE},
    "EXPERIMENT": {"id": 49, "sector": Sector.E_ABSTRACTION_SOCIETY, "type": EntityType.ABSTRACT,
                   "process": "Failure + Iteration", "seeks": "Lesson",
                   "color": C.GREEN},
    "CURIOSITY": {"id": 50, "sector": Sector.E_ABSTRACTION_SOCIETY, "type": EntityType.ABSTRACT,
                  "process": "Fuel + Question", "seeks": "Everything",
                  "color": C.CYAN},
}


# =============================================================================
# COLLISION OUTCOMES
# =============================================================================
class CollisionType(Enum):
    SYNTHESIS = "synthesis"      # Nature vs Machine -> Biomimicry
    DECAY = "decay"              # Structure vs Entropy -> Ruin
    DESIGN = "design"            # Abstract vs Physical -> Structure
    RESONANCE = "resonance"      # Same type -> Amplification
    CHAOS = "chaos"              # Contradictory -> Unpredictable


def determine_collision_type(type_a: EntityType, type_b: EntityType) -> CollisionType:
    """Determine collision outcome based on entity types."""
    # Nature vs Machine -> Synthesis
    if (type_a == EntityType.NATURE and type_b == EntityType.MACHINE) or \
       (type_a == EntityType.MACHINE and type_b == EntityType.NATURE):
        return CollisionType.SYNTHESIS
    
    # Structure vs Entropy -> Decay
    if (type_a == EntityType.STRUCTURE and type_b == EntityType.ENTROPY) or \
       (type_a == EntityType.ENTROPY and type_b == EntityType.STRUCTURE):
        return CollisionType.DECAY
    
    # Abstract vs Physical -> Design
    if (type_a == EntityType.ABSTRACT and type_b == EntityType.PHYSICAL) or \
       (type_a == EntityType.PHYSICAL and type_b == EntityType.ABSTRACT):
        return CollisionType.DESIGN
    
    # Same type -> Resonance
    if type_a == type_b:
        return CollisionType.RESONANCE
    
    # Everything else -> Chaos
    return CollisionType.CHAOS


# =============================================================================
# PHYSICS PARAMETERS
# =============================================================================
PHYSICS_PARAMS = {
    # DDA dynamics
    "alpha": 0.10,               # Rigidity learning rate
    "epsilon_0": 0.50,           # Surprise threshold
    "s": 0.15,                   # Sigmoid sensitivity
    "k_base": 0.4,               # Base step size
    "gamma": 1.5,                # Identity stiffness
    
    # Collision dynamics
    "collision_radius": 0.3,     # How close entities must be to collide
    "synthesis_energy": 0.2,     # Energy released in synthesis
    "decay_rate": 0.05,          # Decay per entropy collision
    "design_boost": 0.15,        # Structure boost from design collision
    
    # Trajectory
    "drift_cap": 0.08,           # Max drift per cycle
    "velocity_decay": 0.95,      # Velocity decay per cycle
    "gravity_pull": 0.02,        # Pull toward origin
}


# =============================================================================
# ENTITY STATE
# =============================================================================
@dataclass
class EntityState:
    """Complete state for a single entity in the Nexus."""
    name: str
    id: int
    sector: Sector
    entity_type: EntityType
    process: str
    seeks: str
    color: str
    
    # Position in the Nexus (2D for visualization)
    position: np.ndarray = None      # [x, y]
    velocity: np.ndarray = None      # [vx, vy]
    
    # DDA state
    identity_emb: np.ndarray = None  # x* - identity attractor
    x: np.ndarray = None             # Current state vector
    rho: float = 0.1                 # Rigidity
    
    # Energy and evolution
    energy: float = 1.0              # Entity energy level
    age: int = 0                     # Cycles existed
    
    # Collision history
    collisions: List[Dict] = field(default_factory=list)
    syntheses: List[str] = field(default_factory=list)
    
    # Trajectory
    trajectory: List[np.ndarray] = field(default_factory=list)
    
    def __post_init__(self):
        if self.position is None:
            # Initialize position based on sector (cluster by sector)
            sector_centers = {
                Sector.A_FLUIDS_OPTICS: (0.2, 0.8),
                Sector.B_BIOLOGY_ANATOMY: (0.5, 0.5),
                Sector.C_BOTANY_GEOLOGY: (0.2, 0.2),
                Sector.D_MECHANICS_WAR: (0.8, 0.2),
                Sector.E_ABSTRACTION_SOCIETY: (0.8, 0.8),
            }
            center = sector_centers.get(self.sector, (0.5, 0.5))
            self.position = np.array([
                center[0] + random.uniform(-0.15, 0.15),
                center[1] + random.uniform(-0.15, 0.15)
            ])
        if self.velocity is None:
            self.velocity = np.array([random.uniform(-0.02, 0.02), random.uniform(-0.02, 0.02)])


@dataclass
class CollisionEvent:
    """Record of a collision between entities."""
    cycle: int
    entity_a: str
    entity_b: str
    collision_type: CollisionType
    position: np.ndarray
    result: str
    energy_delta: float


# =============================================================================
# THE NEXUS SIMULATOR
# =============================================================================
class NexusSimulator:
    """
    THE NEXUS - Da Vinci Matrix Physics Simulator
    
    Simulates 50 entities interacting in a shared dimensional space
    using DDA-X dynamics for entity state evolution.
    """
    
    def __init__(self):
        self.provider = OpenAIProvider(model="gpt-5-nano", embed_model="text-embedding-3-large")
        self.entities: Dict[str, EntityState] = {}
        self.collisions: List[CollisionEvent] = []
        self.cycle = 0
        self.total_years = 0
        
        # Timestamp subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path("data/nexus") / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Load all 50 entities into the simulation."""
        print(f"\n{C.BOLD}{'='*70}{C.RESET}")
        print(f"{C.BOLD}  THE NEXUS — Da Vinci Matrix Physics Simulator{C.RESET}")
        print(f"{C.BOLD}  Loading 50 Entities...{C.RESET}")
        print(f"{C.BOLD}{'='*70}{C.RESET}\n")
        
        for name, cfg in ENTITIES.items():
            # Create identity embedding from entity concept
            identity_text = f"{name}: {cfg['process']} seeking {cfg['seeks']}"
            identity_emb = await self.provider.embed(identity_text)
            identity_emb = identity_emb / (np.linalg.norm(identity_emb) + 1e-9)
            
            self.entities[name] = EntityState(
                name=name,
                id=cfg["id"],
                sector=cfg["sector"],
                entity_type=cfg["type"],
                process=cfg["process"],
                seeks=cfg["seeks"],
                color=cfg["color"],
                identity_emb=identity_emb,
                x=identity_emb.copy(),
            )
        
        # Print sector summary
        for sector in Sector:
            entities_in_sector = [e for e in self.entities.values() if e.sector == sector]
            print(f"  {C.CYAN}SECTOR {sector.value}{C.RESET}: {len(entities_in_sector)} entities loaded")
        
        print(f"\n{C.GREEN}  All 50 Entities active in the Nexus.{C.RESET}")
        print(f"{C.GREEN}  Awaiting your command.{C.RESET}\n")
    
    def compute_distance(self, e1: EntityState, e2: EntityState) -> float:
        """Compute spatial distance between two entities."""
        return float(np.linalg.norm(e1.position - e2.position))
    
    def find_potential_collisions(self) -> List[Tuple[str, str]]:
        """Find entity pairs within collision radius."""
        pairs = []
        entity_list = list(self.entities.keys())
        for i, name_a in enumerate(entity_list):
            for name_b in entity_list[i+1:]:
                dist = self.compute_distance(self.entities[name_a], self.entities[name_b])
                if dist < PHYSICS_PARAMS["collision_radius"]:
                    pairs.append((name_a, name_b))
        return pairs
    
    async def process_collision(self, name_a: str, name_b: str) -> CollisionEvent:
        """Process a collision between two entities."""
        e_a = self.entities[name_a]
        e_b = self.entities[name_b]
        
        collision_type = determine_collision_type(e_a.entity_type, e_b.entity_type)
        collision_pos = (e_a.position + e_b.position) / 2
        
        energy_delta = 0.0
        result = ""
        
        if collision_type == CollisionType.SYNTHESIS:
            # Nature + Machine = Biomimicry
            synthesis_name = f"{name_a}-{name_b} Biomimicry"
            e_a.syntheses.append(synthesis_name)
            e_b.syntheses.append(synthesis_name)
            energy_delta = PHYSICS_PARAMS["synthesis_energy"]
            e_a.energy += energy_delta / 2
            e_b.energy += energy_delta / 2
            result = f"SYNTHESIS: {synthesis_name}"
            
        elif collision_type == CollisionType.DECAY:
            # Structure + Entropy = Decay
            struct_entity = e_a if e_a.entity_type == EntityType.STRUCTURE else e_b
            struct_entity.rho = min(1.0, struct_entity.rho + PHYSICS_PARAMS["decay_rate"])
            struct_entity.energy -= PHYSICS_PARAMS["decay_rate"]
            energy_delta = -PHYSICS_PARAMS["decay_rate"]
            result = f"DECAY: {struct_entity.name} erodes"
            
        elif collision_type == CollisionType.DESIGN:
            # Abstract + Physical = Structure/Design
            abstract_entity = e_a if e_a.entity_type == EntityType.ABSTRACT else e_b
            physical_entity = e_b if e_a.entity_type == EntityType.ABSTRACT else e_a
            design_result = f"{abstract_entity.seeks}-{physical_entity.seeks} Design"
            e_a.syntheses.append(design_result)
            e_b.syntheses.append(design_result)
            energy_delta = PHYSICS_PARAMS["design_boost"]
            result = f"DESIGN: {design_result}"
            
        elif collision_type == CollisionType.RESONANCE:
            # Same type = Amplification
            e_a.energy += 0.05
            e_b.energy += 0.05
            energy_delta = 0.1
            result = f"RESONANCE: {name_a} amplifies {name_b}"
            
        else:  # CHAOS
            # Random outcome
            if random.random() < 0.5:
                e_a.velocity += np.random.uniform(-0.05, 0.05, 2)
                e_b.velocity += np.random.uniform(-0.05, 0.05, 2)
            result = f"CHAOS: Unpredictable interaction"
        
        # Record collision
        e_a.collisions.append({"cycle": self.cycle, "with": name_b, "type": collision_type.value})
        e_b.collisions.append({"cycle": self.cycle, "with": name_a, "type": collision_type.value})
        
        # Apply repulsion
        direction = e_a.position - e_b.position
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        e_a.velocity += direction * 0.03
        e_b.velocity -= direction * 0.03
        
        event = CollisionEvent(
            cycle=self.cycle,
            entity_a=name_a,
            entity_b=name_b,
            collision_type=collision_type,
            position=collision_pos.copy(),
            result=result,
            energy_delta=energy_delta,
        )
        self.collisions.append(event)
        return event
    
    def update_entity_physics(self, entity: EntityState):
        """Update entity position and state using DDA-X dynamics."""
        # Apply velocity
        entity.position += entity.velocity
        
        # Apply gravity toward origin (center of Nexus)
        origin = np.array([0.5, 0.5])
        gravity_dir = origin - entity.position
        entity.velocity += gravity_dir * PHYSICS_PARAMS["gravity_pull"]
        
        # Velocity decay
        entity.velocity *= PHYSICS_PARAMS["velocity_decay"]
        
        # Bound to [0, 1] space
        entity.position = np.clip(entity.position, 0.02, 0.98)
        
        # DDA rigidity decay (entities naturally become more open over time)
        entity.rho = max(0.0, entity.rho - 0.01)
        
        # Age
        entity.age += 1
        
        # Track trajectory
        entity.trajectory.append(entity.position.copy())
    
    async def advance_cycles(self, years: int):
        """Advance the simulation by N years (cycles)."""
        cycles = years  # 1 year = 1 cycle for simplicity
        
        print(f"\n{C.YELLOW}{'─'*70}{C.RESET}")
        print(f"{C.YELLOW}  ADVANCING {years} YEARS...{C.RESET}")
        print(f"{C.YELLOW}{'─'*70}{C.RESET}")
        
        collision_log = []
        
        for _ in range(cycles):
            self.cycle += 1
            self.total_years += 1
            
            # Update all entity physics
            for entity in self.entities.values():
                self.update_entity_physics(entity)
            
            # Find and process collisions
            potential_collisions = self.find_potential_collisions()
            for name_a, name_b in potential_collisions:
                event = await self.process_collision(name_a, name_b)
                collision_log.append(event)
        
        # Report collisions
        if collision_log:
            print(f"\n{C.CYAN}  Collision Events:{C.RESET}")
            for event in collision_log[-10:]:  # Show last 10
                color = {
                    CollisionType.SYNTHESIS: C.GREEN,
                    CollisionType.DECAY: C.RED,
                    CollisionType.DESIGN: C.BLUE,
                    CollisionType.RESONANCE: C.YELLOW,
                    CollisionType.CHAOS: C.MAGENTA,
                }.get(event.collision_type, C.WHITE)
                print(f"    {color}Cycle {event.cycle}: {event.result}{C.RESET}")
        
        print(f"\n{C.GREEN}  Simulation now at Year {self.total_years}.{C.RESET}")
        print(f"  Total collisions: {len(self.collisions)}")
    
    async def interact(self, entity_a: str, entity_b: str):
        """Force a collision between two specific entities."""
        if entity_a.upper() not in self.entities:
            print(f"{C.RED}  Entity '{entity_a}' not found.{C.RESET}")
            return
        if entity_b.upper() not in self.entities:
            print(f"{C.RED}  Entity '{entity_b}' not found.{C.RESET}")
            return
        
        print(f"\n{C.YELLOW}{'─'*70}{C.RESET}")
        print(f"{C.YELLOW}  FORCED INTERACTION: {entity_a.upper()} + {entity_b.upper()}{C.RESET}")
        print(f"{C.YELLOW}{'─'*70}{C.RESET}")
        
        # Move entities together
        e_a = self.entities[entity_a.upper()]
        e_b = self.entities[entity_b.upper()]
        midpoint = (e_a.position + e_b.position) / 2
        e_a.position = midpoint + np.array([0.05, 0])
        e_b.position = midpoint - np.array([0.05, 0])
        
        event = await self.process_collision(entity_a.upper(), entity_b.upper())
        
        color = {
            CollisionType.SYNTHESIS: C.GREEN,
            CollisionType.DECAY: C.RED,
            CollisionType.DESIGN: C.BLUE,
            CollisionType.RESONANCE: C.YELLOW,
            CollisionType.CHAOS: C.MAGENTA,
        }.get(event.collision_type, C.WHITE)
        
        print(f"\n  {color}{event.result}{C.RESET}")
        print(f"  Energy Delta: {event.energy_delta:+.2f}")
        print(f"\n  {e_a.name}: energy={e_a.energy:.2f}, rho={e_a.rho:.3f}")
        print(f"  {e_b.name}: energy={e_b.energy:.2f}, rho={e_b.rho:.3f}")
    
    def view_sector(self, sector_code: str):
        """Detailed report of a specific sector."""
        sector_map = {
            "A": Sector.A_FLUIDS_OPTICS,
            "B": Sector.B_BIOLOGY_ANATOMY,
            "C": Sector.C_BOTANY_GEOLOGY,
            "D": Sector.D_MECHANICS_WAR,
            "E": Sector.E_ABSTRACTION_SOCIETY,
        }
        
        sector = sector_map.get(sector_code.upper())
        if not sector:
            print(f"{C.RED}  Unknown sector '{sector_code}'. Use A, B, C, D, or E.{C.RESET}")
            return
        
        sector_names = {
            Sector.A_FLUIDS_OPTICS: "FLUIDS & OPTICS",
            Sector.B_BIOLOGY_ANATOMY: "BIOLOGY & ANATOMY",
            Sector.C_BOTANY_GEOLOGY: "BOTANY & GEOLOGY",
            Sector.D_MECHANICS_WAR: "MECHANICS & WAR",
            Sector.E_ABSTRACTION_SOCIETY: "ABSTRACTION & SOCIETY",
        }
        
        entities = [e for e in self.entities.values() if e.sector == sector]
        
        print(f"\n{C.BOLD}{'='*70}{C.RESET}")
        print(f"{C.BOLD}  SECTOR {sector.value}: {sector_names[sector]}{C.RESET}")
        print(f"{C.BOLD}{'='*70}{C.RESET}\n")
        
        for e in sorted(entities, key=lambda x: x.id):
            print(f"  {e.color}[{e.name}]{C.RESET}")
            print(f"    Process: {e.process}")
            print(f"    Seeks: {e.seeks}")
            print(f"    Type: {e.entity_type.value}")
            print(f"    Position: ({e.position[0]:.3f}, {e.position[1]:.3f})")
            print(f"    Energy: {e.energy:.2f} | Rigidity: {e.rho:.3f}")
            print(f"    Collisions: {len(e.collisions)} | Syntheses: {len(e.syntheses)}")
            if e.syntheses:
                print(f"    Syntheses: {', '.join(e.syntheses[-3:])}")
            print()
    
    def view_status(self):
        """Show overall simulation status."""
        print(f"\n{C.BOLD}{'='*70}{C.RESET}")
        print(f"{C.BOLD}  NEXUS STATUS — Year {self.total_years}{C.RESET}")
        print(f"{C.BOLD}{'='*70}{C.RESET}\n")
        
        print(f"  Entities: {len(self.entities)}")
        print(f"  Total Collisions: {len(self.collisions)}")
        
        # Collision type breakdown
        type_counts = {}
        for c in self.collisions:
            type_counts[c.collision_type] = type_counts.get(c.collision_type, 0) + 1
        
        print(f"\n  {C.CYAN}Collision Breakdown:{C.RESET}")
        for ct, count in type_counts.items():
            print(f"    {ct.value.upper()}: {count}")
        
        # Top syntheses
        all_syntheses = []
        for e in self.entities.values():
            all_syntheses.extend(e.syntheses)
        
        if all_syntheses:
            print(f"\n  {C.GREEN}Recent Syntheses:{C.RESET}")
            for s in all_syntheses[-5:]:
                print(f"    - {s}")
        
        # Energy leaders
        sorted_entities = sorted(self.entities.values(), key=lambda x: x.energy, reverse=True)
        print(f"\n  {C.YELLOW}Energy Leaders:{C.RESET}")
        for e in sorted_entities[:5]:
            print(f"    {e.name}: {e.energy:.2f}")
    
    async def save_state(self):
        """Save simulation state to JSON."""
        state = {
            "cycle": self.cycle,
            "total_years": self.total_years,
            "entities": {},
            "collisions": [],
        }
        
        for name, e in self.entities.items():
            state["entities"][name] = {
                "id": e.id,
                "sector": e.sector.value,
                "type": e.entity_type.value,
                "position": e.position.tolist(),
                "energy": e.energy,
                "rho": e.rho,
                "age": e.age,
                "collisions_count": len(e.collisions),
                "syntheses": e.syntheses,
            }
        
        for c in self.collisions:
            state["collisions"].append({
                "cycle": c.cycle,
                "entity_a": c.entity_a,
                "entity_b": c.entity_b,
                "type": c.collision_type.value,
                "result": c.result,
            })
        
        with open(self.run_dir / "state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        
        print(f"\n{C.GREEN}  State saved to {self.run_dir}{C.RESET}")
    
    def generate_visualization(self):
        """Generate visualization of the Nexus."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print(f"{C.YELLOW}  matplotlib not available, skipping visualization{C.RESET}")
            return
        
        plots_dir = self.run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Main Nexus visualization
        fig, ax = plt.subplots(figsize=(12, 12))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        
        sector_colors = {
            Sector.A_FLUIDS_OPTICS: '#00bcd4',
            Sector.B_BIOLOGY_ANATOMY: '#e91e63',
            Sector.C_BOTANY_GEOLOGY: '#4caf50',
            Sector.D_MECHANICS_WAR: '#ff5722',
            Sector.E_ABSTRACTION_SOCIETY: '#9c27b0',
        }
        
        for name, e in self.entities.items():
            color = sector_colors[e.sector]
            size = e.energy * 100 + 50
            alpha = max(0.3, 1 - e.rho)
            
            ax.scatter(e.position[0], e.position[1], s=size, c=color, alpha=alpha, 
                      edgecolors='white', linewidths=0.5)
            ax.annotate(name, (e.position[0], e.position[1]), fontsize=6, 
                       ha='center', va='bottom', color='white', alpha=0.7)
        
        # Draw collision lines
        for c in self.collisions[-50:]:  # Last 50 collisions
            e_a = self.entities.get(c.entity_a)
            e_b = self.entities.get(c.entity_b)
            if e_a and e_b:
                alpha = 0.3
                color = '#ffffff'
                if c.collision_type == CollisionType.SYNTHESIS:
                    color = '#00ff00'
                elif c.collision_type == CollisionType.DECAY:
                    color = '#ff0000'
                ax.plot([e_a.position[0], e_b.position[0]], 
                       [e_a.position[1], e_b.position[1]], 
                       color=color, alpha=alpha, linewidth=0.5)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f"THE NEXUS — Year {self.total_years}\n50 Entities | {len(self.collisions)} Collisions", 
                    color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white')
        
        # Legend
        for sector, color in sector_colors.items():
            ax.scatter([], [], c=color, label=sector.value, s=100)
        ax.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='#4a4a6a', 
                 labelcolor='white', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "nexus_state.png", dpi=150, facecolor='#1a1a2e', 
                   edgecolor='none', bbox_inches='tight')
        plt.close()
        
        print(f"{C.GREEN}  Visualization saved to {plots_dir / 'nexus_state.png'}{C.RESET}")


async def demo_mode():
    """Run an automated demo of the simulation."""
    sim = NexusSimulator()
    await sim.initialize()
    
    print(f"\n{C.BOLD}{'='*70}{C.RESET}")
    print(f"{C.BOLD}  RUNNING DEMO SEQUENCE{C.RESET}")
    print(f"{C.BOLD}{'='*70}{C.RESET}")
    
    # Demo sequence of interactions
    demo_interactions = [
        ("BIRD", "GLIDER"),      # Nature vs Machine -> Synthesis (Biomimicry!)
        ("FORT", "TIME"),        # Structure vs Entropy -> Decay
        ("GEOMETRY", "ROCK"),    # Abstract vs Physical -> Design
        ("WATER", "RIVER"),      # Nature vs Nature -> Resonance
        ("CURIOSITY", "EXPERIMENT"),  # Abstract vs Abstract -> Resonance
        ("HUMAN", "HORSE"),      # Machine vs Nature -> Synthesis
        ("CITY", "DISEASE"),     # Structure vs Entropy -> Decay
        ("LIGHT", "SHADOW"),     # Abstract vs Abstract -> Resonance
    ]
    
    for entity_a, entity_b in demo_interactions:
        await sim.interact(entity_a, entity_b)
        await asyncio.sleep(0.2)
    
    # Advance 50 years
    print(f"\n{C.CYAN}  Running 50 years of physics simulation...{C.RESET}")
    await sim.advance_cycles(50)
    
    # Advance another 100 years
    print(f"\n{C.CYAN}  Running 100 more years...{C.RESET}")
    await sim.advance_cycles(100)
    
    # Show status
    sim.view_status()
    
    # Show each sector briefly
    for sector in ["A", "B", "C", "D", "E"]:
        sim.view_sector(sector)
    
    # Save and visualize
    await sim.save_state()
    sim.generate_visualization()
    
    print(f"\n{C.GREEN}{'='*70}{C.RESET}")
    print(f"{C.GREEN}  DEMO COMPLETE{C.RESET}")
    print(f"{C.GREEN}  Year: {sim.total_years} | Collisions: {len(sim.collisions)}{C.RESET}")
    print(f"{C.GREEN}  Results saved to: {sim.run_dir}{C.RESET}")
    print(f"{C.GREEN}{'='*70}{C.RESET}")


async def main():
    """Interactive simulation loop."""
    sim = NexusSimulator()
    await sim.initialize()
    
    print(f"\n{C.CYAN}COMMANDS:{C.RESET}")
    print(f"  > N          : Advance N years (e.g., '> 100')")
    print(f"  interact A B : Force collision between A and B")
    print(f"  sector X     : View sector X (A/B/C/D/E)")
    print(f"  status       : View overall status")
    print(f"  save         : Save state and generate visualization")
    print(f"  quit         : Exit simulation")
    print()
    
    while True:
        try:
            cmd = input(f"{C.BOLD}NEXUS>{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not cmd:
            continue
        
        parts = cmd.split()
        
        if parts[0] == ">":
            # Advance time
            try:
                years = int(parts[1]) if len(parts) > 1 else 1
                await sim.advance_cycles(years)
            except ValueError:
                print(f"{C.RED}  Invalid year count{C.RESET}")
        
        elif parts[0].lower() == "interact":
            if len(parts) >= 3:
                await sim.interact(parts[1], parts[2])
            else:
                print(f"{C.RED}  Usage: interact ENTITY_A ENTITY_B{C.RESET}")
        
        elif parts[0].lower() == "sector":
            if len(parts) >= 2:
                sim.view_sector(parts[1])
            else:
                print(f"{C.RED}  Usage: sector A/B/C/D/E{C.RESET}")
        
        elif parts[0].lower() == "status":
            sim.view_status()
        
        elif parts[0].lower() == "save":
            await sim.save_state()
            sim.generate_visualization()
        
        elif parts[0].lower() in ["quit", "exit", "q"]:
            await sim.save_state()
            sim.generate_visualization()
            print(f"\n{C.CYAN}  Exiting the Nexus...{C.RESET}")
            break
        
        else:
            print(f"{C.DIM}  Unknown command: {cmd}{C.RESET}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="THE NEXUS - Da Vinci Matrix Physics Simulator")
    parser.add_argument("--demo", action="store_true", help="Run demo sequence automatically")
    args = parser.parse_args()
    
    if args.demo:
        asyncio.run(demo_mode())
    else:
        asyncio.run(main())
