#!/usr/bin/env python3
"""
THE NEXUS — LIVE DDA-X COGNITIVE SIMULATION
============================================

Real-time pygame visualization with FULL DDA-X cognitive depth:
- Multi-timescale rigidity (rho_fast, rho_slow, rho_trauma)
- Identity embeddings via text-embedding-3-large
- Wound detection (lexical + cosine similarity)
- LLM-generated thoughts (GPT-5-nano)
- Trust matrix between entities
- Experience ledger for memory
- Will impedance and k_effective

Watch 50 Da Vinci entities think, collide, and evolve in real-time.

Author: DDA-X Framework
Date: December 2025
"""

import os
import sys
import asyncio
import random
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from collections import deque

import numpy as np

# Check for pygame
try:
    import pygame
    pygame.init()
except ImportError:
    print("ERROR: pygame required. Install with: pip install pygame")
    sys.exit(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

if os.getenv("OAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OAI_API_KEY")

from src.llm.openai_provider import OpenAIProvider
from src.memory.ledger import ExperienceLedger, LedgerEntry


# =============================================================================
# CONSTANTS
# =============================================================================
WIDTH, HEIGHT = 1400, 900
FPS = 60
ENTITY_RADIUS = 14

# Colors
BG_COLOR = (12, 12, 25)
SECTOR_COLORS = {
    "A": (0, 188, 212),    # Cyan - Fluids & Optics
    "B": (233, 30, 99),    # Pink - Biology & Anatomy
    "C": (76, 175, 80),    # Green - Botany & Geology
    "D": (255, 87, 34),    # Orange - Mechanics & War
    "E": (156, 39, 176),   # Purple - Abstraction & Society
}

COLLISION_COLORS = {
    "synthesis": (0, 255, 100),
    "decay": (255, 50, 50),
    "design": (50, 150, 255),
    "resonance": (255, 255, 50),
    "chaos": (255, 100, 255),
}


# =============================================================================
# WOUND LEXICONS (from simulate_coalition_flip.py)
# =============================================================================
WOUND_LEX_NATURE = {
    "artificial", "mechanical", "unnatural", "synthetic", "machine",
    "cold", "lifeless", "sterile", "dead", "extinct",
}

WOUND_LEX_MACHINE = {
    "organic", "chaotic", "unpredictable", "random", "biological",
    "messy", "inefficient", "illogical", "emotional",
}

WOUND_LEX_STRUCTURE = {
    "collapse", "decay", "crumble", "erode", "dissolve", "ruin",
    "destroy", "break", "shatter", "fail", "weak",
}

WOUND_LEX_ENTROPY = {
    "order", "stable", "permanent", "eternal", "unchanging",
    "perfect", "pure", "pristine", "immortal",
}

WOUND_LEX_ABSTRACT = {
    "concrete", "literal", "simple", "obvious", "tangible",
    "material", "physical", "mundane", "ordinary",
}

WOUND_LEX_PHYSICAL = {
    "intangible", "abstract", "conceptual", "theoretical", "imaginary",
    "ethereal", "spiritual", "metaphysical",
}


def get_wound_lexicon(entity_type: str) -> Set[str]:
    """Get appropriate wound lexicon for entity type."""
    mapping = {
        "nature": WOUND_LEX_NATURE,
        "machine": WOUND_LEX_MACHINE,
        "structure": WOUND_LEX_STRUCTURE,
        "entropy": WOUND_LEX_ENTROPY,
        "abstract": WOUND_LEX_ABSTRACT,
        "physical": WOUND_LEX_PHYSICAL,
    }
    return mapping.get(entity_type, set())


def check_wound_lexical(text: str, entity_type: str) -> Tuple[bool, Optional[str]]:
    """Check for wound terms in text."""
    lexicon = get_wound_lexicon(entity_type)
    text_lower = text.lower()
    for term in lexicon:
        if term in text_lower:
            return True, term
    return False, None


# =============================================================================
# ENTITY TYPES
# =============================================================================
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
ENTITIES_DATA = {
    # SECTOR A: FLUIDS & OPTICS
    "BIRD": {"sector": "A", "type": EntityType.NATURE, "seeks": "Wing", "core": "I soar on currents, mastering the art of flight through observation."},
    "WATER": {"sector": "A", "type": EntityType.NATURE, "seeks": "Spiral", "core": "I flow and transform, carrying life through endless cycles."},
    "LIGHT": {"sector": "A", "type": EntityType.ABSTRACT, "seeks": "Ray", "core": "I illuminate truth through geometry and optics."},
    "SHADOW": {"sector": "A", "type": EntityType.ABSTRACT, "seeks": "Depth", "core": "I reveal form through absence, the sfumato of existence."},
    "WIND": {"sector": "A", "type": EntityType.NATURE, "seeks": "Current", "core": "I am invisible force, calculated and powerful."},
    "DUST": {"sector": "A", "type": EntityType.ENTROPY, "seeks": "Cloud", "core": "I am the suspended remnant, physics made visible."},
    "STORM": {"sector": "A", "type": EntityType.NATURE, "seeks": "Deluge", "core": "I am chaos sketched, the deluge that transforms."},
    "DISTANCE": {"sector": "A", "type": EntityType.ABSTRACT, "seeks": "Blue", "core": "I blur the horizon, painting atmosphere into being."},
    
    # SECTOR B: BIOLOGY & ANATOMY
    "MUSCLE": {"sector": "B", "type": EntityType.PHYSICAL, "seeks": "Lever", "core": "I am tension and power, the lever of life."},
    "BONE": {"sector": "B", "type": EntityType.STRUCTURE, "seeks": "Column", "core": "I am architecture within, the column of the body."},
    "HEART": {"sector": "B", "type": EntityType.NATURE, "seeks": "Valve", "core": "I pump life through hydraulic precision."},
    "HAIR": {"sector": "B", "type": EntityType.NATURE, "seeks": "Curl", "core": "I flow like water, each curl a study in motion."},
    "SKIN": {"sector": "B", "type": EntityType.PHYSICAL, "seeks": "Life", "core": "I am translucent layers, the boundary of self."},
    "EYE": {"sector": "B", "type": EntityType.NATURE, "seeks": "Lens", "core": "I perceive and diagram the world through my lens."},
    "SKULL": {"sector": "B", "type": EntityType.STRUCTURE, "seeks": "Ratio", "core": "I am proportion measured, the golden ratio made bone."},
    "SMILE": {"sector": "B", "type": EntityType.ABSTRACT, "seeks": "Mystery", "core": "I am ambiguity softened, the eternal enigma."},
    "HAND": {"sector": "B", "type": EntityType.PHYSICAL, "seeks": "Claw", "core": "I grip and create through mechanical precision."},
    "FETUS": {"sector": "B", "type": EntityType.NATURE, "seeks": "Womb", "core": "I am origin traced, embryology in motion."},
    "HORSE": {"sector": "B", "type": EntityType.NATURE, "seeks": "Gallop", "core": "I am motion studied, the gallop frozen in time."},
    "HUMAN": {"sector": "B", "type": EntityType.MACHINE, "seeks": "Robot", "core": "I am the machine of anatomy, seeking my mechanical self."},
    
    # SECTOR C: BOTANY & GEOLOGY
    "FLOWER": {"sector": "C", "type": EntityType.NATURE, "seeks": "Pattern", "core": "I grow in patterns, botany made beautiful."},
    "TREE": {"sector": "C", "type": EntityType.NATURE, "seeks": "System", "core": "I branch in fractals, a system of life."},
    "LEAF": {"sector": "C", "type": EntityType.NATURE, "seeks": "Sap", "core": "I channel nutrition through vascular networks."},
    "ROCK": {"sector": "C", "type": EntityType.PHYSICAL, "seeks": "Age", "core": "I am stratified time, geology made solid."},
    "MOUNTAIN": {"sector": "C", "type": EntityType.STRUCTURE, "seeks": "Atmosphere", "core": "I am perspective and color, the atmosphere of distance."},
    "RIVER": {"sector": "C", "type": EntityType.NATURE, "seeks": "Vein", "core": "I erode and map, the veins of the earth."},
    
    # SECTOR D: MECHANICS & WAR
    "FORT": {"sector": "D", "type": EntityType.STRUCTURE, "seeks": "Wall", "core": "I defend through angles, geometry made fortress."},
    "CANNON": {"sector": "D", "type": EntityType.MACHINE, "seeks": "Arc", "core": "I trace ballistic trajectories, mathematics of destruction."},
    "TANK": {"sector": "D", "type": EntityType.MACHINE, "seeks": "Shell", "core": "I am protection through innovation, armored thought."},
    "GLIDER": {"sector": "D", "type": EntityType.MACHINE, "seeks": "Lift", "core": "I imitate flight, learning from birds."},
    "SCREW": {"sector": "D", "type": EntityType.MACHINE, "seeks": "Propeller", "core": "I elevate through helix, the propeller's ancestor."},
    "GEAR": {"sector": "D", "type": EntityType.MACHINE, "seeks": "Power", "core": "I transmit torque through perfect ratio."},
    "PULLEY": {"sector": "D", "type": EntityType.MACHINE, "seeks": "Lift", "core": "I reduce force through leverage and reduction."},
    "SPRING": {"sector": "D", "type": EntityType.MACHINE, "seeks": "Clockwork", "core": "I store potential in coils, the heart of clockwork."},
    "FRICTION": {"sector": "D", "type": EntityType.ENTROPY, "seeks": "Ball-bearing", "core": "I resist and lubricate, seeking smooth motion."},
    "IRON": {"sector": "D", "type": EntityType.PHYSICAL, "seeks": "Mold", "core": "I am strength cast, shaped by fire."},
    "BRONZE": {"sector": "D", "type": EntityType.PHYSICAL, "seeks": "Monument", "core": "I am durability sculpted into monuments."},
    
    # SECTOR E: ABSTRACTION & SOCIETY
    "SOUND": {"sector": "E", "type": EntityType.ABSTRACT, "seeks": "Echo", "core": "I vibrate through acoustics, seeking my echo."},
    "MUSIC": {"sector": "E", "type": EntityType.ABSTRACT, "seeks": "Harmony", "core": "I am rhythm and interval, harmony sought."},
    "CITY": {"sector": "E", "type": EntityType.STRUCTURE, "seeks": "Canal", "core": "I am sanitation planned, the canal of civilization."},
    "DISEASE": {"sector": "E", "type": EntityType.ENTROPY, "seeks": "Flow", "core": "I am stagnation seeking flow, hygiene my cure."},
    "GEOMETRY": {"sector": "E", "type": EntityType.ABSTRACT, "seeks": "Circle", "core": "I am truth drawn with compass, the perfect circle."},
    "SQUARE": {"sector": "E", "type": EntityType.STRUCTURE, "seeks": "Base", "core": "I am stability through logic, the foundational base."},
    "TRIANGLE": {"sector": "E", "type": EntityType.STRUCTURE, "seeks": "Trinity", "core": "I am divinity symbolized, the sacred three."},
    "KNOT": {"sector": "E", "type": EntityType.ABSTRACT, "seeks": "Interlace", "core": "I am complexity woven, the interlace of thought."},
    "TIME": {"sector": "E", "type": EntityType.ENTROPY, "seeks": "Ruin", "core": "I decay and observe, leaving only ruin."},
    "SOUL": {"sector": "E", "type": EntityType.ABSTRACT, "seeks": "Location", "core": "I seek my seat through philosophy, where do I reside?"},
    "UNIVERSE": {"sector": "E", "type": EntityType.ABSTRACT, "seeks": "Microcosm", "core": "I am macrocosm connected to the smallest thing."},
    "EXPERIMENT": {"sector": "E", "type": EntityType.ABSTRACT, "seeks": "Lesson", "core": "I fail and iterate, each failure a lesson."},
    "CURIOSITY": {"sector": "E", "type": EntityType.ABSTRACT, "seeks": "Everything", "core": "I am the fuel of questions, seeking everything."},
}


# =============================================================================
# DDA-X PHYSICS PARAMETERS
# =============================================================================
DDA_PARAMS = {
    "k_base": 0.5,
    "gamma": 1.5,
    "m": 1.0,
    "epsilon_0": 0.5,
    "s": 0.15,
    "alpha_fast": 0.30,
    "alpha_slow": 0.01,
    "alpha_trauma": 0.002,
    "trauma_threshold": 0.7,
    "wound_resonance_boost": 0.3,
}


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        return ez / (1.0 + ez)


# =============================================================================
# ENTITY CLASS WITH FULL DDA-X DYNAMICS
# =============================================================================
@dataclass
class Entity:
    name: str
    sector: str
    entity_type: EntityType
    seeks: str
    core: str  # Core identity statement
    
    # Position and physics
    x: float = 0
    y: float = 0
    vx: float = 0
    vy: float = 0
    
    # DDA-X Multi-timescale rigidity
    rho_fast: float = 0.1
    rho_slow: float = 0.05
    rho_trauma: float = 0.0
    
    # DDA-X Core parameters
    gamma: float = 1.5
    energy: float = 1.0
    epsilon_0: float = 0.5
    
    # Embeddings (set async)
    identity_emb: np.ndarray = None
    wound_emb: np.ndarray = None
    
    # Trust toward other entities
    trust: Dict[str, float] = field(default_factory=dict)
    
    # Experience ledger
    ledger: ExperienceLedger = None
    
    # Tracking
    last_epsilon: float = 0.0
    last_thought: str = ""
    collision_count: int = 0
    synthesis_count: int = 0
    wound_activations: int = 0
    
    # Visual
    radius: float = ENTITY_RADIUS
    trail: List[Tuple[float, float]] = field(default_factory=list)
    
    def __post_init__(self):
        from pathlib import Path
        import tempfile
        # Create temp storage for in-memory ledger
        temp_dir = Path(tempfile.gettempdir()) / "nexus_ledgers" / self.name
        self.ledger = ExperienceLedger(storage_path=temp_dir, max_entries=50)
    
    @property
    def rho(self) -> float:
        return min(1.0, 0.5 * self.rho_fast + 0.3 * self.rho_slow + 1.0 * self.rho_trauma)
    
    @property
    def k_effective(self) -> float:
        return DDA_PARAMS["k_base"] * (1 - self.rho)
    
    @property
    def will_impedance(self) -> float:
        k_eff = max(0.01, self.k_effective)
        m = max(0.1, self.energy * 0.5)
        return self.gamma / (m * k_eff)
    
    def update_rigidity(self, epsilon: float, wound_activated: bool = False):
        """Update multi-timescale rigidity based on prediction error."""
        self.last_epsilon = epsilon
        
        # Wound resonance boosts epsilon
        if wound_activated:
            epsilon = min(1.0, epsilon + DDA_PARAMS["wound_resonance_boost"])
            self.wound_activations += 1
        
        z = (epsilon - self.epsilon_0) / DDA_PARAMS["s"]
        sig = sigmoid(z)
        
        delta_fast = DDA_PARAMS["alpha_fast"] * (sig - 0.5)
        self.rho_fast = float(np.clip(self.rho_fast + delta_fast, 0.0, 1.0))
        
        delta_slow = DDA_PARAMS["alpha_slow"] * (sig - 0.5)
        self.rho_slow = float(np.clip(self.rho_slow + delta_slow, 0.0, 1.0))
        
        # Trauma - ASYMMETRIC
        if epsilon > DDA_PARAMS["trauma_threshold"]:
            delta_trauma = DDA_PARAMS["alpha_trauma"] * (epsilon - DDA_PARAMS["trauma_threshold"])
            self.rho_trauma = float(np.clip(self.rho_trauma + delta_trauma, 0.0, 1.0))
    
    def update_trust(self, other_name: str, delta: float):
        """Update trust toward another entity."""
        current = self.trust.get(other_name, 0.5)
        self.trust[other_name] = float(np.clip(current + delta, 0.0, 1.0))
    
    def update(self, dt: float):
        k_eff = self.k_effective
        
        self.x += self.vx * dt * 60 * (0.5 + k_eff * 0.5)
        self.y += self.vy * dt * 60 * (0.5 + k_eff * 0.5)
        
        cx, cy = WIDTH / 2, HEIGHT / 2
        dx, dy = cx - self.x, cy - self.y
        dist = max(1, (dx**2 + dy**2)**0.5)
        
        pull = self.gamma * k_eff * 0.02
        self.vx += dx / dist * pull
        self.vy += dy / dist * pull
        
        damping = 0.98 - self.rho * 0.05
        self.vx *= damping
        self.vy *= damping
        
        margin = 50
        if self.x < margin:
            self.x = margin
            self.vx = abs(self.vx) * 0.5
        if self.x > WIDTH - margin:
            self.x = WIDTH - margin
            self.vx = -abs(self.vx) * 0.5
        if self.y < margin:
            self.y = margin
            self.vy = abs(self.vy) * 0.5
        if self.y > HEIGHT - margin:
            self.y = HEIGHT - margin
            self.vy = -abs(self.vy) * 0.5
        
        self.rho_fast = max(0, self.rho_fast - 0.003)
        self.rho_slow = max(0, self.rho_slow - 0.0003)
        
        self.trail.append((self.x, self.y))
        if len(self.trail) > 25:
            self.trail.pop(0)
    
    def draw(self, screen, font, show_dda: bool = False):
        color = SECTOR_COLORS.get(self.sector, (255, 255, 255))
        
        # Trail
        for i, (tx, ty) in enumerate(self.trail):
            alpha = i / len(self.trail) * 0.4
            pygame.draw.circle(screen, color, (int(tx), int(ty)), 2)
        
        size = int(self.radius + self.energy * 0.4)
        outline_width = 1 + int(self.rho * 3)
        
        # Trauma ring
        if self.rho_trauma > 0.005:
            trauma_size = size + 6
            trauma_surf = pygame.Surface((trauma_size * 2 + 10, trauma_size * 2 + 10), pygame.SRCALPHA)
            trauma_alpha = int(200 * self.rho_trauma)
            pygame.draw.circle(trauma_surf, (255, 50, 50, trauma_alpha), 
                             (trauma_size + 5, trauma_size + 5), trauma_size, 3)
            screen.blit(trauma_surf, (int(self.x) - trauma_size - 5, int(self.y) - trauma_size - 5))
        
        # Glow
        glow_alpha = int(40 * (1 - self.rho))
        glow_surf = pygame.Surface((size * 4, size * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, glow_alpha), (size * 2, size * 2), size * 2)
        screen.blit(glow_surf, (int(self.x) - size * 2, int(self.y) - size * 2))
        
        # Entity
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), size)
        pygame.draw.circle(screen, (255, 255, 255), (int(self.x), int(self.y)), size, outline_width)
        
        # Label
        label = font.render(self.name, True, (255, 255, 255))
        screen.blit(label, (int(self.x) - label.get_width() // 2, int(self.y) + size + 2))
        
        # DDA stats
        if show_dda:
            dda_text = f"r={self.rho:.2f} k={self.k_effective:.2f}"
            dda_label = font.render(dda_text, True, (150, 150, 150))
            screen.blit(dda_label, (int(self.x) - dda_label.get_width() // 2, int(self.y) + size + 12))


# =============================================================================
# COLLISION EFFECT
# =============================================================================
@dataclass
class CollisionEffect:
    x: float
    y: float
    color: Tuple[int, int, int]
    text: str
    thought: str = ""
    life: float = 1.0
    
    def update(self, dt: float):
        self.life -= dt * 1.5
        self.y -= 15 * dt
    
    def draw(self, screen, font, thought_font):
        if self.life <= 0:
            return
        alpha = int(255 * self.life)
        
        # Ring
        radius = int(40 * (1 - self.life) + 15)
        surf = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*self.color, alpha), (radius + 2, radius + 2), radius, 2)
        screen.blit(surf, (int(self.x) - radius - 2, int(self.y) - radius - 2))
        
        # Collision text
        label = font.render(self.text, True, self.color)
        label.set_alpha(alpha)
        screen.blit(label, (int(self.x) - label.get_width() // 2, int(self.y) - 35))
        
        # Thought (if any)
        if self.thought and self.life > 0.5:
            thought_label = thought_font.render(f'"{self.thought[:50]}..."', True, (200, 200, 255))
            thought_label.set_alpha(int(alpha * 0.8))
            screen.blit(thought_label, (int(self.x) - thought_label.get_width() // 2, int(self.y) - 55))


# =============================================================================
# COLLISION LOGIC
# =============================================================================
def determine_collision(type_a: EntityType, type_b: EntityType) -> Tuple[str, str]:
    if (type_a == EntityType.NATURE and type_b == EntityType.MACHINE) or \
       (type_a == EntityType.MACHINE and type_b == EntityType.NATURE):
        return "synthesis", "BIOMIMICRY!"
    
    if (type_a == EntityType.STRUCTURE and type_b == EntityType.ENTROPY) or \
       (type_a == EntityType.ENTROPY and type_b == EntityType.STRUCTURE):
        return "decay", "DECAY"
    
    if (type_a == EntityType.ABSTRACT and type_b == EntityType.PHYSICAL) or \
       (type_a == EntityType.PHYSICAL and type_b == EntityType.ABSTRACT):
        return "design", "DESIGN"
    
    if type_a == type_b:
        return "resonance", "RESONANCE"
    
    return "chaos", "CHAOS"


# =============================================================================
# LLM THOUGHT GENERATOR (Async)
# =============================================================================
class ThoughtGenerator:
    """Generates entity thoughts using GPT-4o-mini in background thread."""
    
    def __init__(self):
        # Using gpt-4o-mini - fast and confirmed working
        self.provider = OpenAIProvider(model="gpt-4o-mini", embed_model="text-embedding-3-large")
        self.thought_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        """Background thread to process LLM requests."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self.running:
            try:
                task = self.thought_queue.get(timeout=0.1)
                if task is None:
                    continue
                
                e1_name, e2_name, e1_core, e2_core, ctype = task
                
                prompt = f"""You are {e1_name}, a concept from Da Vinci's mind.
Your core: "{e1_core}"

You just collided with {e2_name} in a {ctype} event.

Express your reaction in ONE short sentence (max 12 words). Be poetic and Da Vinci-like."""

                print(f"[LLM] Requesting thought for {e1_name}...")
                
                # Add timeout with asyncio.wait_for
                async def get_with_timeout():
                    return await asyncio.wait_for(
                        self.provider.complete(prompt=prompt, temperature=0.9),
                        timeout=10.0
                    )
                
                response = loop.run_until_complete(get_with_timeout())
                
                thought = response.strip().strip('"').strip("'")
                print(f"[LLM] Got: {thought[:60]}...")
                self.result_queue.put((e1_name, thought))
                
            except queue.Empty:
                continue
            except asyncio.TimeoutError:
                print(f"[LLM TIMEOUT] Request took too long")
            except Exception as e:
                print(f"[LLM ERROR] {e}")
    
    def request_thought(self, e1, e2, ctype: str):
        """Request a thought generation (non-blocking)."""
        print(f"[QUEUE] Queuing thought for {e1.name} x {e2.name} ({ctype})")
        self.thought_queue.put((e1.name, e2.name, e1.core, e2.core, ctype))
    
    def get_thoughts(self) -> List[Tuple[str, str]]:
        """Get all available thoughts (non-blocking)."""
        thoughts = []
        while True:
            try:
                thoughts.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return thoughts
    
    async def embed_entity(self, entity: Entity):
        """Create embeddings for entity identity and wounds."""
        identity_text = f"{entity.name}: {entity.core}"
        entity.identity_emb = await self.provider.embed(identity_text)
        entity.identity_emb = entity.identity_emb / (np.linalg.norm(entity.identity_emb) + 1e-9)
        
        wound_terms = list(get_wound_lexicon(entity.entity_type.value))
        if wound_terms:
            wound_text = f"Things that wound {entity.name}: {', '.join(wound_terms[:5])}"
            entity.wound_emb = await self.provider.embed(wound_text)
            entity.wound_emb = entity.wound_emb / (np.linalg.norm(entity.wound_emb) + 1e-9)
    
    def stop(self):
        self.running = False


# =============================================================================
# MAIN SIMULATION
# =============================================================================
async def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("THE NEXUS — Da Vinci Matrix [DDA-X COGNITIVE]")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 9)
    title_font = pygame.font.SysFont("Consolas", 22, bold=True)
    info_font = pygame.font.SysFont("Consolas", 12)
    thought_font = pygame.font.SysFont("Georgia", 11, italic=True)
    
    # Initialize thought generator
    print("Initializing GPT-5-nano thought generator...")
    thought_gen = ThoughtGenerator()
    
    # Initialize entities
    print("Loading 50 entities with embeddings...")
    entities: Dict[str, Entity] = {}
    sector_positions = {
        "A": (250, 200),
        "B": (WIDTH // 2, HEIGHT // 2),
        "C": (250, HEIGHT - 200),
        "D": (WIDTH - 300, HEIGHT - 200),
        "E": (WIDTH - 300, 200),
    }
    
    for name, data in ENTITIES_DATA.items():
        cx, cy = sector_positions[data["sector"]]
        entities[name] = Entity(
            name=name,
            sector=data["sector"],
            entity_type=data["type"],
            seeks=data["seeks"],
            core=data["core"],
            x=cx + random.uniform(-120, 120),
            y=cy + random.uniform(-100, 100),
            vx=random.uniform(-1.5, 1.5),
            vy=random.uniform(-1.5, 1.5),
        )
    
    # Create embeddings for all entities
    print("Creating identity embeddings (this may take a moment)...")
    for entity in entities.values():
        await thought_gen.embed_entity(entity)
    print("Embeddings complete. Starting simulation...")
    
    effects: List[CollisionEffect] = []
    collision_count = 0
    synthesis_count = 0
    decay_count = 0
    thoughts_generated = 0
    year = 0
    
    running = True
    paused = False
    show_labels = True
    show_dda = False
    thought_rate_limiter = 0  # Limit LLM calls
    
    # Recent thoughts display
    recent_thoughts: deque = deque(maxlen=5)
    
    while running:
        dt = clock.tick(FPS) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_l:
                    show_labels = not show_labels
                elif event.key == pygame.K_d:
                    show_dda = not show_dda
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get generated thoughts
        for entity_name, thought in thought_gen.get_thoughts():
            if entity_name in entities:
                entities[entity_name].last_thought = thought
                recent_thoughts.append(f"{entity_name}: {thought}")
                print(f"[THOUGHT] {entity_name}: {thought}")
                thoughts_generated += 1
        
        if not paused:
            year += dt * 10
            thought_rate_limiter += dt
            
            # Update entities
            for entity in entities.values():
                entity.update(dt)
            
            # Check collisions
            entity_list = list(entities.values())
            for i, e1 in enumerate(entity_list):
                for e2 in entity_list[i+1:]:
                    dx = e2.x - e1.x
                    dy = e2.y - e1.y
                    dist = (dx**2 + dy**2)**0.5
                    
                    if dist < e1.radius + e2.radius + 12:
                        collision_count += 1
                        e1.collision_count += 1
                        e2.collision_count += 1
                        ctype, ctext = determine_collision(e1.entity_type, e2.entity_type)
                        
                        # Check wound activation via cosine similarity (if embeddings exist)
                        wound_activated = False
                        if e1.identity_emb is not None and e2.wound_emb is not None:
                            sim = float(np.dot(e1.identity_emb, e2.wound_emb))
                            if sim > 0.3:
                                wound_activated = True
                        
                        epsilon = random.uniform(0.3, 0.9)
                        
                        if ctype == "synthesis":
                            epsilon = random.uniform(0.2, 0.5)
                            e1.energy += 0.4
                            e2.energy += 0.4
                            e1.synthesis_count += 1
                            e2.synthesis_count += 1
                            e1.update_trust(e2.name, 0.05)
                            e2.update_trust(e1.name, 0.05)
                            synthesis_count += 1
                            
                        elif ctype == "decay":
                            epsilon = random.uniform(0.7, 1.0)
                            if e1.entity_type == EntityType.STRUCTURE:
                                e1.update_rigidity(epsilon, wound_activated)
                                e1.energy = max(0.5, e1.energy - 0.15)
                                e1.update_trust(e2.name, -0.1)
                            else:
                                e2.update_rigidity(epsilon, wound_activated)
                                e2.energy = max(0.5, e2.energy - 0.15)
                                e2.update_trust(e1.name, -0.1)
                            decay_count += 1
                            
                        elif ctype == "design":
                            epsilon = random.uniform(0.4, 0.6)
                            e1.energy += 0.25
                            e2.energy += 0.25
                            e1.update_rigidity(epsilon)
                            e2.update_rigidity(epsilon)
                            
                        elif ctype == "resonance":
                            epsilon = random.uniform(0.1, 0.3)
                            e1.energy += 0.1
                            e2.energy += 0.1
                            e1.update_rigidity(epsilon)
                            e2.update_rigidity(epsilon)
                            e1.update_trust(e2.name, 0.02)
                            e2.update_trust(e1.name, 0.02)
                            
                        else:
                            epsilon = random.uniform(0.4, 0.8)
                            e1.update_rigidity(epsilon)
                            e2.update_rigidity(epsilon)
                        
                        # Log to experience ledger (proper DDA-X)
                        import time
                        if e1.identity_emb is not None:
                            e1.ledger.add_entry(LedgerEntry(
                                timestamp=time.time(),
                                state_vector=np.array([e1.x, e1.y, e1.rho, e1.energy]),
                                action_id=f"{ctype}_collision",
                                observation_embedding=e2.identity_emb if e2.identity_emb is not None else np.zeros(3072),
                                outcome_embedding=e1.identity_emb,
                                prediction_error=epsilon,
                                context_embedding=e1.identity_emb,
                                task_id=f"collision_{e2.name}",
                                rigidity_at_time=e1.rho,
                                was_successful=(ctype in ["synthesis", "design", "resonance"]),
                                metadata={"partner": e2.name, "year": int(year), "type": ctype}
                            ))
                        
                        # Request thought (rate limited to ~5/sec)
                        if thought_rate_limiter > 0.2 and ctype in ["synthesis", "decay", "design"]:
                            thought_gen.request_thought(e1, e2, ctype)
                            thought_rate_limiter = 0
                        
                        # Repulsion
                        if dist > 0:
                            nx, ny = dx / dist, dy / dist
                            push = 2.5 * ((e1.k_effective + e2.k_effective) / 2 + 0.3)
                            e1.vx -= nx * push
                            e1.vy -= ny * push
                            e2.vx += nx * push
                            e2.vy += ny * push
                        
                        mx, my = (e1.x + e2.x) / 2, (e1.y + e2.y) / 2
                        effects.append(CollisionEffect(
                            x=mx, y=my,
                            color=COLLISION_COLORS.get(ctype, (255, 255, 255)),
                            text=f"{e1.name} x {e2.name}: {ctext}",
                            thought=e1.last_thought if e1.last_thought else ""
                        ))
            
            # Update effects
            for effect in effects[:]:
                effect.update(dt)
                if effect.life <= 0:
                    effects.remove(effect)
        
        # Draw
        screen.fill(BG_COLOR)
        
        # Sector labels
        for sector, (sx, sy) in sector_positions.items():
            label = info_font.render(f"SECTOR {sector}", True, SECTOR_COLORS[sector])
            screen.blit(label, (sx - label.get_width() // 2, 40))
        
        # Entities
        for entity in entities.values():
            entity.draw(screen, font if show_labels else pygame.font.SysFont("Consolas", 1), show_dda)
        
        # Effects
        for effect in effects:
            effect.draw(screen, info_font, thought_font)
        
        # HUD
        title = title_font.render("THE NEXUS — DDA-X Cognitive Simulation", True, (255, 255, 255))
        screen.blit(title, (10, 8))
        
        mean_rho = sum(e.rho for e in entities.values()) / len(entities)
        mean_k = sum(e.k_effective for e in entities.values()) / len(entities)
        traumatized = sum(1 for e in entities.values() if e.rho_trauma > 0.01)
        total_wounds = sum(e.wound_activations for e in entities.values())
        
        stats = [
            f"Year: {int(year)} | Collisions: {collision_count}",
            f"Syntheses: {synthesis_count} | Decays: {decay_count}",
            f"",
            f"--- DDA-X DYNAMICS ---",
            f"Mean rho: {mean_rho:.3f}",
            f"Mean k_eff: {mean_k:.3f}",
            f"Traumatized: {traumatized}/50",
            f"Wound Activations: {total_wounds}",
            f"Thoughts Generated: {thoughts_generated}",
            f"",
            f"[SPACE] Pause [L] Labels",
            f"[D] DDA Stats [ESC] Quit",
        ]
        for i, stat in enumerate(stats):
            color = (100, 200, 255) if "DDA-X" in stat else (180, 180, 180)
            label = info_font.render(stat, True, color)
            screen.blit(label, (10, 40 + i * 15))
        
        # Right side: Rankings
        right_x = WIDTH - 180
        sorted_by_rho = sorted(entities.values(), key=lambda e: e.rho, reverse=True)[:5]
        sorted_by_energy = sorted(entities.values(), key=lambda e: e.energy, reverse=True)[:5]
        
        screen.blit(info_font.render("MOST RIGID:", True, (255, 100, 100)), (right_x, 45))
        for i, e in enumerate(sorted_by_rho):
            txt = f"{e.name}: {e.rho:.2f}"
            lbl = font.render(txt, True, (255, 150, 150))
            screen.blit(lbl, (right_x, 62 + i * 13))
        
        screen.blit(info_font.render("MOST ENERGY:", True, (100, 255, 100)), (right_x, 135))
        for i, e in enumerate(sorted_by_energy):
            txt = f"{e.name}: {e.energy:.1f}"
            lbl = font.render(txt, True, (150, 255, 150))
            screen.blit(lbl, (right_x, 152 + i * 13))
        
        # Recent thoughts
        screen.blit(info_font.render("RECENT THOUGHTS:", True, (200, 200, 255)), (right_x - 50, 230))
        for i, thought in enumerate(recent_thoughts):
            lbl = font.render(thought[:40] + "..." if len(thought) > 40 else thought, True, (180, 180, 220))
            screen.blit(lbl, (right_x - 50, 248 + i * 12))
        
        if paused:
            pause_label = title_font.render("PAUSED", True, (255, 255, 0))
            screen.blit(pause_label, (WIDTH // 2 - pause_label.get_width() // 2, HEIGHT // 2))
        
        pygame.display.flip()
    
    thought_gen.stop()
    pygame.quit()


if __name__ == "__main__":
    asyncio.run(main())
