#!/usr/bin/env python3
"""
SOLVE COLLATZ - SERIOUS MATHEMATICAL PROOF ATTEMPT
===================================================

A society of elite mathematical agents rigorously collaborate to prove
the Collatz Conjecture (3n+1 problem) in ≤20 collective steps.

Infrastructure:
- Brain: LM Studio (gpt-oss-20b or better)
- Guts: DDAState + MultiTimescaleRigidity + TrustMatrix
- Memory: Ollama embeddings (nomic-embed-text, 768-dim)
- Tools: SymPy code execution, simulated web search for known results

Key Design:
- Very low starting rigidity (ρ≈0.05-0.15) for maximum creative openness
- High initial trust for genuine collaboration
- Low temperature (0.3-0.6) for mathematical precision
- Tool integration: agents can propose code/search, we execute and feed back

Agent Personas:
1. Euler (Induction Master): Structural induction, base cases
2. Gauss (Number Theory Legend): Modular arithmetic, cycles, invariants
3. Ramanujan (Pattern Genius): Intuition, generating functions, leaps
4. Hilbert (Formal Rigor): Logic, halting, complete formalization
5. Noether (Symmetry Expert): Transformations, tree structure, invariants
6. Tao (Modern Synthesizer): Probabilistic methods, density, bridges
"""

import asyncio
import sys
import os
import re
import numpy as np
import time
import random
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from io import StringIO
import contextlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.core.dynamics import MultiTimescaleRigidity
from src.memory.ledger import ExperienceLedger, LedgerEntry
from src.society.trust import TrustMatrix
from src.llm.openai_provider import OpenAIProvider

# Try to import sympy for mathematical tool execution
try:
    import sympy
    from sympy import symbols, simplify, factor, solve, Mod, floor, ceiling, log, oo
    from sympy import Integer, Rational, sqrt, gcd, lcm, divisors, isprime
    from sympy import Sum, Product, binomial, factorial, fibonacci
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("[Warning] SymPy not available. Code execution will be limited.")


# ═══════════════════════════════════════════════════════════
# TERMINAL COLORS
# ═══════════════════════════════════════════════════════════

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    
    # Agent-specific colors
    EULER = "\033[94m"       # Blue - methodical
    GAUSS = "\033[93m"       # Yellow - brilliant
    RAMANUJAN = "\033[95m"   # Magenta - intuitive
    HILBERT = "\033[92m"     # Green - rigorous
    NOETHER = "\033[96m"     # Cyan - structural
    TAO = "\033[91m"         # Red - modern
    
    # Status colors
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    WHITE = "\033[97m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"


# ═══════════════════════════════════════════════════════════
# KNOWN RESULTS DATABASE (Simulated Web Search)
# ═══════════════════════════════════════════════════════════

KNOWN_RESULTS = {
    "terras": """
Terras (1976): For almost all positive integers (in the sense of natural density), 
the Collatz sequence eventually reaches a value smaller than the starting value. 
More precisely, for any ε > 0, the set of n for which all iterates stay ≥ n has 
density 0.
""",
    
    "lagarias": """
Lagarias Survey (2010): Key known results:
1. No non-trivial cycles exist with period < 10^17.
2. All n < 2^68 ≈ 2.95×10^20 reach 1.
3. The problem is Π₂-complete: undecidable extensions exist.
4. Lower bounds on cycle lengths: any cycle must have ≥ 186 billion elements.
5. The 3n+1 problem is equivalent to a reachability problem in certain automata.
""",
    
    "tao": """
Tao (2019): "Almost all Collatz orbits attain almost bounded values"
- For any function f(n) → ∞, almost all n have max orbit value < f(n)·n.
- Uses entropy/ergodic methods on the logarithmic dynamics.
- The "2/3 vs 3/2" heuristic: expected log change per step ≈ log(3/4) < 0, 
  suggesting almost-sure descent to 1.
- Key insight: probabilistic independence approximation for residue classes.
""",
    
    "krasikov": """
Krasikov-Lagarias (2003): Lower bounds on counterexamples.
If a non-trivial cycle exists with minimal element m, then m > 10^10.
Refined computational searches have pushed this further.
""",
    
    "no_cycles": """
Cycle Analysis:
1. The only cycle containing 1 is: 1 → 4 → 2 → 1.
2. Any other cycle (if exists) must avoid even numbers staying even.
3. For odd n in a cycle: n → 3n+1 → (3n+1)/2 (since 3n+1 even).
4. Net transformation for odd: n → (3n+1)/2^k for some k≥1.
5. Cycle equation: n = ((3^a × n + Σ) / 2^b) for integers a, b, with Σ algebraic.
6. Proved: No cycles with period ≤ 10^17 (computational verification).
""",

    "tree_structure": """
Collatz Tree Structure:
- Inverse map: every n has predecessor 2n (always).
- If n ≡ 1 (mod 3), then (n-1)/3 is also a predecessor (if positive odd).
- Tree rooted at 1, with infinite branching.
- All positive integers appear exactly once ⟺ Collatz is true.
- Key: Every integer must have a path TO 1 (forward) ⟺ be reachable FROM 1 (backward).
""",

    "modular_analysis": """
Modular Arithmetic Analysis:
- n mod 2: determines operation (odd → 3n+1, even → n/2).
- n mod 3: 3n+1 mod 3 = (n+1) mod 3. Cycles through residues.
- n mod 6: More refined analysis of trajectory behavior.
- Parity vectors: encoding sequences of odd/even steps.
- 2-adic analysis: studying limiting behavior in Z_2.
"""
}


# ═══════════════════════════════════════════════════════════
# TOOL EXECUTION ENGINE
# ═══════════════════════════════════════════════════════════

class ToolEngine:
    """Execute code and search tools proposed by agents."""
    
    def __init__(self):
        self.execution_history: List[Dict] = []
    
    def extract_tool_calls(self, message: str) -> List[Dict]:
        """
        Extract tool calls from agent message.
        Formats:
        - ```python\n...\n``` or ```sympy\n...\n```
        - [SEARCH: query] or [LOOKUP: topic]
        - [COMPUTE: expression]
        """
        calls = []
        
        # Extract code blocks
        code_pattern = r'```(?:python|sympy)?\s*\n(.*?)```'
        for match in re.finditer(code_pattern, message, re.DOTALL | re.IGNORECASE):
            calls.append({"type": "code", "content": match.group(1).strip()})
        
        # Extract search queries
        search_pattern = r'\[(?:SEARCH|LOOKUP|QUERY):\s*([^\]]+)\]'
        for match in re.finditer(search_pattern, message, re.IGNORECASE):
            calls.append({"type": "search", "query": match.group(1).strip()})
        
        # Extract compute expressions
        compute_pattern = r'\[COMPUTE:\s*([^\]]+)\]'
        for match in re.finditer(compute_pattern, message, re.IGNORECASE):
            calls.append({"type": "compute", "expr": match.group(1).strip()})
        
        return calls
    
    def execute_code(self, code: str) -> str:
        """Safely execute mathematical code using SymPy."""
        if not SYMPY_AVAILABLE:
            return "[Code execution unavailable: SymPy not installed]"
        
        # Create safe execution environment
        safe_globals = {
            "sympy": sympy,
            "symbols": symbols,
            "simplify": simplify,
            "factor": factor,
            "solve": solve,
            "Mod": Mod,
            "floor": floor,
            "ceiling": ceiling,
            "log": log,
            "Integer": Integer,
            "Rational": Rational,
            "sqrt": sqrt,
            "gcd": gcd,
            "lcm": lcm,
            "divisors": divisors,
            "isprime": isprime,
            "Sum": Sum,
            "Product": Product,
            "binomial": binomial,
            "factorial": factorial,
            "fibonacci": fibonacci,
            "oo": oo,
            "range": range,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "print": print,
            "__builtins__": {},
        }
        
        # Capture output
        output = StringIO()
        result = None
        
        try:
            with contextlib.redirect_stdout(output):
                exec(code, safe_globals, safe_globals)
                # Try to get last expression value
                lines = code.strip().split('\n')
                last_line = lines[-1].strip()
                if last_line and not any(last_line.startswith(kw) for kw in ['if', 'for', 'while', 'def', 'class', 'import', 'from', 'print', '#']):
                    try:
                        result = eval(last_line, safe_globals, safe_globals)
                    except:
                        pass
            
            printed = output.getvalue()
            if result is not None:
                return f"Result: {result}" + (f"\nOutput: {printed}" if printed else "")
            elif printed:
                return f"Output: {printed.strip()}"
            else:
                return "[Code executed successfully, no output]"
                
        except Exception as e:
            return f"[Execution Error: {type(e).__name__}: {e}]"
    
    def search_known_results(self, query: str) -> str:
        """Search the known results database."""
        query_lower = query.lower()
        
        best_match = None
        best_score = 0
        
        for key, content in KNOWN_RESULTS.items():
            # Simple keyword matching
            score = sum(1 for word in query_lower.split() if word in key or word in content.lower())
            if score > best_score:
                best_score = score
                best_match = content
        
        if best_match:
            return f"[Known Result Found]:\n{best_match.strip()}"
        else:
            return f"[No specific result found for '{query}'. Try: terras, lagarias, tao, cycles, tree, modular]"
    
    def compute_expression(self, expr: str) -> str:
        """Evaluate a mathematical expression."""
        if not SYMPY_AVAILABLE:
            try:
                return f"Result: {eval(expr)}"
            except:
                return "[Compute unavailable without SymPy]"
        
        try:
            result = sympy.sympify(expr)
            simplified = sympy.simplify(result)
            return f"Result: {simplified}"
        except Exception as e:
            return f"[Compute Error: {e}]"
    
    def execute_tools(self, message: str) -> Optional[str]:
        """Execute all tool calls in a message and return combined results."""
        calls = self.extract_tool_calls(message)
        
        if not calls:
            return None
        
        results = []
        for call in calls:
            if call["type"] == "code":
                result = self.execute_code(call["content"])
                results.append(f"📟 Code Execution:\n{result}")
                self.execution_history.append({"type": "code", "result": result})
            
            elif call["type"] == "search":
                result = self.search_known_results(call["query"])
                results.append(f"🔍 Search: {call['query']}\n{result}")
                self.execution_history.append({"type": "search", "query": call["query"], "result": result})
            
            elif call["type"] == "compute":
                result = self.compute_expression(call["expr"])
                results.append(f"🧮 Compute: {call['expr']}\n{result}")
                self.execution_history.append({"type": "compute", "expr": call["expr"], "result": result})
        
        return "\n\n".join(results) if results else None


# ═══════════════════════════════════════════════════════════
# MATHEMATICIAN AGENT CONFIGURATIONS
# ═══════════════════════════════════════════════════════════

MATHEMATICIANS = {
    "EULER": {
        "name": "Euler",
        "color": C.EULER,
        "identity": {
            "core": "Build from the ground up. Induction is the ladder to infinity.",
            "persona": "Master of structural induction and infinite processes. Methodical, exhaustive, patient. Believes every proof begins with a solid base case.",
            "interests": ["induction", "base cases", "infinite series", "recursion", "well-ordering"],
            "tools": "I may propose induction schemas or recursive computations.",
        },
        "dda_params": {
            "gamma": 1.5,       # Strong identity
            "epsilon_0": 0.40,  # Tolerant
            "alpha": 0.06,      # Slow adaptation (stable)
            "rho": 0.08         # Very low - highly open
        },
        "extraversion": 0.60,
        "reactivity": 0.65,
    },
    
    "GAUSS": {
        "name": "Gauss",
        "color": C.GAUSS,
        "identity": {
            "core": "Find the conserved quantity. Every dynamic has a hidden invariant.",
            "persona": "Prince of Number Theory. Seeks modular structure, cycle analysis, and algebraic invariants. Penetrating, often solitary insights.",
            "interests": ["modular arithmetic", "invariants", "cycles", "residue classes", "density"],
            "tools": "I compute modular properties and search for invariants.",
        },
        "dda_params": {
            "gamma": 1.7,       # Strong identity
            "epsilon_0": 0.35,  # Moderate threshold
            "alpha": 0.07,
            "rho": 0.10         # Very open
        },
        "extraversion": 0.50,
        "reactivity": 0.60,
    },
    
    "RAMANUJAN": {
        "name": "Ramanujan",
        "color": C.RAMANUJAN,
        "identity": {
            "core": "See the hidden harmony in numbers. Patterns speak before proofs.",
            "persona": "Intuitive genius. Proposes bold conjectures, generating functions, identities. Often leaps ahead of formal justification.",
            "interests": ["patterns", "generating functions", "identities", "continued fractions", "q-series"],
            "tools": "I propose pattern-based conjectures and series representations.",
        },
        "dda_params": {
            "gamma": 1.2,       # Flexible identity (creative)
            "epsilon_0": 0.50,  # High tolerance for surprise
            "alpha": 0.10,      # Faster adaptation
            "rho": 0.05         # Most open (pure intuition)
        },
        "extraversion": 0.75,
        "reactivity": 0.85,
    },
    
    "HILBERT": {
        "name": "Hilbert",
        "color": C.HILBERT,
        "identity": {
            "core": "Every step must be airtight. Formalize, axiomatize, verify.",
            "persona": "Supreme formalist. Demands rigorous definitions, complete axiom systems, halting analysis. The conscience of the proof.",
            "interests": ["formalism", "axioms", "decidability", "halting", "complete systems"],
            "tools": "I formalize claims into precise logical statements and check for gaps.",
        },
        "dda_params": {
            "gamma": 2.0,       # Strongest identity (principled)
            "epsilon_0": 0.28,  # Low threshold (rigorous)
            "alpha": 0.05,      # Very slow change
            "rho": 0.15         # Slightly higher baseline (principled skepticism)
        },
        "extraversion": 0.45,
        "reactivity": 0.55,
    },
    
    "NOETHER": {
        "name": "Noether",
        "color": C.NOETHER,
        "identity": {
            "core": "Uncover the symmetry that forces convergence. Structure is destiny.",
            "persona": "Abstract algebraist. Sees transformations, tree structures, invariants under group actions. Elegant, structural thinking.",
            "interests": ["symmetry", "transformations", "tree structure", "group actions", "abstraction"],
            "tools": "I analyze the Collatz tree and transformation structure.",
        },
        "dda_params": {
            "gamma": 1.5,
            "epsilon_0": 0.42,
            "alpha": 0.08,
            "rho": 0.08         # Very open
        },
        "extraversion": 0.55,
        "reactivity": 0.65,
    },
    
    "TAO": {
        "name": "Tao",
        "color": C.TAO,
        "identity": {
            "core": "Combine rigorous partials into global convergence. Bridge all methods.",
            "persona": "Modern polymath. Uses probabilistic arguments, density estimates, computational verification. Synthesizes disparate approaches.",
            "interests": ["probability", "density arguments", "synthesis", "almost-all results", "modern methods"],
            "tools": "I bridge probabilistic and deterministic methods, synthesize partial results.",
        },
        "dda_params": {
            "gamma": 1.4,
            "epsilon_0": 0.45,
            "alpha": 0.09,
            "rho": 0.07         # Very open (synthesizer)
        },
        "extraversion": 0.70,
        "reactivity": 0.75,
    },
}


# ═══════════════════════════════════════════════════════════
# SEED MESSAGE
# ═══════════════════════════════════════════════════════════

COLLATZ_SEED = """Team, the Collatz Conjecture: For any positive integer n, repeatedly apply:
  • n → 3n + 1  (if n is odd)
  • n → n / 2   (if n is even)

The claim: every starting n eventually reaches 1.

Known facts we should use:
1. Verified computationally for all n < 10^20.
2. Terras (1976): Almost all orbits descend below their starting point.
3. Tao (2019): Almost all orbits attain almost bounded values (probabilistic descent).
4. No non-trivial cycles exist with fewer than 10^17 elements.
5. The Collatz tree (inverse map) is fully connected—every n is reachable from 1 backward.

With our combined expertise—induction, number theory, pattern recognition, formalism, symmetry, and synthesis—let's construct a complete proof. We have 20 messages. Who has an opening insight?"""


# ═══════════════════════════════════════════════════════════
# AGENT DATACLASS
# ═══════════════════════════════════════════════════════════

@dataclass
class MathAgent:
    """Mathematical agent with full DDA-X dynamics."""
    id: str
    name: str
    color: str
    config: Dict
    dda_state: DDAState
    rigidity: MultiTimescaleRigidity
    ledger: ExperienceLedger
    identity_embedding: np.ndarray
    extraversion: float
    reactivity: float
    last_spoke: float = 0.0
    interaction_count: int = 0
    contributions: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# COLLATZ SOLVER SIMULATION
# ═══════════════════════════════════════════════════════════

class CollatzSolverSimulation:
    """
    Serious mathematical proof simulation with tool integration.
    
    Features:
    - Very low rigidity for maximum openness
    - Tool execution (code, search)
    - Rigorous mathematical collaboration
    - Proof assembly at end
    """
    
    def __init__(self):
        self.provider = OpenAIProvider(
            model="gpt-5.2",
            embed_model="text-embedding-3-large"
        )
        
        self.agents: Dict[str, MathAgent] = {}
        self.agent_ids = list(MATHEMATICIANS.keys())
        self.agent_id_to_idx = {aid: i for i, aid in enumerate(self.agent_ids)}
        self.trust_matrix = TrustMatrix(len(MATHEMATICIANS))
        
        self.tool_engine = ToolEngine()
        
        self.conversation: List[Dict] = []
        self.proof_elements: List[Dict] = []
        self.lemmas: List[Dict] = []  # Track proposed lemmas
        self.embed_dim = 3072
        
        self.proof_complete = False
        self.completion_step = -1
        
        self.experiment_dir = Path("data/collatz_solver")
        if self.experiment_dir.exists():
            shutil.rmtree(self.experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize_agent(self, agent_id: str, config: Dict) -> MathAgent:
        """Initialize mathematician with DDA-X state."""
        name = config["name"]
        
        # Rich identity embedding
        identity_text = f"{config['identity']['core']} {config['identity']['persona']} {' '.join(config['identity']['interests'])}"
        identity_emb = await self.provider.embed(identity_text)
        identity_emb = identity_emb / (np.linalg.norm(identity_emb) + 1e-9)
        self.embed_dim = len(identity_emb)
        
        params = config["dda_params"]
        dda_state = DDAState(
            x=identity_emb.copy(),
            x_star=identity_emb.copy(),
            gamma=params["gamma"],
            epsilon_0=params["epsilon_0"],
            alpha=params["alpha"],
            s=0.1,
            rho=params["rho"],
            x_pred=identity_emb.copy()
        )
        
        ledger_path = self.experiment_dir / f"{agent_id}_ledger"
        ledger = ExperienceLedger(
            storage_path=ledger_path,
            lambda_recency=0.003,  # Longer memory
            lambda_salience=2.5    # Strong salience weighting
        )
        
        rigidity = MultiTimescaleRigidity()
        rigidity.rho_fast = params["rho"]
        rigidity.rho_effective = params["rho"]
        
        return MathAgent(
            id=agent_id,
            name=name,
            color=config["color"],
            config=config,
            dda_state=dda_state,
            rigidity=rigidity,
            ledger=ledger,
            identity_embedding=identity_emb,
            extraversion=config["extraversion"],
            reactivity=config["reactivity"],
        )
    
    async def setup(self):
        """Initialize all agents with high mutual trust."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  COLLATZ CONJECTURE SOLVER - SERIOUS PROOF ATTEMPT{C.RESET}")
        print(f"{C.BOLD}  DDA-X Elite Mathematical Society with Tool Integration{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        print(f"\n{C.WHITE}Initializing Mathematicians...{C.RESET}\n")
        
        for agent_id, config in MATHEMATICIANS.items():
            self.agents[agent_id] = await self.initialize_agent(agent_id, config)
            agent = self.agents[agent_id]
            params = config["dda_params"]
            print(f"  {agent.color}●{C.RESET} {agent.name:12} | γ={params['gamma']:.1f} ρ₀={params['rho']:.2f} | {config['identity']['core'][:45]}...")
        
        # High initial trust (elite collaborators)
        print(f"\n{C.DIM}Setting high mutual trust (collaborative team)...{C.RESET}")
        for i in range(len(self.agent_ids)):
            for j in range(len(self.agent_ids)):
                if i != j:
                    self.trust_matrix._trust[i, j] = 0.75
        
        print(f"\n{C.GREEN}✓ Team initialized. Tool engine ready.{C.RESET}")
        if SYMPY_AVAILABLE:
            print(f"{C.GREEN}✓ SymPy available for symbolic computation.{C.RESET}")
        else:
            print(f"{C.YELLOW}⚠ SymPy not available. Code execution limited.{C.RESET}")
    
    def calculate_response_probability(
        self, 
        agent: MathAgent, 
        msg_embedding: np.ndarray, 
        speaker_id: Optional[str],
        current_step: int
    ) -> float:
        """Calculate response probability favoring low-rigidity, high-relevance agents."""
        
        # Relevance to current discussion
        relevance = np.dot(msg_embedding, agent.identity_embedding)
        relevance = max(0.15, (relevance + 1) / 2)
        
        # Openness bonus: lower ρ → much higher probability
        openness = 1.0 - (agent.dda_state.rho * 0.6)
        
        # Trust factor
        trust_boost = 1.0
        if speaker_id and speaker_id in self.agent_id_to_idx:
            spk_idx = self.agent_id_to_idx[speaker_id]
            obs_idx = self.agent_id_to_idx[agent.id]
            trust = self.trust_matrix.get_trust(obs_idx, spk_idx)
            trust_boost = 0.85 + (trust * 0.3)
        
        # Cooldown
        time_since = current_step - agent.last_spoke
        cooldown = min(1.0, time_since / 1.5)
        
        # Base + factors
        prob = agent.extraversion * relevance * openness * trust_boost * cooldown
        
        return np.clip(prob, 0.08, 0.92)
    
    def build_context(self, max_messages: int = 10) -> str:
        """Build proof discussion context."""
        recent = self.conversation[-max_messages:]
        lines = []
        for msg in recent:
            sender = msg.get("sender", "Unknown")
            text = msg.get("text", "")
            # Truncate very long messages
            if len(text) > 300:
                text = text[:300] + "..."
            lines.append(f"{sender}: {text}")
        return "\n".join(lines) if lines else "[Starting discussion]"
    
    def build_lemma_summary(self) -> str:
        """Summarize proposed lemmas."""
        if not self.lemmas:
            return ""
        
        summary = "\n\nPROPOSED LEMMAS SO FAR:\n"
        for i, lemma in enumerate(self.lemmas, 1):
            summary += f"  L{i}. [{lemma['proposer']}] {lemma['statement'][:100]}...\n"
        return summary
    
    def build_system_prompt(self, agent: MathAgent) -> str:
        """Build rigorous mathematical system prompt."""
        identity = agent.config["identity"]
        
        return f"""You are {agent.name}, a world-class mathematician attempting to prove the Collatz Conjecture.

YOUR CORE: {identity['core']}
YOUR STYLE: {identity['persona']}
EXPERTISE: {', '.join(identity['interests'])}
TOOLS: {identity['tools']}

COLLABORATION PROTOCOL:
1. Build on colleagues' insights. Cite by name: "As Gauss noted..." or "Extending Euler's induction..."
2. Propose SPECIFIC claims:
   - "LEMMA: [statement]"
   - "CLAIM: [statement]"
   - "OBSERVATION: [insight]"
3. Use tools when helpful:
   - Code: ```python\n[sympy code]\n```
   - Search: [SEARCH: topic] or [LOOKUP: result name]
   - Compute: [COMPUTE: expression]
4. Identify gaps: "GAP: We need to show..."
5. Be rigorous but creative. This is an open problem.
6. If you believe we have a complete proof, state: "PROOF COMPLETE: [summary]"

Keep responses focused (3-5 sentences or one clear mathematical statement). Rigor over length."""
    
    async def generate_response(self, agent: MathAgent, trigger_msg: Dict) -> str:
        """Generate rigorous mathematical contribution."""
        context = self.build_context()
        lemma_summary = self.build_lemma_summary()
        system_prompt = self.build_system_prompt(agent)
        
        prompt = f"""THE COLLATZ CONJECTURE: ∀n∈ℤ⁺, iterating T(n) = {{3n+1 if odd, n/2 if even}} eventually reaches 1.

DISCUSSION:{lemma_summary}
{context}

Latest from {trigger_msg['sender']}: "{trigger_msg['text'][:200]}..."

As {agent.name}, provide your next rigorous insight, lemma, or proof step:"""

        # Very low temperature for rigor, slightly modulated by openness
        temperature = 0.3 + 0.2 * (1 - agent.dda_state.rho)
        
        response = ""
        try:
            response = await self.provider.complete_with_rigidity(
                prompt,
                rigidity=agent.dda_state.rho,
                system_prompt=system_prompt,
                max_tokens=4096  # Increase token budget for reasoning
            )
            print(f"{C.DIM}[Debug] Raw response length: {len(response) if response else 0}{C.RESET}")
        except Exception as e:
            print(f"{C.DIM}[Generation error: {e}]{C.RESET}")
            response = "Let me reconsider the structural constraints here."
        
        return response.strip()
    
    def extract_lemmas(self, agent_name: str, response: str):
        """Extract and store any lemmas proposed."""
        lemma_patterns = [
            r'LEMMA:\s*(.+?)(?:\n|$)',
            r'CLAIM:\s*(.+?)(?:\n|$)',
            r'PROPOSITION:\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in lemma_patterns:
            for match in re.finditer(pattern, response, re.IGNORECASE):
                self.lemmas.append({
                    "proposer": agent_name,
                    "statement": match.group(1).strip(),
                    "step": len(self.conversation)
                })
    
    async def process_agent_response(
        self, 
        agent: MathAgent, 
        response: str, 
        current_step: int
    ) -> Tuple[float, float]:
        """Process response with DDA dynamics."""
        
        try:
            resp_emb = await self.provider.embed(response)
            resp_emb = resp_emb / (np.linalg.norm(resp_emb) + 1e-9)
        except:
            resp_emb = agent.dda_state.x_pred.copy()
        
        epsilon = np.linalg.norm(agent.dda_state.x_pred - resp_emb)
        rho_before = agent.dda_state.rho
        
        agent.dda_state.update_rigidity(epsilon)
        agent.rigidity.update(epsilon)
        
        # Slower prediction update for stability
        agent.dda_state.x_pred = 0.75 * agent.dda_state.x_pred + 0.25 * resp_emb
        
        agent.last_spoke = current_step
        agent.interaction_count += 1
        agent.contributions.append(response)
        
        # Ledger entry
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=agent.dda_state.x.copy(),
            action_id="proof_contribution",
            observation_embedding=resp_emb,
            outcome_embedding=resp_emb,
            prediction_error=epsilon,
            context_embedding=resp_emb,
            metadata={"text": response[:200], "step": current_step}
        )
        agent.ledger.add_entry(entry)
        
        return epsilon, rho_before
    
    def update_trust(self, speaker: MathAgent, response_emb: np.ndarray):
        """Update trust based on predictability/alignment."""
        spk_idx = self.agent_id_to_idx[speaker.id]
        
        for obs_id, observer in self.agents.items():
            if obs_id == speaker.id:
                continue
            
            obs_idx = self.agent_id_to_idx[obs_id]
            alignment = np.dot(response_emb, observer.identity_embedding)
            self.trust_matrix.update_trust(obs_idx, spk_idx, max(0, 1 - alignment))
    
    def check_proof_complete(self, response: str) -> bool:
        """Check for proof completion signals."""
        signals = [
            "proof complete",
            "we have proven",
            "this completes the proof",
            "the conjecture is proved",
            "q.e.d.",
            "proof established",
            "we have a complete argument"
        ]
        response_lower = response.lower()
        return any(s in response_lower for s in signals)
    
    async def run(self, max_steps: int = 20, max_responders: int = 2):
        """Run the serious proof attempt."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  PROOF SESSION - Maximum {max_steps} Collective Messages{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        print(f"\n{C.WHITE}{C.BOLD}[SEED MESSAGE]{C.RESET}")
        print(f"{C.DIM}{COLLATZ_SEED}{C.RESET}")
        
        seed_emb = await self.provider.embed(COLLATZ_SEED)
        seed_emb = seed_emb / (np.linalg.norm(seed_emb) + 1e-9)
        
        current_msg = {
            "sender": "Moderator",
            "agent_id": None,
            "text": COLLATZ_SEED,
            "emb": seed_emb
        }
        self.conversation.append(current_msg)
        
        print(f"\n{C.BOLD}{'─'*70}{C.RESET}")
        print(f"{C.BOLD}  MATHEMATICAL DISCUSSION{C.RESET}")
        print(f"{C.BOLD}{'─'*70}{C.RESET}\n")
        
        step = 0
        message_count = 0
        
        while message_count < max_steps and not self.proof_complete:
            step += 1
            
            # Calculate probabilities
            candidates = []
            probs = {}
            
            for agent_id, agent in self.agents.items():
                if agent.id == current_msg.get("agent_id"):
                    continue
                
                prob = self.calculate_response_probability(
                    agent, current_msg["emb"], current_msg.get("agent_id"), step
                )
                probs[agent_id] = prob
                
                if random.random() < prob:
                    candidates.append(agent)
            
            if not candidates:
                eligible = [a for a in self.agents.values() if a.id != current_msg.get("agent_id")]
                candidates = [max(eligible, key=lambda a: probs.get(a.id, 0.5))]
            
            candidates.sort(key=lambda a: probs.get(a.id, 0), reverse=True)
            responders = candidates[:max_responders]
            
            for speaker in responders:
                if self.proof_complete or message_count >= max_steps:
                    break
                
                print(f"{C.DIM}→ {speaker.name} is thinking...{C.RESET}")
                response = await self.generate_response(speaker, current_msg)
                
                
                if not response or len(response.strip()) < 5:
                    print(f"{C.RED}[Warning] Response too short/empty: '{response}'{C.RESET}")
                    continue
                
                message_count += 1
                
                # Extract lemmas
                self.extract_lemmas(speaker.name, response)
                
                # Process DDA dynamics
                epsilon, rho_before = await self.process_agent_response(speaker, response, step)
                delta_rho = speaker.dda_state.rho - rho_before
                
                # Execute any tools
                tool_results = self.tool_engine.execute_tools(response)
                
                # Embed for next iteration
                try:
                    response_emb = await self.provider.embed(response)
                    response_emb = response_emb / (np.linalg.norm(response_emb) + 1e-9)
                except:
                    response_emb = speaker.identity_embedding.copy()
                
                self.update_trust(speaker, response_emb)
                
                current_msg = {
                    "sender": speaker.name,
                    "agent_id": speaker.id,
                    "text": response,
                    "emb": response_emb
                }
                self.conversation.append(current_msg)
                
                self.proof_elements.append({
                    "step": message_count,
                    "agent": speaker.name,
                    "contribution": response,
                    "epsilon": epsilon,
                    "rho": speaker.dda_state.rho,
                    "tool_results": tool_results
                })
                
                # Display
                rho_color = C.RED if delta_rho > 0.005 else C.GREEN if delta_rho < -0.005 else C.DIM
                
                print(f"{C.BOLD}[{message_count}/{max_steps}]{C.RESET} {speaker.color}[{speaker.name}]{C.RESET}")
                print(f"{response}")
                print(f"{C.DIM}   ε:{epsilon:.3f} Δρ:{rho_color}{delta_rho:+.4f}{C.RESET}{C.DIM} ρ:{speaker.dda_state.rho:.3f}{C.RESET}")
                
                if tool_results:
                    print(f"\n{C.CYAN}   ── Tool Execution ──{C.RESET}")
                    for line in tool_results.split('\n'):
                        print(f"   {C.CYAN}{line}{C.RESET}")
                    print()
                else:
                    print()
                
                # Check completion
                if self.check_proof_complete(response):
                    self.proof_complete = True
                    self.completion_step = message_count
                    print(f"\n{C.GREEN}{C.BOLD}{'═'*70}{C.RESET}")
                    print(f"{C.GREEN}{C.BOLD}  ★ PROOF DECLARED COMPLETE at message {message_count}!{C.RESET}")
                    print(f"{C.GREEN}{C.BOLD}{'═'*70}{C.RESET}")
                    break
                
                await asyncio.sleep(0.3)
        
        await self.display_final_state()
        await self.assemble_proof()
        await self.save_report()
    
    async def display_final_state(self):
        """Display final agent states."""
        print(f"\n\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  FINAL STATE SUMMARY{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        print(f"\n{C.WHITE}Agent States:{C.RESET}")
        print(f"{'─'*70}")
        print(f"{'Agent':12} | {'Final ρ':10} | {'Trauma':10} | {'Msgs':6} | {'Avg Trust':10}")
        print(f"{'─'*70}")
        
        for agent_id, agent in self.agents.items():
            idx = self.agent_id_to_idx[agent_id]
            
            trust_received = []
            for other_idx in range(len(self.agent_ids)):
                if other_idx != idx:
                    trust_received.append(self.trust_matrix.get_trust(other_idx, idx))
            avg_trust = np.mean(trust_received) if trust_received else 0.0
            
            trauma = getattr(agent.rigidity, 'rho_trauma', 0.0)
            
            print(f"{agent.color}{agent.name:12}{C.RESET} | {agent.dda_state.rho:10.4f} | {trauma:10.4f} | {agent.interaction_count:6} | {avg_trust:10.4f}")
        
        print(f"{'─'*70}")
        
        # Trust matrix
        print(f"\n{C.WHITE}Trust Matrix:{C.RESET}")
        print(f"{'':14}", end="")
        for aid in self.agent_ids:
            print(f"{aid[:4]:>7}", end="")
        print()
        
        for i, obs_id in enumerate(self.agent_ids):
            agent = self.agents[obs_id]
            print(f"{agent.color}{obs_id:14}{C.RESET}", end="")
            for j in range(len(self.agent_ids)):
                trust = self.trust_matrix.get_trust(i, j)
                print(f"{trust:7.3f}", end="")
            print()
        
        print(f"\n{C.WHITE}Proof Status: ", end="")
        if self.proof_complete:
            print(f"{C.GREEN}COMPLETE at step {self.completion_step}{C.RESET}")
        else:
            print(f"{C.YELLOW}PARTIAL (limit reached){C.RESET}")
        
        print(f"{C.WHITE}Total Messages: {len(self.conversation) - 1}{C.RESET}")
        print(f"{C.WHITE}Lemmas Proposed: {len(self.lemmas)}{C.RESET}")
    
    async def assemble_proof(self):
        """Assemble the complete proof argument from contributions."""
        print(f"\n\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  ASSEMBLED PROOF ARGUMENT{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}\n")
        
        if self.lemmas:
            print(f"{C.WHITE}PROPOSED LEMMAS:{C.RESET}")
            for i, lemma in enumerate(self.lemmas, 1):
                print(f"  L{i}. [{C.CYAN}{lemma['proposer']}{C.RESET}] {lemma['statement']}")
            print()
        
        print(f"{C.WHITE}PROOF ELEMENTS (in order):{C.RESET}\n")
        
        for elem in self.proof_elements:
            agent = self.agents.get(elem['agent'].upper(), None)
            color = agent.color if agent else C.WHITE
            
            print(f"{C.DIM}Step {elem['step']}{C.RESET} {color}[{elem['agent']}]{C.RESET}")
            print(f"  {elem['contribution'][:300]}{'...' if len(elem['contribution']) > 300 else ''}")
            if elem.get('tool_results'):
                print(f"  {C.CYAN}→ Tool: {elem['tool_results'][:100]}...{C.RESET}")
            print()
    
    async def save_report(self):
        """Save comprehensive report."""
        report_path = self.experiment_dir / "collatz_proof_report.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Collatz Conjecture Solver - Proof Attempt Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Status:** {'COMPLETE' if self.proof_complete else 'PARTIAL'}\n")
            f.write(f"**Messages Used:** {len(self.conversation) - 1}\n")
            f.write(f"**Lemmas Proposed:** {len(self.lemmas)}\n\n")
            
            f.write("## Mathematical Team\n\n")
            f.write("| Agent | Core Approach | Final ρ | Trauma | Contributions |\n")
            f.write("|-------|---------------|---------|--------|---------------|\n")
            for agent in self.agents.values():
                trauma = getattr(agent.rigidity, 'rho_trauma', 0.0)
                f.write(f"| {agent.name} | {agent.config['identity']['core'][:40]}... | {agent.dda_state.rho:.4f} | {trauma:.4f} | {agent.interaction_count} |\n")
            
            if self.lemmas:
                f.write("\n## Proposed Lemmas\n\n")
                for i, lemma in enumerate(self.lemmas, 1):
                    f.write(f"**L{i}** [{lemma['proposer']}]: {lemma['statement']}\n\n")
            
            f.write("\n## Full Transcript\n\n")
            for i, msg in enumerate(self.conversation):
                f.write(f"### [{i}] {msg['sender']}\n\n")
                f.write(f"{msg['text']}\n\n")
            
            f.write("\n## Proof Elements (Structured)\n\n")
            for elem in self.proof_elements:
                f.write(f"### Step {elem['step']} — {elem['agent']} (ε={elem['epsilon']:.3f}, ρ={elem['rho']:.4f})\n\n")
                f.write(f"{elem['contribution']}\n\n")
                if elem.get('tool_results'):
                    f.write(f"**Tool Output:**\n```\n{elem['tool_results']}\n```\n\n")
            
            f.write("\n## Trust Matrix (Final)\n\n")
            f.write("|" + "|".join(["Observer↓"] + [a[:5] for a in self.agent_ids]) + "|\n")
            f.write("|" + "---|" * (len(self.agent_ids) + 1) + "\n")
            for i, obs_id in enumerate(self.agent_ids):
                row = [f"{self.trust_matrix.get_trust(i, j):.3f}" for j in range(len(self.agent_ids))]
                f.write(f"|**{obs_id[:5]}**|" + "|".join(row) + "|\n")
            
            if self.tool_engine.execution_history:
                f.write("\n## Tool Execution Log\n\n")
                for i, exec_log in enumerate(self.tool_engine.execution_history, 1):
                    f.write(f"**{i}.** Type: `{exec_log['type']}`\n")
                    if 'query' in exec_log:
                        f.write(f"   Query: {exec_log['query']}\n")
                    f.write(f"   Result: {exec_log['result'][:200]}...\n\n")
        
        print(f"\n{C.GREEN}✓ Report saved to {report_path}{C.RESET}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

async def main():
    """Run the serious Collatz proof attempt."""
    print(f"\n{C.BOLD}╔{'═'*68}╗{C.RESET}")
    print(f"{C.BOLD}║  COLLATZ CONJECTURE SOLVER                                         ║{C.RESET}")
    print(f"{C.BOLD}║  Serious Mathematical Proof Attempt via DDA-X Agent Society        ║{C.RESET}")
    print(f"{C.BOLD}╚{'═'*68}╝{C.RESET}")
    
    sim = CollatzSolverSimulation()
    
    try:
        await sim.run(
            max_steps=20,
            max_responders=2
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        print(f"\n{C.YELLOW}Session interrupted. Saving partial report...{C.RESET}")
        await sim.save_report()
        raise


if __name__ == "__main__":
    if sys.platform == 'win32':
        os.system('color')
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.DIM}Session interrupted.{C.RESET}")
    except Exception as e:
        print(f"\n{C.RED}Error: {e}{C.RESET}")
        import traceback
        traceback.print_exc()
