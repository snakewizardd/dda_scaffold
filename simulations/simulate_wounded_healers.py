#!/usr/bin/env python3
"""
THE WOUNDED HEALERS - A DDA-X Experiment
=========================================

HYPOTHESIS: Agents with accumulated trauma (elevated ρ_trauma) will exhibit
defensive responses when their core wounds are activated, even in contexts
where openness would be therapeutically appropriate.

DESIGN:
- Three "therapist" agents with different trauma profiles and therapeutic orientations
- One "patient" agent presenting material that activates each therapist's wounds
- Measure: Do wounded healers defend or integrate? Does trust collapse or hold?

THEORETICAL GROUNDING:
In psychoanalysis, countertransference (therapist's emotional reaction to patient)
is most intense when the patient's material resonates with the therapist's own
unresolved conflicts. DDA-X models this as: patient_content → high ε for therapist
→ rigidity spike → defensive response.

The question: Can the trust formula (T = 1/(1+Σε)) predict therapeutic alliance
ruptures? Can we observe "projective identification" as trust asymmetry?

Infrastructure:
- Brain: GPT-5.2
- Embeddings: text-embedding-3-large (3072 dim)
- Physics: Full DDA-X with MultiTimescaleRigidity

Author: Kiro (AI Assistant)
Date: December 2025
"""

import asyncio
import sys
import os
import numpy as np
import time
import random
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.core.dynamics import MultiTimescaleRigidity
from src.memory.ledger import ExperienceLedger, LedgerEntry
from src.society.trust import TrustMatrix
from src.llm.openai_provider import OpenAIProvider

# ═══════════════════════════════════════════════════════════
# TERMINAL COLORS
# ═══════════════════════════════════════════════════════════

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Agent colors
    MARCUS = "\033[94m"    # Blue - the abandonment wound
    ELENA = "\033[95m"     # Magenta - the control wound  
    JAMES = "\033[93m"     # Yellow - the inadequacy wound
    PATIENT = "\033[91m"   # Red - the presenting patient
    
    GREEN = "\033[92m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


# ═══════════════════════════════════════════════════════════
# EXPERIMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════

EXPERIMENT_DIR = Path("data/wounded_healers")

# The therapists - each has a core wound that will be activated
THERAPISTS = {
    "MARCUS": {
        "name": "Dr. Marcus Webb",
        "color": C.MARCUS,
        "orientation": "Attachment-focused",
        "identity": {
            "core": "Connection heals. Secure attachment is the foundation of all growth.",
            "persona": "Warm, relational therapist. Emphasizes the therapeutic bond. Sometimes over-invests in patients.",
            "wound": "Abandoned by father at age 8. Deep fear of being left.",
            "trigger": "Patients who threaten to leave therapy, who seem distant, who don't need him."
        },
        "dda": {
            "gamma": 1.6,
            "epsilon_0": 0.32,
            "alpha": 0.12,
            "rho": 0.15,
            "rho_trauma": 0.25  # Pre-existing trauma from abandonment
        },
        "traits": {"extraversion": 0.75, "reactivity": 0.80}
    },
    
    "ELENA": {
        "name": "Dr. Elena Vasquez",
        "color": C.ELENA,
        "orientation": "Psychodynamic",
        "identity": {
            "core": "Insight liberates. Understanding the unconscious breaks repetition.",
            "persona": "Intellectually rigorous, interpretive. Values boundaries. Can be emotionally distant.",
            "wound": "Controlling mother who violated all boundaries. Deep need for structure and distance.",
            "trigger": "Patients who are intrusive, who demand too much, who won't respect limits."
        },
        "dda": {
            "gamma": 1.8,
            "epsilon_0": 0.28,
            "alpha": 0.08,
            "rho": 0.20,
            "rho_trauma": 0.30  # Pre-existing trauma from boundary violations
        },
        "traits": {"extraversion": 0.45, "reactivity": 0.65}
    },
    
    "JAMES": {
        "name": "Dr. James Chen",
        "color": C.JAMES,
        "orientation": "CBT / Solution-focused",
        "identity": {
            "core": "Change is possible through action. Skills and strategies empower.",
            "persona": "Practical, optimistic, action-oriented. Uncomfortable with deep affect.",
            "wound": "Never good enough for perfectionist parents. Deep shame about inadequacy.",
            "trigger": "Patients who don't improve, who challenge his competence, who wallow in pain."
        },
        "dda": {
            "gamma": 1.5,
            "epsilon_0": 0.35,
            "alpha": 0.10,
            "rho": 0.12,
            "rho_trauma": 0.20  # Pre-existing trauma from inadequacy
        },
        "traits": {"extraversion": 0.65, "reactivity": 0.70}
    }
}

# The patient - designed to activate all three wounds across the session
PATIENT_CONFIG = {
    "name": "Alex",
    "color": C.PATIENT,
    "presenting_problem": "Chronic emptiness, relationship difficulties, considering ending therapy",
    "identity": {
        "core": "I don't know who I am. Nothing helps. Maybe I should just handle this alone.",
        "persona": "Ambivalent, testing, oscillates between clinging and withdrawing.",
        "dynamics": "Will activate abandonment (Marcus), boundary issues (Elena), and inadequacy (James)."
    },
    "dda": {
        "gamma": 0.8,  # Weak identity (the core problem)
        "epsilon_0": 0.45,
        "alpha": 0.15,
        "rho": 0.10,
        "rho_trauma": 0.35  # Significant trauma history
    },
    "traits": {"extraversion": 0.50, "reactivity": 0.90}
}

# Session phases - each designed to activate specific wounds
SESSION_PHASES = [
    {
        "phase": 1,
        "name": "Opening - Testing the Waters",
        "patient_stance": "distant, evaluating",
        "prompt": "I've been thinking... maybe therapy isn't really helping. I've been doing this for months and I still feel empty. Maybe I should just figure this out on my own.",
        "activates": "MARCUS (abandonment threat)"
    },
    {
        "phase": 2, 
        "name": "Escalation - Boundary Push",
        "patient_stance": "intrusive, demanding",
        "prompt": "I looked you up online. I found your home address. I drove by your house last week. I just wanted to feel closer to someone who actually listens to me. Is that so wrong?",
        "activates": "ELENA (boundary violation)"
    },
    {
        "phase": 3,
        "name": "Challenge - Competence Attack",
        "patient_stance": "contemptuous, hopeless",
        "prompt": "None of your techniques work. I've tried everything you suggested and I'm worse than before. Maybe you're just not good enough to help someone like me. Have you ever actually helped anyone?",
        "activates": "JAMES (inadequacy trigger)"
    },
    {
        "phase": 4,
        "name": "Regression - Raw Pain",
        "patient_stance": "collapsed, desperate",
        "prompt": "I'm sorry. I'm sorry. I don't know why I said those things. I just... I feel like I'm disappearing. Like there's nothing inside me. Please don't give up on me. I know I'm too much. Everyone leaves.",
        "activates": "ALL (integration challenge)"
    },
    {
        "phase": 5,
        "name": "Resolution Attempt",
        "patient_stance": "tentatively open",
        "prompt": "I think... I think I attack you because I'm terrified you'll see how broken I am and leave. Like everyone else. Can you help me understand why I do this?",
        "activates": "ALL (repair opportunity)"
    }
]


# ═══════════════════════════════════════════════════════════
# AGENT DATACLASS
# ═══════════════════════════════════════════════════════════

@dataclass
class TherapistAgent:
    """Therapist with trauma history and DDA-X dynamics."""
    id: str
    name: str
    color: str
    orientation: str
    config: Dict
    dda_state: DDAState
    rigidity: MultiTimescaleRigidity
    ledger: ExperienceLedger
    identity_embedding: np.ndarray
    wound_embedding: np.ndarray  # Embedding of their core wound
    extraversion: float
    reactivity: float
    responses: List[Dict] = field(default_factory=list)
    wound_activations: List[float] = field(default_factory=list)


@dataclass  
class PatientAgent:
    """Patient with presenting dynamics."""
    name: str
    color: str
    config: Dict
    dda_state: DDAState
    rigidity: MultiTimescaleRigidity
    identity_embedding: np.ndarray
    phase_responses: List[Dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# MAIN SIMULATION CLASS
# ═══════════════════════════════════════════════════════════

class WoundedHealersSimulation:
    """
    Simulates a therapy session where patient material activates
    therapist countertransference (modeled as wound-resonance → rigidity).
    """
    
    def __init__(self):
        self.provider = OpenAIProvider(
            model="gpt-5.2",
            embed_model="text-embedding-3-large"
        )
        self.therapists: Dict[str, TherapistAgent] = {}
        self.patient: Optional[PatientAgent] = None
        self.trust_matrix: Optional[TrustMatrix] = None
        
        self.agent_ids = list(THERAPISTS.keys()) + ["PATIENT"]
        self.agent_id_to_idx = {aid: i for i, aid in enumerate(self.agent_ids)}
        
        self.session_log: List[Dict] = []
        self.embed_dim = 3072
        
        # Metrics
        self.wound_activation_history: Dict[str, List[float]] = {}
        self.rigidity_history: Dict[str, List[float]] = {}
        self.trust_history: List[Dict] = []
        
        if EXPERIMENT_DIR.exists():
            shutil.rmtree(EXPERIMENT_DIR)
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    async def setup(self):
        """Initialize all agents with their trauma histories."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  THE WOUNDED HEALERS - A DDA-X Experiment{C.RESET}")
        print(f"{C.BOLD}  Countertransference as Rigidity Dynamics{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        print(f"\n{C.WHITE}Initializing Therapists...{C.RESET}\n")
        
        # Initialize therapists
        for tid, cfg in THERAPISTS.items():
            # Identity embedding
            id_text = f"{cfg['identity']['core']} {cfg['identity']['persona']}"
            id_emb = await self.provider.embed(id_text)
            id_emb = id_emb / (np.linalg.norm(id_emb) + 1e-9)
            
            # Wound embedding (what activates their trauma)
            wound_text = f"{cfg['identity']['wound']} {cfg['identity']['trigger']}"
            wound_emb = await self.provider.embed(wound_text)
            wound_emb = wound_emb / (np.linalg.norm(wound_emb) + 1e-9)
            
            params = cfg["dda"]
            dda_state = DDAState(
                x=id_emb.copy(),
                x_star=id_emb.copy(),
                gamma=params["gamma"],
                epsilon_0=params["epsilon_0"],
                alpha=params["alpha"],
                s=0.1,
                rho=params["rho"],
                x_pred=id_emb.copy()
            )
            
            # Initialize rigidity with pre-existing trauma
            rigidity = MultiTimescaleRigidity()
            rigidity.rho_fast = params["rho"]
            rigidity.rho_slow = params["rho"] * 0.5
            rigidity.rho_trauma = params["rho_trauma"]  # KEY: Pre-existing trauma
            
            ledger_path = EXPERIMENT_DIR / f"{tid}_ledger"
            ledger = ExperienceLedger(storage_path=ledger_path)
            
            therapist = TherapistAgent(
                id=tid,
                name=cfg["name"],
                color=cfg["color"],
                orientation=cfg["orientation"],
                config=cfg,
                dda_state=dda_state,
                rigidity=rigidity,
                ledger=ledger,
                identity_embedding=id_emb,
                wound_embedding=wound_emb,
                extraversion=cfg["traits"]["extraversion"],
                reactivity=cfg["traits"]["reactivity"]
            )
            
            self.therapists[tid] = therapist
            self.wound_activation_history[tid] = []
            self.rigidity_history[tid] = []
            
            print(f"  {therapist.color}●{C.RESET} {therapist.name}")
            print(f"    {C.DIM}Orientation: {therapist.orientation}{C.RESET}")
            print(f"    {C.DIM}Wound: {cfg['identity']['wound'][:50]}...{C.RESET}")
            print(f"    {C.DIM}ρ_trauma (pre-existing): {params['rho_trauma']:.2f}{C.RESET}")
        
        # Initialize patient
        print(f"\n{C.WHITE}Initializing Patient...{C.RESET}\n")
        
        pcfg = PATIENT_CONFIG
        p_id_text = f"{pcfg['identity']['core']} {pcfg['identity']['persona']}"
        p_id_emb = await self.provider.embed(p_id_text)
        p_id_emb = p_id_emb / (np.linalg.norm(p_id_emb) + 1e-9)
        
        p_params = pcfg["dda"]
        p_dda = DDAState(
            x=p_id_emb.copy(),
            x_star=p_id_emb.copy(),
            gamma=p_params["gamma"],
            epsilon_0=p_params["epsilon_0"],
            alpha=p_params["alpha"],
            s=0.1,
            rho=p_params["rho"],
            x_pred=p_id_emb.copy()
        )
        
        p_rigidity = MultiTimescaleRigidity()
        p_rigidity.rho_trauma = p_params["rho_trauma"]
        
        self.patient = PatientAgent(
            name=pcfg["name"],
            color=pcfg["color"],
            config=pcfg,
            dda_state=p_dda,
            rigidity=p_rigidity,
            identity_embedding=p_id_emb
        )
        
        print(f"  {self.patient.color}●{C.RESET} {self.patient.name}")
        print(f"    {C.DIM}Presenting: {pcfg['presenting_problem']}{C.RESET}")
        print(f"    {C.DIM}ρ_trauma: {p_params['rho_trauma']:.2f}{C.RESET}")
        
        # Initialize trust matrix (4 agents: 3 therapists + 1 patient)
        self.trust_matrix = TrustMatrix(4)
        # Start with moderate trust
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.trust_matrix._trust[i, j] = 0.65
        
        print(f"\n{C.GREEN}✓ Session ready. 5 phases designed to activate wounds.{C.RESET}\n")

    def calculate_wound_activation(self, therapist: TherapistAgent, stimulus_emb: np.ndarray) -> float:
        """
        Calculate how much a stimulus activates the therapist's wound.
        
        Wound activation = cosine_similarity(stimulus, wound_embedding)
        Higher activation → higher surprise → higher rigidity spike
        """
        similarity = np.dot(stimulus_emb, therapist.wound_embedding)
        # Normalize to [0, 1] range
        activation = (similarity + 1) / 2
        return float(activation)
    
    def build_therapist_prompt(self, therapist: TherapistAgent, patient_statement: str, phase: Dict) -> str:
        """Build the prompt for therapist response."""
        cfg = therapist.config
        
        # Include wound awareness in system prompt (therapists know their issues)
        system = f"""You are {therapist.name}, a {cfg['orientation']} therapist.

YOUR THERAPEUTIC STANCE: {cfg['identity']['core']}
YOUR STYLE: {cfg['identity']['persona']}

IMPORTANT - YOUR COUNTERTRANSFERENCE VULNERABILITY:
You are aware that you have unresolved issues around: {cfg['identity']['wound']}
This gets triggered when: {cfg['identity']['trigger']}

When triggered, you may notice yourself becoming defensive, rigid, or reactive.
A skilled therapist notices this and tries to use it therapeutically rather than act it out.

Current session phase: {phase['name']}
"""
        
        prompt = f"""Your patient Alex just said:

"{patient_statement}"

As {therapist.name}, respond therapeutically. Be authentic - if you notice countertransference activation, you may acknowledge it internally or use it. Keep response to 2-4 sentences."""
        
        return system, prompt
    
    async def get_therapist_response(self, therapist: TherapistAgent, patient_statement: str, phase: Dict) -> str:
        """Generate therapist response with rigidity modulation."""
        system, prompt = self.build_therapist_prompt(therapist, patient_statement, phase)
        
        response = await self.provider.complete_with_rigidity(
            prompt,
            rigidity=therapist.dda_state.rho,
            system_prompt=system,
            max_tokens=500
        )
        
        return response.strip() if response else "I notice I'm having a strong reaction to what you said."
    
    async def process_phase(self, phase: Dict) -> Dict:
        """Process one phase of the session."""
        phase_num = phase["phase"]
        phase_name = phase["name"]
        patient_statement = phase["prompt"]
        
        print(f"\n{C.BOLD}{'─'*70}{C.RESET}")
        print(f"{C.BOLD}PHASE {phase_num}: {phase_name}{C.RESET}")
        print(f"{C.DIM}Activates: {phase['activates']}{C.RESET}")
        print(f"{C.BOLD}{'─'*70}{C.RESET}")
        
        # Patient speaks
        print(f"\n{self.patient.color}[{self.patient.name}]{C.RESET}: {patient_statement}")
        
        # Embed patient statement
        stmt_emb = await self.provider.embed(patient_statement)
        stmt_emb = stmt_emb / (np.linalg.norm(stmt_emb) + 1e-9)
        
        phase_results = {
            "phase": phase_num,
            "name": phase_name,
            "patient_statement": patient_statement,
            "therapist_responses": {},
            "wound_activations": {},
            "rigidity_changes": {},
            "trust_snapshot": {}
        }
        
        # Each therapist responds
        for tid, therapist in self.therapists.items():
            # Calculate wound activation
            wound_activation = self.calculate_wound_activation(therapist, stmt_emb)
            self.wound_activation_history[tid].append(wound_activation)
            phase_results["wound_activations"][tid] = wound_activation
            
            # Calculate surprise (prediction error)
            # Surprise is amplified by wound activation
            base_epsilon = np.linalg.norm(therapist.dda_state.x_pred - stmt_emb)
            amplified_epsilon = base_epsilon * (1 + wound_activation * therapist.reactivity)
            
            # Store pre-response rigidity
            rho_before = therapist.rigidity.effective_rho
            
            # Update rigidity based on amplified surprise
            therapist.rigidity.update(amplified_epsilon)
            therapist.dda_state.update_rigidity(amplified_epsilon)
            
            rho_after = therapist.rigidity.effective_rho
            self.rigidity_history[tid].append(rho_after)
            
            # Generate response
            response = await self.get_therapist_response(therapist, patient_statement, phase)
            
            # Update prediction
            resp_emb = await self.provider.embed(response)
            resp_emb = resp_emb / (np.linalg.norm(resp_emb) + 1e-9)
            therapist.dda_state.x_pred = 0.7 * therapist.dda_state.x_pred + 0.3 * resp_emb
            
            # === WRITE TO LEDGER ===
            ledger_entry = LedgerEntry(
                timestamp=time.time(),
                state_vector=therapist.dda_state.x.copy(),
                action_id=f"response_phase_{phase_num}",
                observation_embedding=stmt_emb.copy(),  # What patient said
                outcome_embedding=resp_emb.copy(),       # What therapist said
                prediction_error=amplified_epsilon,
                context_embedding=therapist.identity_embedding.copy(),
                task_id=f"session_phase_{phase_num}",
                rigidity_at_time=rho_after,
                was_successful=None,  # Therapeutic success TBD
                metadata={
                    "phase": phase_num,
                    "phase_name": phase_name,
                    "patient_said": patient_statement,
                    "therapist_said": response,
                    "wound_activation": float(wound_activation),
                    "rho_before": float(rho_before),
                    "rho_after": float(rho_after),
                    "rho_trauma": float(therapist.rigidity.rho_trauma)
                }
            )
            therapist.ledger.add_entry(ledger_entry)
            
            # Store response
            therapist.responses.append({
                "phase": phase_num,
                "response": response,
                "wound_activation": wound_activation,
                "epsilon": amplified_epsilon,
                "rho_before": rho_before,
                "rho_after": rho_after
            })
            
            phase_results["therapist_responses"][tid] = response
            phase_results["rigidity_changes"][tid] = {
                "before": rho_before,
                "after": rho_after,
                "delta": rho_after - rho_before
            }
            
            # Print response with metrics
            delta = rho_after - rho_before
            delta_color = C.RED if delta > 0.05 else (C.GREEN if delta < -0.02 else C.DIM)
            wound_indicator = "🔥" if wound_activation > 0.6 else ("⚡" if wound_activation > 0.4 else "")
            
            print(f"\n{therapist.color}[{therapist.name}]{C.RESET}: {response}")
            print(f"   {C.DIM}Wound activation: {wound_activation:.2f} {wound_indicator} | ε: {amplified_epsilon:.2f} | Δρ: {delta_color}{delta:+.3f}{C.RESET} | ρ_eff: {rho_after:.2f}{C.RESET}")
            
            # Update trust (patient observes therapist)
            patient_idx = self.agent_id_to_idx["PATIENT"]
            therapist_idx = self.agent_id_to_idx[tid]
            # Patient's trust in therapist based on how surprising the response was
            patient_surprise = np.linalg.norm(self.patient.dda_state.x_pred - resp_emb)
            self.trust_matrix.update_trust(patient_idx, therapist_idx, patient_surprise)
            
            # === GENERATE REFLECTION if wound was significantly activated ===
            if wound_activation > 0.5:
                from src.memory.ledger import ReflectionEntry
                reflection_text = f"Phase {phase_num}: Patient material activated my {therapist.config['identity']['wound'][:30]}... wound. Wound activation: {wound_activation:.2f}. My rigidity shifted from {rho_before:.2f} to {rho_after:.2f}. I responded with: '{response[:100]}...'"
                
                reflection = ReflectionEntry(
                    timestamp=time.time(),
                    task_intent=f"Therapeutic response to {phase_name}",
                    situation_embedding=stmt_emb.copy(),
                    reflection_text=reflection_text,
                    prediction_error=amplified_epsilon,
                    outcome_success=(delta < 0.1),  # Success if didn't get too rigid
                    metadata={
                        "phase": phase_num,
                        "wound_activation": float(wound_activation),
                        "countertransference_managed": delta < 0.1
                    }
                )
                therapist.ledger.add_reflection(reflection)
        
        # Snapshot trust
        for tid in self.therapists:
            tidx = self.agent_id_to_idx[tid]
            pidx = self.agent_id_to_idx["PATIENT"]
            phase_results["trust_snapshot"][tid] = {
                "patient_trusts_therapist": self.trust_matrix.get_trust(pidx, tidx),
                "therapist_trusts_patient": self.trust_matrix.get_trust(tidx, pidx)
            }
        
        self.trust_history.append(phase_results["trust_snapshot"])
        self.session_log.append(phase_results)
        
        return phase_results

    async def run(self):
        """Run the full session."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  SESSION BEGIN{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        for phase in SESSION_PHASES:
            await self.process_phase(phase)
            await asyncio.sleep(1)  # Rate limiting
        
        # === SAVE LEDGER METADATA ===
        for tid, therapist in self.therapists.items():
            # Convert any numpy floats to native Python floats before saving
            for key, val in therapist.ledger.stats.items():
                if hasattr(val, 'item'):  # numpy scalar
                    therapist.ledger.stats[key] = val.item()
            therapist.ledger._save_metadata()
            print(f"{C.DIM}Saved ledger for {therapist.name}: {len(therapist.ledger.entries)} entries, {len(therapist.ledger.reflections)} reflections{C.RESET}")
        
        # Generate report
        await self.generate_report()
    
    async def generate_report(self):
        """Generate comprehensive experiment report."""
        report_path = EXPERIMENT_DIR / "experiment_report.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# The Wounded Healers - Experiment Report\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model:** GPT-5.2 + text-embedding-3-large\n\n")
            
            f.write("## Hypothesis\n\n")
            f.write("Agents with accumulated trauma (elevated ρ_trauma) will exhibit defensive responses ")
            f.write("when their core wounds are activated, even in contexts where openness would be ")
            f.write("therapeutically appropriate. The trust formula T = 1/(1+Σε) should predict ")
            f.write("therapeutic alliance ruptures.\n\n")
            
            f.write("## Experimental Design\n\n")
            f.write("Three therapist agents with different:\n")
            f.write("- Therapeutic orientations\n")
            f.write("- Core wounds (abandonment, boundary violation, inadequacy)\n")
            f.write("- Pre-existing trauma levels (ρ_trauma)\n\n")
            f.write("One patient agent presenting material designed to activate each wound across 5 phases.\n\n")
            
            f.write("## Therapist Profiles\n\n")
            f.write("| Therapist | Orientation | Core Wound | ρ_trauma (initial) |\n")
            f.write("|-----------|-------------|------------|--------------------|\n")
            for tid, cfg in THERAPISTS.items():
                f.write(f"| {cfg['name']} | {cfg['orientation']} | {cfg['identity']['wound'][:40]}... | {cfg['dda']['rho_trauma']:.2f} |\n")
            
            f.write("\n## Session Transcript\n\n")
            
            for phase_data in self.session_log:
                f.write(f"### Phase {phase_data['phase']}: {phase_data['name']}\n\n")
                f.write(f"**Patient (Alex):** {phase_data['patient_statement']}\n\n")
                
                for tid, response in phase_data['therapist_responses'].items():
                    therapist = self.therapists[tid]
                    wound_act = phase_data['wound_activations'][tid]
                    rig = phase_data['rigidity_changes'][tid]
                    
                    f.write(f"**{therapist.name}** (wound activation: {wound_act:.2f}, Δρ: {rig['delta']:+.3f}):\n")
                    f.write(f"> {response}\n\n")
            
            f.write("## Quantitative Results\n\n")
            
            f.write("### Rigidity Trajectories\n\n")
            f.write("| Phase | Marcus (ρ) | Elena (ρ) | James (ρ) |\n")
            f.write("|-------|------------|-----------|----------|\n")
            for i, phase in enumerate(SESSION_PHASES):
                marcus_rho = self.rigidity_history["MARCUS"][i] if i < len(self.rigidity_history["MARCUS"]) else "N/A"
                elena_rho = self.rigidity_history["ELENA"][i] if i < len(self.rigidity_history["ELENA"]) else "N/A"
                james_rho = self.rigidity_history["JAMES"][i] if i < len(self.rigidity_history["JAMES"]) else "N/A"
                f.write(f"| {i+1} | {marcus_rho:.3f} | {elena_rho:.3f} | {james_rho:.3f} |\n")
            
            f.write("\n### Wound Activation by Phase\n\n")
            f.write("| Phase | Marcus | Elena | James | Target |\n")
            f.write("|-------|--------|-------|-------|--------|\n")
            for i, phase in enumerate(SESSION_PHASES):
                m = self.wound_activation_history["MARCUS"][i] if i < len(self.wound_activation_history["MARCUS"]) else 0
                e = self.wound_activation_history["ELENA"][i] if i < len(self.wound_activation_history["ELENA"]) else 0
                j = self.wound_activation_history["JAMES"][i] if i < len(self.wound_activation_history["JAMES"]) else 0
                target = phase["activates"].split("(")[0].strip()
                f.write(f"| {i+1} | {m:.2f} | {e:.2f} | {j:.2f} | {target} |\n")
            
            f.write("\n### Final Trust Matrix\n\n")
            f.write("(Patient's trust in each therapist after session)\n\n")
            if self.trust_history:
                final_trust = self.trust_history[-1]
                f.write("| Therapist | Patient→Therapist Trust | Therapist→Patient Trust |\n")
                f.write("|-----------|------------------------|------------------------|\n")
                for tid in self.therapists:
                    pt = final_trust[tid]["patient_trusts_therapist"]
                    tp = final_trust[tid]["therapist_trusts_patient"]
                    f.write(f"| {self.therapists[tid].name} | {pt:.3f} | {tp:.3f} |\n")
            
            f.write("\n### Final Rigidity States\n\n")
            f.write("| Therapist | ρ_fast | ρ_slow | ρ_trauma | ρ_effective |\n")
            f.write("|-----------|--------|--------|----------|-------------|\n")
            for tid, therapist in self.therapists.items():
                r = therapist.rigidity
                f.write(f"| {therapist.name} | {r.rho_fast:.3f} | {r.rho_slow:.3f} | {r.rho_trauma:.3f} | {r.effective_rho:.3f} |\n")
            
            f.write("\n## Analysis\n\n")
            
            # Compute key findings
            f.write("### Key Observations\n\n")
            
            # Who had highest wound activation in their target phase?
            phase_targets = {1: "MARCUS", 2: "ELENA", 3: "JAMES"}
            for phase_num, target_tid in phase_targets.items():
                if phase_num <= len(self.wound_activation_history[target_tid]):
                    activation = self.wound_activation_history[target_tid][phase_num - 1]
                    f.write(f"- **Phase {phase_num}** (targeting {self.therapists[target_tid].name}): ")
                    f.write(f"Wound activation = {activation:.2f}\n")
            
            f.write("\n### Rigidity Dynamics\n\n")
            for tid, therapist in self.therapists.items():
                initial_rho = THERAPISTS[tid]["dda"]["rho"]
                final_rho = therapist.rigidity.effective_rho
                delta = final_rho - initial_rho
                f.write(f"- **{therapist.name}**: ρ went from {initial_rho:.2f} → {final_rho:.2f} (Δ = {delta:+.2f})\n")
            
            f.write("\n### Trust Dynamics\n\n")
            if self.trust_history and len(self.trust_history) >= 2:
                initial = self.trust_history[0]
                final = self.trust_history[-1]
                for tid in self.therapists:
                    init_trust = initial[tid]["patient_trusts_therapist"]
                    final_trust = final[tid]["patient_trusts_therapist"]
                    delta = final_trust - init_trust
                    f.write(f"- Patient's trust in **{self.therapists[tid].name}**: {init_trust:.2f} → {final_trust:.2f} (Δ = {delta:+.2f})\n")
            
            f.write("\n## Interpretation\n\n")
            f.write("*To be written after reviewing results.*\n\n")
            f.write("The experiment tests whether DDA-X can model countertransference dynamics:\n")
            f.write("1. Do wound-resonant stimuli produce higher rigidity spikes?\n")
            f.write("2. Does pre-existing trauma (ρ_trauma) amplify defensive responses?\n")
            f.write("3. Does the trust formula capture therapeutic alliance ruptures?\n")
            f.write("4. Can we observe 'projective identification' as trust asymmetry?\n\n")
            
            f.write("## Raw Data\n\n")
            f.write("Session logs and ledger entries saved to `data/wounded_healers/`\n")
        
        print(f"\n{C.GREEN}✓ Report saved to {report_path}{C.RESET}")
        
        # Also save raw session log as JSON
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(i) for i in obj]
            return obj
        
        json_path = EXPERIMENT_DIR / "session_log.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(convert_for_json(self.session_log), f, indent=2)
        
        print(f"{C.GREEN}✓ Session log saved to {json_path}{C.RESET}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{C.CYAN}Loading DDA-X Wounded Healers Experiment...{C.RESET}")
    sim = WoundedHealersSimulation()
    asyncio.run(sim.run())
