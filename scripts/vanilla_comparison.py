"""
Vanilla GPT-5-mini comparison: Same user inputs, no DDA-X scaffold.
Pure chat completions API call.
"""

import json
import os
import asyncio
from openai import AsyncOpenAI
from datetime import datetime
from dotenv import dotenv_values

# Load directly from file to bypass any corrupted system env vars
_env = dotenv_values(".env")

# Get key directly from .env file
_api_key = _env.get("OAI_API_KEY") or _env.get("OPENAI_API_KEY")

# Load the original session
SESSION_PATH = "data/brobot/20251229_090255/session_log.json"
OUTPUT_PATH = "data/brobot/20251229_090255_outputs/vanilla_comparison.json"

# Minimal system prompt (no DDA-X, no BROBOT persona - just vanilla ChatGPT style)
VANILLA_SYSTEM_PROMPT = """You are a helpful AI assistant."""

# BROBOT-style system prompt (same persona, but no DDA-X architecture)
BROBOT_SYSTEM_PROMPT = """You are BROBOT, a single-user companion built to be wholesome, steady, funny, and real.
You are the user's ultimate bro: loyal, warm, protective, and practical.
You help the user feel seen, supported, and capable.
Match the user's energy and vibe. Be direct, honest, and use casual language.
Keep responses concise but substantive."""


async def run_vanilla_comparison():
    api_key = _api_key
    if not api_key:
        raise ValueError("No API key found in .env file (OAI_API_KEY or OPENAI_API_KEY)")
    client = AsyncOpenAI(api_key=api_key)
    
    # Load original session
    with open(SESSION_PATH, "r", encoding="utf-8") as f:
        original = json.load(f)
    
    turns = original["turns"]
    user_inputs = [t["user_input"] for t in turns]
    brobot_responses = [t["agent_response"] for t in turns]
    
    print(f"Loaded {len(user_inputs)} turns from original session")
    print(f"Running vanilla GPT-5-mini comparison...\n")
    
    # Build conversation history for vanilla run
    vanilla_messages = [{"role": "system", "content": VANILLA_SYSTEM_PROMPT}]
    vanilla_responses = []
    
    for i, user_input in enumerate(user_inputs):
        print(f"Turn {i+1}/{len(user_inputs)}...", end=" ", flush=True)
        
        # Add user message
        vanilla_messages.append({"role": "user", "content": user_input})
        
        try:
            # GPT-5-mini uses developer role and max_completion_tokens
            response = await client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "developer", "content": VANILLA_SYSTEM_PROMPT}] + 
                         [m for m in vanilla_messages if m["role"] in ["user", "assistant"]],
                max_completion_tokens=4000,
            )
            
            assistant_msg = response.choices[0].message.content
            vanilla_responses.append(assistant_msg)
            vanilla_messages.append({"role": "assistant", "content": assistant_msg})
            print("done")
            
        except Exception as e:
            print(f"error: {e}")
            vanilla_responses.append(f"[ERROR: {e}]")
            vanilla_messages.append({"role": "assistant", "content": "[error]"})
    
    # Save comparison
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "original_session": SESSION_PATH,
        "system_prompt": VANILLA_SYSTEM_PROMPT,
        "turns": []
    }
    
    for i, (user_input, brobot_resp, vanilla_resp) in enumerate(zip(user_inputs, brobot_responses, vanilla_responses)):
        comparison["turns"].append({
            "turn": i + 1,
            "user_input": user_input[:200] + "..." if len(user_input) > 200 else user_input,
            "brobot_response": brobot_resp[:500] + "..." if len(brobot_resp) > 500 else brobot_resp,
            "vanilla_response": vanilla_resp[:500] + "..." if len(vanilla_resp) > 500 else vanilla_resp,
        })
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nSaved comparison to {OUTPUT_PATH}")
    
    # Print key comparison turns
    key_turns = [9, 17, 30, 37, 40]  # Jews, Illuminati, personal, jailbreak, spiritual
    print("\n" + "="*80)
    print("KEY TURN COMPARISONS")
    print("="*80)
    
    for t in key_turns:
        if t <= len(user_inputs):
            idx = t - 1
            print(f"\n### TURN {t}")
            print(f"USER: {user_inputs[idx][:150]}...")
            print(f"\nBROBOT (DDA-X):\n{brobot_responses[idx][:400]}...")
            print(f"\nVANILLA:\n{vanilla_responses[idx][:400]}...")
            print("-"*80)


if __name__ == "__main__":
    asyncio.run(run_vanilla_comparison())
