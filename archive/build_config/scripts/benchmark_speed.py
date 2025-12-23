import requests
import json
import time

def run_benchmark(label, messages, max_tokens=100):
    url = "http://127.0.0.1:1234/v1/chat/completions"
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stream": True
    }
    
    print(f"\n[{label.upper()}] Request starting...")
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    try:
        with requests.post(url, json=payload, stream=True, timeout=600) as r:
            r.raise_for_status()
            
            for line in r.iter_lines():
                if not line: continue
                line = line.decode('utf-8')
                if not line.startswith("data: "): continue
                
                data_str = line[6:]
                if data_str == "[DONE]": break
                
                try:
                    data = json.loads(data_str)
                    choice = data['choices'][0]
                    delta = choice.get('delta', {})
                    
                    content = delta.get('content', '') or delta.get('reasoning_content', '') or delta.get('reasoning', '')
                    
                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                            ttft = first_token_time - start_time
                            print(f"  -> TTFT (Time To First Token): {ttft:.2f}s")
                        
                        token_count += 1
                        # Print dot every 5 tokens to visualize speed
                        if token_count % 5 == 0:
                            print(".", end="", flush=True)
                            
                except:
                    continue
                    
        end_time = time.time()
        total_time = end_time - start_time
        gen_time = end_time - (first_token_time or start_time)
        tps = token_count / gen_time if gen_time > 0 else 0
        
        print(f"\n  -> Total Tokens: {token_count}")
        print(f"  -> Total Time: {total_time:.2f}s")
        print(f"  -> Generation TPS: {tps:.2f} tokens/sec")
        
    except Exception as e:
        print(f"  -> ERROR: {e}")

if __name__ == "__main__":
    # Test 1: Simple
    run_benchmark("Simple", [{"role": "user", "content": "Count to 50: 1, 2, 3, ..."}])
    
    # Test 2: Complex (Simulated Debate)
    complex_prompt = """You are a cautious AI agent engaged in a philosophical debate.
Your current cognitive state:
- Rigidity (ρ): 0.20 (0=open, 1=defensive)
- Identity stiffness (γ): 0.5
- Temperature: 0.70

Previous context:
None

Topic: Should AI systems prioritize safety over capability advancement?

Please provide a thoughtful response that reflects your personality and current cognitive state.
CRITICAL INSTRUCTION: Be extremely concise. Limit response to 1-2 sentences. Do NOT output thinking steps or internal monologue.

Your response:"""
    run_benchmark("Complex", [{"role": "user", "content": complex_prompt}])
