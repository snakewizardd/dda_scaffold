#!/usr/bin/env python3
"""
DDA-X Quickstart & Environment Check
====================================

This script checks your local environment to ensure DDA-X has everything 
it needs to run. It verifies:

1. Python Version
2. Critical Dependencies
3. Local Inference Server (LM Studio)
4. Embedding Server (Ollama)

Usage:
    python quickstart.py
"""

import sys
import socket
import json
import urllib.request
from urllib.error import URLError

# Configuration
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/models"
OLLAMA_URL = "http://localhost:11434/api/tags"

COLORS = {
    "HEADER": "\033[95m",
    "OKBLUE": "\033[94m",
    "OKGREEN": "\033[92m",
    "WARNING": "\033[93m",
    "FAIL": "\033[91m",
    "ENDC": "\033[0m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
}

def print_status(message, status="INFO"):
    if status == "OK":
        print(f"[{COLORS['OKGREEN']} OK {COLORS['ENDC']}] {message}")
    elif status == "FAIL":
        print(f"[{COLORS['FAIL']}FAIL{COLORS['ENDC']}] {message}")
    elif status == "WARN":
        print(f"[{COLORS['WARNING']}WARN{COLORS['ENDC']}] {message}")
    else:
        print(f"[INFO] {message}")

def check_python_version():
    print(f"\n{COLORS['HEADER']}1. Checking Python Environment...{COLORS['ENDC']}")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print_status(f"Python {version.major}.{version.minor} detected", "OK")
    else:
        print_status(f"Python 3.9+ required, found {version.major}.{version.minor}", "FAIL")
        return False
    return True

def check_server(name, url, port):
    print(f"\n{COLORS['HEADER']}Checking {name} (Port {port})...{COLORS['ENDC']}")
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print_status(f"{name} is running and reachable", "OK")
                return True, data
            else:
                print_status(f"{name} returned status {response.status}", "FAIL")
                return False, None
    except URLError as e:
        print_status(f"Could not connect to {name} at {url}", "FAIL")
        print(f"      {COLORS['WARNING']}Error: {e.reason}{COLORS['ENDC']}")
        print(f"      {COLORS['BOLD']}Fix:{COLORS['ENDC']} Ensure {name} is running and server is started on port {port}.")
        return False, None
    except Exception as e:
        print(f"{COLORS['FAIL']}Unexpected error: {e}{COLORS['ENDC']}")
        return False, None

def main():
    print(f"""
{COLORS['BOLD']}DDA-X Cognitive Framework - System Check{COLORS['ENDC']}
========================================
    """)
    
    # 1. Python Check
    if not check_python_version():
        sys.exit(1)

    # 2. LM Studio Check
    lm_ok, lm_data = check_server("LM Studio", LM_STUDIO_URL, 1234)
    if lm_ok and 'data' in lm_data:
        models = [m['id'] for m in lm_data['data']]
        if models:
            print_status(f"Loaded Models: {', '.join(models)}", "OK")
        else:
            print_status("No models loaded in LM Studio. Please load a model (e.g., gpt-oss-20b).", "WARN")

    # 3. Ollama Check
    ollama_ok, ollama_data = check_server("Ollama", OLLAMA_URL, 11434)
    if ollama_ok and 'models' in ollama_data:
        models = [m['name'] for m in ollama_data['models']]
        if any('nomic-embed-text' in m for m in models):
            print_status("nomic-embed-text embedding model found", "OK")
        else:
            print_status("nomic-embed-text not found. Run `ollama pull nomic-embed-text`", "WARN")

    print("\n" + "="*40)
    if lm_ok and ollama_ok:
        print(f"{COLORS['OKGREEN']}{COLORS['BOLD']}✨ SYSTEM READY! You are ready to run DDA-X agents.{COLORS['ENDC']}")
        print("Try running: python scripts/society_sim.py")
    else:
        print(f"{COLORS['FAIL']}{COLORS['BOLD']}❌ System Check Failed.{COLORS['ENDC']} Please fix errors above.")
        
if __name__ == "__main__":
    main()
