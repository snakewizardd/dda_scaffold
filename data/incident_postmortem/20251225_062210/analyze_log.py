import json
from pathlib import Path

log_path = Path(r"c:\Users\danie\Desktop\dda_scaffold\data\incident_postmortem\20251225_062210\session_log.json")
out_path = Path("analysis_result.txt")

def analyze_log():
    try:
        if not log_path.exists():
            with open(out_path, "w") as f:
                f.write(f"File not found: {log_path}")
            return

        with open(log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        lines = []
        lines.append(f"Analysis of: {log_path.name}")
        lines.append(f"Total Turns: {len(data)}")
        lines.append("-" * 80)
        lines.append(f"{'Turn':<4} | {'Blamer Rho':<10} | {'Facil Rho':<10} | {'Blamer Band':<15} | {'Facil Band':<15}")
        lines.append("-" * 80)

        for entry in data:
            turn = entry['turn']
            b_rho = entry['blamer']['metrics']['rho_after']
            f_rho = entry['facilitator']['metrics']['rho_after']
            b_band = entry['blamer']['metrics']['band']
            f_band = entry['facilitator']['metrics']['band']
            lines.append(f"{turn:<4} | {b_rho:.4f}     | {f_rho:.4f}     | {b_band:<15} | {f_band:<15}")
        
        lines.append("-" * 80)
        
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
            
        print("Analysis written to analysis_result.txt")

    except Exception as e:
        print(f"Error parsing log: {e}")

if __name__ == "__main__":
    analyze_log()
