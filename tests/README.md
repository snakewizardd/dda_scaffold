# DDA-X Test Suite

> **45/45 tests passing — 100% validation of core formulations**

## Running Tests

```bash
# Activate virtual environment
.\venv\Scripts\Activate

# Run the comprehensive test suite
python tests/test_ddax_claims.py

# Or run with pytest (if installed)
python -m pytest tests/ -v
```

## Test Coverage

| Category | Tests | Description |
|----------|-------|-------------|
| **D1: Surprise-Rigidity** | 4 | Monotonic ρ increase with ε, temperature mapping |
| **D2: Identity Attractor** | 3 | Core force dominance, resistance to perturbation |
| **D3: Exploration Dampening** | 6 | UCT × (1-ρ) formula verification |
| **D4: Multi-Timescale Trauma** | 5 | Asymmetric accumulation, composition |
| **D5: Trust Predictability** | 5 | T = 1/(1+Σε) formula, asymmetry |
| **D6: Hierarchical Identity** | 3 | γ ordering, force magnitude hierarchy |
| **D7: Metacognition** | 5 | Mode detection, self-report accuracy |
| **Core Physics** | 4 | State evolution equations |
| **Force Aggregation** | 3 | Channel composition |
| **Memory Retrieval** | 2 | Salience-weighted scoring |
| **Live Backend** | 5 | Ollama embedding integration |

## Output

Test results are saved to:
- `test_results/test_results.json` — Full JSON results
- `test_results/ddax_test_results.png` — Visual summary
- `test_results/suite_summary.md` — Markdown summary

## Requirements

- Python 3.10+
- Ollama running with `nomic-embed-text` (for live embedding tests)
- All other tests run without external dependencies
