#!/usr/bin/env python3
import subprocess

# List of epsilon-heom values to try
eps_values = [0.01, 0.05, 0.1, 0.2, 0.4]

script = "adult_tvae.py"

for eps in eps_values:
    print(f"\n=== Running with --epsilon-heom-knn-any {eps} ===\n")
    # Call the script as a subprocess
    subprocess.run(
        ["python3", script, "--epsilon-heom-knn-any", str(eps)],
        check=True
    )