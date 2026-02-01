#!/usr/bin/env python3
import subprocess

# List of epsilon-heom values to try
eps_values = [0.01]

script = "cardio_tvae.py"

for eps in eps_values:
    print(f"\n=== Running with --epsilon-heom-knn-any-weighted {eps} ===\n")
    # Call the script as a subprocess
    subprocess.run(
        ["python3", script, "--epsilon-heom-knn-any-weighted", str(eps)],
        check=True
    )