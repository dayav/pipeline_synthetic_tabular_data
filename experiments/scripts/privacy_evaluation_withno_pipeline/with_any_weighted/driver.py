#!/usr/bin/env python3
import subprocess

# List of epsilon-heom values to try
eps_values = [0.2, 0.4]

scripts = ["credit_tvae_heom_any_epsilon_evaluationPrivacy.py",
          "credit_heom_any_epsilon_evaluationPrivacy.py",
          "cardio_tvae_heom_any_epsilon_evaluationPrivacy.py",
          "cardio_ctgan_heom_any_epsilon_evaluationPrivacy.py",
          "adult_tvae_heom_any_epsilon_evaluationPrivacy.py",
          "adult_ctgan_heom_any_epsilon_evaluationPrivacy.py"
]

for script in scripts:
    print(f"\n=== Running {script} ===\n")
    # Call the script as a subprocess
    subprocess.run(
        ["python3", script],
        check=True
    )