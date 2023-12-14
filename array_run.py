import subprocess
import os
import numpy as np


for sgd_per_round in np.arange(1, 10):
    for model_inertia in np.arange(0.1, 1, 0.1):
        subprocess.run(["python", "train.py", "--model_inertia", model_inertia, "--sgd_per_round", sgd_per_round, "--result_path", "./results/result1"])