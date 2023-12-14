import subprocess
import os
import numpy as np
from argparse import ArgumentParser
from datetime import datetime

def parse_arguments():

    # Create an argument parser object
    parser = ArgumentParser(description='Argparse for the range of the parameters')

    # Add arguments to the parser
    ## Training Param Args ##
    parser.add_argument('--sgd_per_round_range', nargs=3, type=int, default=[1, 10, 1], help='range for sgd_per_round in the experiments in the [min max step] format')
    parser.add_argument('--model_inertia_range', nargs=3, type=float, default=[0.1, 0.9, 0.1], help='range for model_inertia in the experiments in the [min max step] format')
    parser.add_argument('--save_dir', type=str, default='./results', help="directory to save the results in (default: ./results) - will be automatically created if doesn't exist")
    # Parse the command-line arguments
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    sgd_min, sgd_max, sgd_step = args.sgd_per_round_range
    lambda_min, lambda_max, lambda_step = args.model_inertia_range
    save_dir = os.path.join(args.save_dir, datetime.now().strftime('%Y%m%d_%H%M%S')+f"_sgd_{sgd_min}_{sgd_max}_{sgd_step}_lambda_{lambda_min}_{lambda_max}_{lambda_step}")
    
    for sgd_per_round in np.arange(sgd_min, sgd_max, sgd_step):
        for model_inertia in np.arange(lambda_min, lambda_max, lambda_step):
            subprocess.run(["python", "train.py", "--model_inertia", str(model_inertia), "--sgd_per_round", str(sgd_per_round), "--result_path", save_dir])
