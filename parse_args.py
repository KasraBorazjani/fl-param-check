from argparse import ArgumentParser
import torch

def parse_arguments():

    # Create an argument parser object
    parser = ArgumentParser(description='Argparse for learning and network parameters')

    # Add arguments to the parser
    
    ## Training Param Args ##
    parser.add_argument('--sgd_per_round', type=int, default=5, help='number of local epochs per federated round (default: 1)')
    parser.add_argument('--num_clients', type=int, default=30, help='number of clients simulated (default: 50)')
    parser.add_argument('--neighbors_per_client', type=int, default=5, help='number of clients simulated (default: 50)')
    parser.add_argument('--model_inertia', type=float, default=0.05, help='inertia for model aggregation (default: 50)')


    ## Dataset Args ##
    parser.add_argument('--total_ds_len', type=int, default=300, help='total dataset length for each client (default: 200)')
    parser.add_argument('--primary_label_fraction', type=float, default=0.75, help="fraction of each client's dataset that is occupied by the preferred label (default: 0.75)")
    parser.add_argument('--num_secondaries', type=int, default=1, help='number of secondary labels (default: 1)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size (default: 16)')
    parser.add_argument('--shuffle_dataset', action='store_true', default=True, help='shuffling dataset (default: True)')

    ## Runtime Args ##
    parser.add_argument('--acc_used', action='store_true', default=False, help='whether acceleration (GPU/MPS) is used (default: False)')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed - set manually to adapt repeatability (default: 42)')
    parser.add_argument('--num_fed_rounds', type=int, default=70, help='maximum number of federated rounds to run the code for (default: 100)')
    parser.add_argument('--data_path', type=str, default="../data/mnist", help='directory for data to be loaded from')
    parser.add_argument('--saved_model_path', type=str, default="../saved_models", help='directory for the saved models to be loaded from')
    parser.add_argument('--init_lr', type=float, default=1e-2, help='Initial learning rate (default: 1e-2)')
    parser.add_argument('--result_path', type=str, default="./results", help='directory for the results to be stored')

    # Parse the command-line arguments
    args = parser.parse_args()

    return args

def modify_args(args):
    args.device = torch.device("cpu")
    if args.acc_used:
        if torch.cuda.is_available():
                args.device = torch.device("cuda:1")
        elif torch.backends.mps.is_available():
            args.device = torch.device("mps")
        else:
            print("accelerator not supported at the moment. falling back to cpu")

    
    return args