from network import Network
from parse_args import parse_arguments, modify_args
from plots import draw_bar_plot
import numpy as np
import torch



def main():
    args = parse_arguments()
    args = modify_args(args)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)


    network = Network(args)

    network.run()

    # print(len(network.clients))
    
                                        
    
    

    

    


if __name__ == '__main__':
    main()