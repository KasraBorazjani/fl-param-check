import os
from datetime import datetime

def setup_result_path(args):

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
    

    path_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.result_path,path_time+"_lambda_"+str(args.model_inertia)+"_local_length"+str(args.sgd_per_round))
    os.mkdir(save_dir)
    
    return save_dir