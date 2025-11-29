import argparse

from configargparse import ArgumentParser
from typing import Union


# Needed to parse booleans from command line properly
def str2bool(v: Union[bool, str]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args() -> argparse.Namespace:
    parser = ArgumentParser(description='ORDNA', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument('--arg_log', default=False, type=str2bool, help='if set to True, save arguments to config file')
    parser.add_argument('--seed', default=0, type=int, help='seed for Pytorch Lightning function seed_everything()')

    # Required paths
    parser.add_argument('--embeddings_file', required=True, type=str, help='Path to the embeddings CSV file.')
    parser.add_argument('--protection_file', required=True, type=str, help='Path to the protection labels CSV file.')
    parser.add_argument('--habitat_file', required=True, type=str, help='Path to the habitat labels CSV file.')
    parser.add_argument('--k_cross_file',   required=True, type=str, help='Path to the k-fold split CSV (spygen_code + set=train/validation).')
    parser.add_argument("--lambda_domain", type=float, default=1.0, help="Weight of the domain-adversarial loss (DANN)")
    
    # Model and training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation.')
    parser.add_argument('--num_classes', required=True, type=int, help='Number of classes for classification.')
    parser.add_argument('--initial_learning_rate', type=float, default=1e-4, help='Initial learning rate for the optimizer.')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs to train.')

    
    # Training control
  
    parser.add_argument('--accelerator', type=str, default='gpu', help='Specify the accelerator device ("gpu" or "cpu").')


    return parser.parse_args()


def write_config_file(args: argparse.Namespace, path: str = 'config.cfg') -> None:
    with open(path, 'w') as f:
        for k in sorted(args.__dict__):
            if args.__dict__[k] != None:
                print(k, '=', args.__dict__[k], file=f)
