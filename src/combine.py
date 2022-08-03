'''

'''
import argparse


def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str,
                        help="path to model")

    parser.add_argument("--targets", type=str,
                        help="path to targets csv")

    parser.add_argument("--drugs", type=str,
                        help="path to drugs csv")

    parser.add_argument("--out", type=str,
                        help="path to save results")

    parser.add_argument("--batch_size", type=int,
                        help="batch size of dataloader")

    parser.add_argument("--num_workers", type=int,
                        help="data loader # workers")

    return parser.parse_args()

if __name__ == '__main__': 
    
    args = get_args()