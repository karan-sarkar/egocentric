import argparse

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--ckpt', type=str)

    return parser.parse_args()