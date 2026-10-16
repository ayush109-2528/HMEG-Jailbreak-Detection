import argparse
from ml.training import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train from the imported MongoDB dataset; 0 means all rows.')
    parser.add_argument('--max-train', type=int, default=20000)
    parser.add_argument('--max-eval', type=int, default=1000)
    parser.add_argument('--semantic', action='store_true')
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--studies', action='store_true')
    parser.add_argument('--held-out-source')
    args = parser.parse_args()
    if args.max_train < 0 or args.max_eval < 0:
        parser.error('Limits must be nonnegative')
    train(**vars(args))
