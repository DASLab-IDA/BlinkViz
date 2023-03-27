import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spns')
    parser.add_argument('--nn')
    parser.add_argument('--dataset')
    #parser.add_argument('--dataset_batch', default="/home/qym/datasets/single_queries/")
    args = parser.parse_args()

    spns = args.spns
    nn = args.nn
    dataset = args.dataset

    