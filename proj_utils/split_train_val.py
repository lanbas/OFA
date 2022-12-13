import csv 
import argparse
import random
import sys

csv.field_size_limit(sys.maxsize)
random.seed(69)

def main(args):
    csv_reader = csv.reader(open(args.csv_in, 'r'), delimiter='\t')
    train_writer = csv.writer(open(args.train_out, 'w'), delimiter='\t')
    val_writer = csv.writer(open(args.val_out, 'w'), delimiter='\t')

    for row in csv_reader:
        if random.random() < args.val_split:
            val_writer.writerow(row)
        else:
            train_writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for determining unroll label')
    parser.add_argument('--csv-in', type=str,
                        help='folder with problems inside')
    parser.add_argument('--train-out', type=str, default='../data/train.csv',
                        help='where to move problems that we do not have results for')
    parser.add_argument('--val-out', type=str, default='../data/val.csv',
                        help='locaton for problem folder with description, sample inputs, and random solution')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='proportion of data that is moved to the validation split')                 
    args = parser.parse_args()
    main(args)