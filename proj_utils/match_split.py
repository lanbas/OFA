import os 
import csv

def main(args):
    train_ids = open(args.train_split_to_match, 'r').readlines() 
    train_ids = [x.strip() for x in train_ids]
    val_ids = open(args.val_split_to_match, 'r').readlines()
    val_ids = [x.strip() for x in val_ids]

    # for line in your full tsv 
        # if image_id is in train_ids, write to train file 
        # else write to val file 
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-in", type=str, help="Your csv that you want to split into train/val according to antoher split")
    parser.add_argument("--train-out", type=str, help="tsv to save train data to")
    parser.add_argument("--val-out", type=str, help="tsv to save val data to")
    parser.add_argument("--train-split-to-match", type=str, help="txt file containing image ids in train set")
    parser.add_argument("--val-split-to-match", type=str, help="txt file containing image ids in val set")
    args= parser.parse_args()
    main(args)    