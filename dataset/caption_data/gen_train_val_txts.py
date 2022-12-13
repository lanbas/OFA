import csv 
import argparse
import random
import sys

csv.field_size_limit(sys.maxsize)
train_tsv = 'dataset1_full_train.tsv'
val_tsv = 'dataset1_full_val.tsv'

train_reader = csv.reader(open(train_tsv, 'r'), delimiter='\t')
val_reader = csv.reader(open(val_tsv, 'r'), delimiter='\t')

for split, reader in [('train', train_reader), ('val', val_reader)]:
    with open(f'dataset1_full_{split}_ids.txt', 'w') as f_ptr:
        for row in reader:
            image_id = row[1]
            f_ptr.write(image_id + '\n')

