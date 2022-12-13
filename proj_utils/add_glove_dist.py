import numpy as np
import json
import math

def gen_word2glove_dict(embedding_dim):
    word2embedding = {}
    with open(f'./glove.6B.{embedding_dim}d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float32)
            word2embedding[word] = vect
    return word2embedding
    print("GloVe vectors loaded!")

def add_glove_dist():
    #read in json file
    with open(f'./caption_results/test_predict_ckpt5_results.json', 'r') as openfile:
        json_object = json.load(openfile)

    word2embedding = gen_word2glove_dict(100)
    total_parsed = 0
    num_correct = 0             #number correct based on exact comparison
    num_correct_glove = 0       #number correct based on glove distance
    for item in json_object.items():
        d = item[1]
        if not type(d) is dict:
            continue
        labels = d["label"]
        preds = d["prediction"]

        #calculate glove distance between predicted and ground truth verb noun
        glove_dist = 0.0
        for ind1 in range(len(labels)):
            dist = 0
            label_embedding = word2embedding[labels[ind1]]
            pred_embedding = word2embedding[preds[ind1]]
            for ind2 in range(len(label_embedding)):
                dist += (label_embedding[ind2] - pred_embedding[ind2])**2
            glove_dist += math.sqrt(dist)
        d["glove_dist"] = glove_dist
        if d["correct"] == 1:
            num_correct += 1
        #reevaluate correctness based on glove distance
        if glove_dist <= 6:
            d["correct"] = 1
            num_correct_glove += 1
        else:
            d["correct"] = 0
        total_parsed += 1
    print("Proportion correct based on exactness: ", num_correct/total_parsed)
    print("Proportion correct based on glove distance: ", num_correct_glove/total_parsed)

    #write to json file
    with open(f'./caption_results/pfa_ckpt5_glove_dist_added.json', 'w') as outfile:
        json.dump(json_object, outfile, indent=2)

add_glove_dist()