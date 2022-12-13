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

#read in json file
with open('pred_for_all_results.json', 'r') as openfile:
    json_object = json.load(openfile)

word2embedding = gen_word2glove_dict(100)
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


#write to json file
with open('pfa_glove_dist_added.json', 'w') as outfile:
    json.dump(json_object, outfile, indent=2)
