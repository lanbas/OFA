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
        glove_dist = [0.0, 0.0]         #glove_dist[0]: distance between verbs
                                        #glove_dist[1]: distance between nouns
        for ind1 in range(len(labels)):
            dist = 0
            label_embedding = word2embedding[labels[ind1]]
            pred_embedding = word2embedding[preds[ind1]]
            for ind2 in range(len(label_embedding)):
                dist += (label_embedding[ind2] - pred_embedding[ind2])**2
            glove_dist[ind1] = math.sqrt(dist)
        d["glove_dist_verb"] = glove_dist[0]
        d["glove_dist_noun"] = glove_dist[1]
        if d["correct"] == 1:
            num_correct += 1
        #reevaluate correctness based on glove distance
        if glove_dist[0] <= 5.3 and glove_dist[1] <= 5:
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

    #add json to dictionary, sort dictionary by glove distance, print array to file
    verb_dist = {}
    noun_dist = {}
    both_dist = {}
    for item in json_object.items():
        d = item[1]
        if not type(d) is dict:
            continue
        v = d["label"][0] + " " + d["prediction"][0]
        n = d["label"][1] + " " + d["prediction"][1]
        b = d["label"][0] + " " + d["label"][1] + ", " + d["prediction"][0] + " " + d["prediction"][1]
        verb_dist[v] = d["glove_dist_verb"]
        noun_dist[n] = d["glove_dist_noun"]
        both_dist[b] = d["glove_dist_verb"] + d["glove_dist_noun"]

    f = open("verb_dists.txt", "a")
    for item in sorted(verb_dist, key=verb_dist.get):
        s = item + ": " + str(verb_dist[item])
        f.write(s)
    f.close()

    g = open("noun_dists.txt", "a")
    for item in sorted(noun_dist, key=noun_dist.get):
        s = item + ": " + str(noun_dist[item]) + "\n"
        g.write(s)
    g.close()

    h = open("both_dists.txt", "a")
    for item in sorted(both_dist, key=both_dist.get):
        s = item + ": " + str(both_dist[item]) + "\n"
        h.write(s)
    h.close()

add_glove_dist()