import numpy as np
import pdb
import json
import csv
import sys
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

csv.field_size_limit(sys.maxsize)

def ae_words2embeddings(ae_words, word2embedding):
    ae_embeddings = []
    for a, e in ae_words:
        if a in word2embedding:
            a_embed = word2embedding[a]
        else:
            print(f"NO EMBEDDING FOR ACTION {a}")
            exit(1)
        
        if e in word2embedding:
            e_embed = word2embedding[e]
        else:
            print(f"NO EMBEDDING FOR EFFECT {e}")
            exit(1)
        
        ae_embeddings.append([a_embed, e_embed])

    return ae_embeddings

def gen_word2glove_dict(embedding_dim):
    word2embedding = {}
    with open(f'./glove.6B.{embedding_dim}d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float32)
            word2embedding[word] = vect
    print("GloVe vectors loaded!")

    return word2embedding

def predict_label(caption, ae_embeddings, ae_words, word2embedding, label_idx):
    # preprocess caption

    # Convert caption words to embeddings
    caption_embeddings = [] 
    for w in caption.split():
        if w in word2embedding:
            caption_embeddings.append(word2embedding[w])
    caption_embeddings = np.array(caption_embeddings) # (N, 100), N words in caption

    # Find closest 2 action
    # pdb.set_trace()

    caption_words = caption.split()
    min_score = 1000000000
    best_idx = -1
    closest2a, closest2e = None, None
    scores = []
    for i, (a, e) in enumerate(ae_embeddings): # a = (100,) e = (100,)
        dist2a = np.sum((caption_embeddings - a)**2, axis=1) # (N,)
        min2a_idx = np.argmin(dist2a) # 1xN
        min2a_dist = dist2a[min2a_idx]

        dist2e = np.sum((caption_embeddings - e)**2, axis=1)
        min2e_idx = np.argmin(dist2e) # 1xN
        min2e_dist = dist2e[min2e_idx]

        score = (min2a_dist + min2e_dist) / 2
        scores.append(score)
        if score < min_score:
            closest2a = caption_words[min2a_idx]
            closest2e = caption_words[min2e_idx]
            min_score = score
            best_idx = i
    
    # pdb.set_trace()
    if best_idx == label_idx:
        score_diff_to_correct = 0
    else:
        score_diff_to_correct = scores[label_idx] - scores[best_idx]
    top_5 = np.array(ae_words)[np.argsort(scores)[:5]]
    
    return ae_words[best_idx], top_5, closest2a, closest2e, score_diff_to_correct, min_score

def get_ae_words(ae_gt):
    ae_words = []
    for obj in ae_gt:
        ae_words.append(list(obj.values())[0])
    return ae_words
    
def get_ae_gt(dataset_file):
    csv_reader = csv.reader(open(dataset_file, 'r'), delimiter='\t')

    # row = uniqid, imgid, caption, garbage, imgbase64
    gt = {}
    for row in csv_reader:
        img_id, ae = row[1], row[2]
        gt[img_id] = ae.split()

    return gt

def preprocess_caption(caption):
    # pdb.set_trace()
    caption_tokens = word_tokenize(caption)
    pos_tags = pos_tag(caption_tokens)
    lemmatized_caption = []
    for word, tag in pos_tags:
        if word in stopwords.words():
            continue

        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        if wntag is not None:
            lemmatized_caption.append(lemmatizer.lemmatize(word, pos=wntag))
        else:
            lemmatized_caption.append(word)

    return " ".join(lemmatized_caption)


def main():
    word2embedding = gen_word2glove_dict(100)
    dataset_file = "../dataset/caption_data/dataset1_full_val.tsv"
    ae_caption_list = json.load(open('ae_list.json', 'r'))
    ae_words = get_ae_words(ae_caption_list)
    ae_gt = get_ae_gt(dataset_file) # {img_id: ['verb', 'noun']}
    ae_embeddings = np.array(ae_words2embeddings(ae_words, word2embedding))
    # pdb.set_trace()

    pred_file = '../results/caption/baseline_1.json'
    pred_list = json.load(open(pred_file, 'r')) # [{image_id: "00", "caption": "predicted caption"}, ...]

    correct = 0
    top5_correct = 0
    results_dict = {}
    score_diffs = []
    match_scores = []
    for pred in pred_list:
        pred_caption = pred['caption']
        pred_caption_proc = preprocess_caption(pred_caption)
        print(pred_caption_proc)
        # pdb.set_trace()
        label = ae_gt[pred['image_id']]
        label_idx = ae_words.index(label)
        
        if len(pred_caption_proc) == 0: # If all stop words
            pred_nv, top5_nv, closest2verb, closest2noun, score_diff, min_score = predict_label(pred_caption, ae_embeddings, ae_words, word2embedding, label_idx)
        else: 
            pred_nv, top5_nv, closest2verb, closest2noun, score_diff, min_score = predict_label(pred_caption_proc, ae_embeddings, ae_words, word2embedding, label_idx)

        results_dict[pred['image_id']] = {"caption": pred_caption,
                                          "caption_processed": pred_caption_proc,
                                          "label": label,
                                          "prediction": pred_nv,
                                          "closest2verb": closest2verb,
                                          "closest2noun": closest2noun,
                                          "correct": int(label == pred_nv),
                                          "top5_correct": int(label in top5_nv)}

        match_scores.append(min_score)
        if score_diff != 0:
            score_diffs.append(score_diff)

        if label == pred_nv:
            correct += 1

        if label in top5_nv:
            top5_correct += 1

    results_dict['accuracy'] = correct / len(pred_list)
    results_dict['top5_accuracy'] = top5_correct / len(pred_list)
    results_dict['score_diff_for_incorrect'] = np.mean(score_diffs)
    results_dict['avg_dist_to_match'] = np.mean(match_scores)

    with open(pred_file.replace(".json", "_results.json"), 'w') as out_ptr:
        json.dump(results_dict, out_ptr, indent=2)

    print(f"ACCURACY = {results_dict['accuracy']}")
    print(f"TOP 5 ACCURACY = {results_dict['top5_accuracy']}")
    print(f"SCORE DIFF FOR INCORRECT = {results_dict['score_diff_for_incorrect']}")
    print(f"SCORE DIFF FOR INCORRECT = {results_dict['avg_dist_to_match']}")

main()

'''
TODO FOR THIS STILL -- WHERE DOES THE ERROR COME FROM?
    From the matching step? OR 
    From the caption step?

How close are we when we are wrong? 
How many nouns are correct? 
How many verbs are wrong? How many closest to verbs are actually verbs? 
    NOTE: There may have been closer words, but closest2noun/verb are two whose AVERAGE was smallest 

How do photos play a role?
    When evluating on all, we get jump in accuracy, suggesting that photos play role, but there's a larger problem

cut chop AND slice onion??? 

'''

# Distance metric
    # Given word embeddings for caption C, action-effect embeddings A, E
        # Find closest embedding in C to A, keep as dist_a
        # Find closest embedding in C to E, keep as dist_e
        # Distance score for this (A, E) pair is avg(dist_a, dist_e)
    # A-E pair with lowest distance score is predicted class