import json
import os 
import argparse 
import pdb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

'''
WHERE DOES THE ERROR COME FROM?
    From the matching step? OR 
        We often can identify correct noun, verbs leave a little to be desired
        Closer look: bouqet --> arrange or "florist --> arrange" or "egg --> break" 
        It's matching non-verbs with verbs b/c that's what embeddings are close to. 
        Right now, captions are unconstrained, and matching does not enforce anything, sooo... we end up with poor matches
    From the caption step?
        Evaluate qualitatively - do a lot of captions make some sense? How many are just straight up wrong? 

How close are we when we are wrong? 

How many nouns are correct? - 0.477 we get correct noun appearing in caption + match 
    How often is the correct noun present in the processed caption? 
        If always gets correct noun when noun is present, we may have captioning issue
        If can't get correct noun when its present -- verbs may cause error, caption may have issues (with verbs)

How many verbs are wrong? How many closest to verbs are actually verbs? - 0.1176 we get correct verb appearing + match
    NOTE: There may have been closer words, but closest2noun/verb are two whose AVERAGE was smallest 
    verbs: Sometimes we get cut vs. chop vs. slice, OR hold vs. carry, OR mix vs. grind, but a lot of it is actually garbage
    e.g. field vs. bend, front vs. shut, close vs. pile

How do photos play a role?
    When evaluating on all, we get jump in accuracy, suggesting that photos play role, but there's a larger problem

Generate confusion matrix and calculate precision/recall

Captions describe current state of the world, not necessarily actions that got us there. 

'''

def metric_compute_correct_nouns(results):
    # pdb.set_trace()
    noun_correct = 0
    verb_correct = 0

    for result_id in results:
        if result_id == "accuracy":
            continue

        result = results[result_id]
        if result['closest2noun'] == result['label'][1]:
            noun_correct += 1

        if result['closest2verb'] == result['label'][0]:
            verb_correct += 1

    print("Proportion of nouns correct", noun_correct / len(results))
    print("Proportion of verbs correct", verb_correct / len(results))

def metric_compute_nouns_present(results):
    noun_present = 0
    verb_present = 0

    for result_id in results:
        if result_id == "accuracy":
            continue

        result = results[result_id]
        if result['label'][1] in result['caption_processed']:
            noun_present += 1

        if result['label'][0] in result['caption_processed']:
            verb_present += 1
    
    print("% with correct noun present", noun_present / len(results))
    print("% with correct verb present", verb_present / len(results))


def formulate_results_as_classification(results):
    class_dict = {} # verbnoun --> class number
    class_list = []

    # One pass to generate class numbers
    class_num = 0
    for result in results:
        if result == "accuracy":
            continue

        verbnoun = "".join(results[result]['label'])
        if verbnoun not in class_dict:
            class_dict[verbnoun] = class_num
            class_list.append(verbnoun)
            class_num += 1

    y_true, y_pred = [], []
    # One pass to generate y_true, y_pred 
    for result in results:
        if result == "accuracy":
            continue

        verbnoun_label = "".join(results[result]['label'])
        verbnoun_pred = "".join(results[result]['prediction'])
        y_true.append(class_dict[verbnoun_label])
        y_pred.append(class_dict[verbnoun_pred])
    
    return y_pred, y_true, class_list

def metric_calculate_pr(results, mode='precision'):
    # pdb.set_trace()
    y_pred, y_true, class_list = formulate_results_as_classification(results)
    cmat = confusion_matrix(y_true, y_pred) # Rows = true label, col = predicted label

    metrics = []
    for class_num in range(cmat.shape[0]):
        tp = cmat[class_num, class_num]
        fp = np.sum(cmat[np.arange(cmat.shape[0]) != class_num, class_num]) # samples of different class that you predicted this class for
        fn = np.sum(cmat[class_num, np.arange(cmat.shape[0]) != class_num]) # samples of this class that you predicted other class for

        if mode == "precision":
            if tp == 0 and fp == 0:
                metric = 0
            else:
                metric = tp / (tp + fp)
        elif mode == "recall":
            metric = tp / (tp + fn)

        metrics.append(metric)

    # ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=class_list)
    # plt.show()
    return np.mean(metrics)

def main(args):
    with open(args.result_json, 'r') as json_in:
        results = json.load(json_in)

    metric_compute_nouns_present(results)
    metric_compute_correct_nouns(results)
    map = metric_calculate_pr(results, mode='precision')
    mar = metric_calculate_pr(results, mode='recall')
    print("mAP", map)
    print("mAR", mar)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", type=str, help="json file of results")
    args= parser.parse_args()
    main(args)    
