"""
This script applies to IOB2 or IOBES tagging scheme.
If you are using a different scheme, please convert to IOB2 or IOBES.

IOB2:
- B = begin, 
- I = inside but not the first, 
- O = outside

e.g. 
John   lives in New   York  City  .
B-PER  O     O  B-LOC I-LOC I-LOC O

IOBES:
- B = begin, 
- E = end, 
- S = singleton, 
- I = inside but not the first or the last, 
- O = outside

e.g.
John   lives in New   York  City  .
S-PER  O     O  B-LOC I-LOC E-LOC O

prefix: IOBES
chunk_type: PER, LOC, etc.
"""
from __future__ import division, print_function, unicode_literals

import sys
import os
from collections import defaultdict

def split_tag(chunk_tag):
    """
    split chunk tag into IOBES prefix and chunk_type
    e.g. 
    B-PER -> (B, PER)
    O -> (O, None)
    """
    if chunk_tag == 'O':
        return ('O', None)
    return chunk_tag.split('-', maxsplit=1)

def is_chunk_end(prev_tag, tag):
    """
    check if the previous chunk ended between the previous and current word
    e.g. 
    (B-PER, I-PER) -> False
    (B-LOC, O)  -> True

    Note: in case of contradicting tags, e.g. (B-PER, I-LOC)
    this is considered as (B-PER, B-LOC)
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == 'O':
        return False
    if prefix2 == 'O':
        return prefix1 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']

def is_chunk_start(prev_tag, tag):
    """
    check if a new chunk started between the previous and current word
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == 'O':
        return False
    if prefix1 == 'O':
        return prefix2 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']


def calc_metrics(tp, p, t, percent=True):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = tp / p if p else 0
    recall = tp / t if t else 0
    fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * fb1
    else:
        return precision, recall, fb1


def count_chunks(true_seqs, pred_seqs):
    """
    true_seqs: a list of true tags
    pred_seqs: a list of predicted tags

    return: 
    correct_chunks: a dict (counter), 
                    key = chunk types, 
                    value = number of correctly identified chunks per type
    true_chunks:    a dict, number of true chunks per type
    pred_chunks:    a dict, number of identified chunks per type

    correct_counts, true_counts, pred_counts: similar to above, but for tags
    """
    correct_chunks = defaultdict(int)
    true_chunks = defaultdict(int)
    pred_chunks = defaultdict(int)

    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    prev_true_tag, prev_pred_tag = 'O', 'O'
    correct_chunk = None

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        if true_tag == pred_tag:
            correct_counts[true_tag] += 1
        true_counts[true_tag] += 1
        pred_counts[pred_tag] += 1

        _, true_type = split_tag(true_tag)
        _, pred_type = split_tag(pred_tag)

        if correct_chunk is not None:
            true_end = is_chunk_end(prev_true_tag, true_tag)
            pred_end = is_chunk_end(prev_pred_tag, pred_tag)

            if pred_end and true_end:
                correct_chunks[correct_chunk] += 1
                correct_chunk = None
            elif pred_end != true_end or true_type != pred_type:
                correct_chunk = None

        true_start = is_chunk_start(prev_true_tag, true_tag)
        pred_start = is_chunk_start(prev_pred_tag, pred_tag)

        if true_start and pred_start and true_type == pred_type:
            correct_chunk = true_type
        if true_start:
            true_chunks[true_type] += 1
        if pred_start:
            pred_chunks[pred_type] += 1

        prev_true_tag, prev_pred_tag = true_tag, pred_tag
    if correct_chunk is not None:
        correct_chunks[correct_chunk] += 1

    return (correct_chunks, true_chunks, pred_chunks, 
        correct_counts, true_counts, pred_counts)

def get_result(correct_chunks, true_chunks, pred_chunks,
    correct_counts, true_counts, pred_counts, verbose=True):
    """
    if verbose, print overall performance, as well as preformance per chunk type;
    otherwise, simply return overall prec, rec, f1 scores
    """
    # sum counts
    sum_correct_chunks = sum(correct_chunks.values())
    sum_true_chunks = sum(true_chunks.values())
    sum_pred_chunks = sum(pred_chunks.values())

    sum_correct_counts = sum(correct_counts.values())
    sum_true_counts = sum(true_counts.values())

    nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != 'O')
    nonO_true_counts = sum(v for k, v in true_counts.items() if k != 'O')

    chunk_types = sorted(list(set(list(true_chunks) + list(pred_chunks))))

    # compute overall precision, recall and FB1 (default values are 0.0)
    prec, rec, f1 = calc_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks)
    res = (prec, rec, f1)
    if not verbose:
        return res

    # print overall performance, and performance per chunk type
    result_str = ""
    # print("processed %i tokens with %i phrases; " % (sum_true_counts, sum_true_chunks), end='')
    result_str += "processed %i tokens with %i phrases; " % (sum_true_counts, sum_true_chunks)
    # print("found: %i phrases; correct: %i.\n" % (sum_pred_chunks, sum_correct_chunks), end='')
    result_str += "found: %i phrases; correct: %i.\n" % (sum_pred_chunks, sum_correct_chunks)
        
    if nonO_true_counts == 0:
        nonO_true_counts = 0.000001
    # print("accuracy: %6.2f%%; (non-O)" % (100*nonO_correct_counts/nonO_true_counts))
    # print("accuracy: %6.2f%%; " % (100*sum_correct_counts/sum_true_counts), end='')
    # print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" % (prec, rec, f1)) 

    # result_str += "accuracy: %6.2f%%; (non-O)\n" % (100*nonO_correct_counts/nonO_true_counts)
    result_str += "accuracy: %6.2f%%; \n" % (100*sum_correct_counts/sum_true_counts)
    result_str += "precision: %6.2f%%;\nrecall: %6.2f%%;\nF1: %6.2f\n" % (prec, rec, f1)

    # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
    for t in chunk_types:
        prec, rec, f1 = calc_metrics(correct_chunks[t], pred_chunks[t], true_chunks[t])
        # print("%17s: " %t , end='')
        # print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
        #             (prec, rec, f1), end='')
        # print("  %d" % pred_chunks[t])

        result_str += "%17s: " %t 
        result_str += "prec: %6.2f%%; rec: %6.2f%%; FB1: %6.2f" %(prec, rec, f1)
        result_str += "  %d\n" % pred_chunks[t]

    return res, result_str
    # you can generate LaTeX output for tables like in
    # http://cnts.uia.ac.be/conll2003/ner/example.tex
    # but I'm not implementing this

def evaluate(true_seqs, pred_seqs, verbose=True):
    (correct_chunks, true_chunks, pred_chunks,
        correct_counts, true_counts, pred_counts) = count_chunks(true_seqs, pred_seqs)
    result, result_str = get_result(correct_chunks, true_chunks, pred_chunks,
        correct_counts, true_counts, pred_counts, verbose=verbose)
    return result, result_str

def evaluate_conll_file(fileIterator):
    true_seqs, pred_seqs = [], []
    
    for line in fileIterator:
        cols = line.strip().split()
        # each non-empty line must contain >= 3 columns
        if not cols:
            true_seqs.append('O')
            pred_seqs.append('O')
        elif len(cols) < 3:
            raise IOError("conlleval: too few columns in line %s\n" % line)
        else:
            # extract tags from last 2 columns
            true_seqs.append(cols[-2])
            pred_seqs.append(cols[-1])
    return evaluate(true_seqs, pred_seqs)


def evaluate_files(file_true, file_pred):
    """
    file_true has two columns: word and true label
    file_pred has only one column maybe?: pred label
    """
    true_seqs, pred_seqs = [], []
    
    with open(file_true, 'r') as fileIterator:
        for line in fileIterator:
            cols = line.strip().split()
            # each non-empty line must contain >= 2 columns
            if not cols:
                true_seqs.append('O')
            else:
                # extract tags from last 2 columns
                true_seqs.append(cols[-1])
                
    with open(file_pred, 'r') as fileIterator:
        for line in fileIterator:
            cols = line.strip().split()
            # each non-empty line must contain >= 3 columns
            if not cols:
                pred_seqs.append('O')
            else:
                # extract tags from last 2 columns
                pred_seqs.append(cols[-1])
    
    return evaluate(true_seqs, pred_seqs)

def evaluate_dir(input_dir, output_dir):
    """
    file_true has two columns: word and true label
    file_pred has only one column maybe?: pred label
    """
    submit_dir = os.path.join(input_dir, 'res') 
    truth_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        print (submit_dir + " doesn't exist")
    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_filename = os.path.join(output_dir, 'scores.txt')              
        output_file = open(output_filename, 'wb')

        gold_list = os.listdir(truth_dir)
        for gold in gold_list:
            gold_file = os.path.join(truth_dir, gold)
            corresponding_submission_file = os.path.join(submit_dir, gold)
            if os.path.exists(corresponding_submission_file):
                return evaluate_files(gold_file, corresponding_submission_file)
    
     

if __name__ == '__main__':
    """
    usage:     conlleval  test_file predict_file
    """
#    evaluate_conll_file(sys.stdin)

    # test_file = sys.argv[1]
    # predict_file = sys.argv[2]
    # print('input file ' : test_file)
    # print('predict file' : predict_file)
    #evaluate_files("../dataset/newseye_test.data", "../dataset/example_pred_empty.data")
    
    print(sys.argv[1])
    print(sys.argv[2])
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]


    #res, result_str = evaluate_dir(input_dir, output_dir)
    res, result_str = evaluate_files(input_dir  , output_dir)
    print(result_str)



#    
# generate empty pred
#    with open("../dataset/example_pred.data", 'r') as f:
#        lines = f.readlines()
#    
#    with open("../dataset/example_pred_empty.data", 'w') as f:
#    
#        for line in lines:
#            cols = line.strip().split()
#            # each non-empty line must contain >= 2 columns
#            print(len(cols))
#            if not cols:
#                f.write('\n')
#            else:
#                # extract tags from last 2 columns
#                f.write(cols[-1] + '\n')
#            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
