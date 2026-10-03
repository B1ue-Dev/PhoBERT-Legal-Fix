"""
Script for chunking/NER evaluation
https://github.com/sighsmile/conlleval
"""

from __future__ import division, print_function, unicode_literals

import sys
from collections import defaultdict
from vphoberttagger.constant import LOGGER


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
    prec, rec, f1 = calc_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks, percent=False)
    res = (prec, rec, f1)
    if not verbose:
        return res

    # print overall performance, and performance per chunk type
    LOGGER.info(f"Processed {sum_true_counts} tokens with {sum_true_chunks} phrases; "
                f"Found: {sum_pred_chunks} phrases; correct: {sum_correct_chunks}.")
    LOGGER.info("Accuracy: %0.4f; (without `O` tag)" % (nonO_correct_counts / nonO_true_counts))
    LOGGER.info("Accuracy: %0.4f;  Precision: %0.4f; Recall: %0.4f; F1-score: %0.4f"
                % (sum_correct_counts / sum_true_counts, prec, rec, f1))
    # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
    for t in chunk_types:
        prec, rec, f1 = calc_metrics(correct_chunks[t], pred_chunks[t], true_chunks[t], percent=False)
        LOGGER.info("%17s: Precision: %0.4f; Recall: %0.4f; F1-score: %0.4f  %d" % (t, prec, rec, f1, pred_chunks[t]))
    return res


def evaluate(true_seqs, pred_seqs, verbose=True):
    correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, pred_counts = count_chunks(true_seqs,
                                                                                                      pred_seqs)
    result = get_result(correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, pred_counts,
                        verbose=verbose)
    return result


def evaluate_detailed(true_sequences, pred_sequences):
    """Return strict entity metrics suitable for a reproducible NER benchmark.

    ``conlleval`` evaluates chunks in a flat tag stream.  We therefore insert
    an ``O`` boundary after every sentence so that an entity cannot be counted
    as continuing into the next sentence.  This is the usual strict NER
    protocol: an entity is correct only when both its span and type match.
    """
    if len(true_sequences) != len(pred_sequences):
        raise ValueError("Gold and predicted sentence counts do not match.")

    true_tags, pred_tags = [], []
    for gold, pred in zip(true_sequences, pred_sequences):
        if len(gold) != len(pred):
            raise ValueError("Gold and predicted token counts do not match.")
        true_tags.extend(gold)
        pred_tags.extend(pred)
        true_tags.append('O')
        pred_tags.append('O')

    correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, pred_counts = count_chunks(
        true_tags, pred_tags
    )
    precision, recall, f1 = get_result(
        correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, pred_counts, verbose=False
    )
    entity_types = sorted(set(true_chunks) | set(pred_chunks))
    per_entity = {}
    for entity_type in entity_types:
        entity_precision, entity_recall, entity_f1 = calc_metrics(
            correct_chunks[entity_type], pred_chunks[entity_type], true_chunks[entity_type], percent=False
        )
        per_entity[entity_type] = {
            'precision': entity_precision,
            'recall': entity_recall,
            'f1-score': entity_f1,
            'support': true_chunks[entity_type],
            'predicted': pred_chunks[entity_type],
            'correct': correct_chunks[entity_type],
        }
    macro_f1 = sum(item['f1-score'] for item in per_entity.values()) / len(per_entity) if per_entity else 0.0
    return {
        'entity_strict_precision_micro': precision,
        'entity_strict_recall_micro': recall,
        'entity_strict_micro_f1': f1,
        'entity_strict_macro_f1': macro_f1,
        'per_entity': per_entity,
        'tokens': sum(true_counts.values()),
        'entities': sum(true_chunks.values()),
        'predicted_entities': sum(pred_chunks.values()),
        'correct_entities': sum(correct_chunks.values()),
    }


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


if __name__ == '__main__':
    """
    usage:     conlleval < file
    """
    evaluate_conll_file(sys.stdin)
