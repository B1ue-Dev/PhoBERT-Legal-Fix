from typing import Union, List
from tqdm import tqdm

import os
import torch
import pandas as pd
import numpy as np


class NerFeatures(object):
    def __init__(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None, labels=None, label_masks=None, **kwargs):
        self.input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        self.token_type_ids = (torch.as_tensor(token_type_ids, dtype=torch.long)
                               if token_type_ids is not None
                               else torch.zeros_like(self.input_ids))
        self.attention_mask = (torch.as_tensor(attention_mask, dtype=torch.long)
                               if attention_mask is not None
                               else torch.ones_like(self.input_ids))
        self.valid_ids = torch.as_tensor(valid_ids, dtype=torch.long) if valid_ids is not None else None
        self.labels = torch.as_tensor(labels, dtype=torch.long) if labels is not None else None
        self.label_masks = torch.as_tensor(label_masks, dtype=torch.long) if label_masks is not None else None


def convert_word_segment_examples_features(data_path: Union[str, os.PathLike],
                                          tokenizer,
                                          label2id,
                                          header_names: List[str],
                                          max_seq_len: int = 256,
                                          use_crf: bool = True) -> List[NerFeatures]:
    """
    Process Vietnamese word-segmented dataset into PhoBERT features with subword alignment.
    Supports whitespace-delimited (spaces or tabs) CoNLL format files.
    """
    features = []
    tokens = []
    tag_ids = []
    
    data = pd.read_csv(data_path,
                       sep=r'\s+',
                       encoding='utf-8',
                       skip_blank_lines=False,
                       names=header_names)
    
    for row_idx, row in tqdm(data.iterrows(), total=len(data), desc=f"Load dataset {data_path}..."):
        if row.notna().token:
            tokens.append(row.token.strip().replace(' ', '_'))
            tag_ids.append(label2id.index(row.ner.strip()))
            if not row_idx == len(data) - 1:
                continue
                
        seq_len = len(tokens)
        if seq_len == 0:
            continue
            
        sentence = ' '.join(tokens)
        encoding = tokenizer(sentence,
                             padding='max_length',
                             truncation=True,
                             max_length=max_seq_len)
        subwords = tokenizer.tokenize(sentence)
        valid_ids = np.zeros(len(encoding.input_ids), dtype=int)
        label_marks = np.zeros(len(encoding.input_ids), dtype=int)
        valid_labels = np.ones(len(encoding.input_ids), dtype=int) * -100
        i = 1
        for idx, subword in enumerate(subwords[:max_seq_len - 2]):
            if idx != 0 and subwords[idx - 1].endswith("@@"):
                continue
            if use_crf:
                valid_ids[i - 1] = idx + 1
            else:
                valid_ids[idx + 1] = 1
            valid_labels[idx + 1] = tag_ids[i - 1]
            i += 1
            
        if max_seq_len >= seq_len:
            label_padding_size = (max_seq_len - seq_len)
            label_marks[:seq_len] = [1] * seq_len
            tag_ids.extend([0] * label_padding_size)
        else:
            tag_ids = tag_ids[:max_seq_len]
            label_marks[:-2] = [1] * (max_seq_len - 2)
            tag_ids[-2:] = [0] * 2
            
        if use_crf and label_marks[0] == 0:
            raise ValueError(f"{sentence} - {tag_ids} has label_mark == 0 at index 0!")

        items = {key: val for key, val in encoding.items()}
        if 'token_type_ids' not in items:
            items['token_type_ids'] = [0] * max_seq_len
        items['labels'] = tag_ids if use_crf else valid_labels
        items['valid_ids'] = valid_ids
        items['label_masks'] = label_marks if use_crf else valid_ids
        features.append(NerFeatures(**items))

        for k, v in items.items():
            assert len(v) == max_seq_len, f"Expected length of {k} is {max_seq_len} but got {len(v)}"

        tokens = []
        tag_ids = []
        
    return features
