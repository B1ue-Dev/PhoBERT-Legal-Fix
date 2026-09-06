from transformers import logging, RobertaForTokenClassification
from torchcrf import CRF

from .base import NerOutput

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.set_verbosity_error()


class PhoBertCrf(RobertaForTokenClassification):
    """
    PhoBERT + CRF Architecture for Vietnamese Named Entity Recognition (NER)
    
    Pipeline:
      1. Input tokens (word-segmented) -> PhoBERT Tokenizer (BPE subwords)
      2. PhoBERT Encoder -> Contextual representations H in R^(T x 768) (aligned via valid_ids)
      3. Dropout + Linear Classifier -> Emission scores (768 -> N_tags)
      4. CRF Layer -> Transition scores & Negative Log-Likelihood Loss
      5. Viterbi Decoding -> Optimal BIO entity tag sequence
    """
    def __init__(self, config):
        super(PhoBertCrf, self).__init__(config=config)
        self.num_labels = config.num_labels
        self.crf = CRF(config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                label_masks=None):
        # 1. PhoBERT contextual representations
        seq_outputs = self.roberta(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask,
                                   head_mask=None)[0]

        # 2. Subword alignment to word-level representations H in R^(T x 768)
        batch_size, max_len, feat_dim = seq_outputs.shape
        range_vector = torch.arange(0, batch_size, dtype=torch.long, device=seq_outputs.device).unsqueeze(1)
        seq_outputs = seq_outputs[range_vector, valid_ids]
        
        # 3. Dropout + Linear emission layer: 768 -> N_tags
        seq_outputs = self.dropout(seq_outputs)
        logits = self.classifier(seq_outputs)  # Emission scores

        # 4. Viterbi decoding via CRF
        seq_tags = self.crf.decode(logits, mask=label_masks != 0)

        # 5. Training loss (Negative Log-Likelihood)
        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=label_masks.type(torch.uint8))
            return NerOutput(loss=-1.0 * log_likelihood, tags=seq_tags)
        else:
            return NerOutput(tags=seq_tags)
