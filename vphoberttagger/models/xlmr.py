from transformers import XLMRobertaForTokenClassification
from torchcrf import CRF

from .base import NerOutput

import torch


class XLMRobertaCrf(XLMRobertaForTokenClassification):
    """XLM-RoBERTa encoder with a CRF decoding layer for word-level NER."""

    def __init__(self, config):
        super().__init__(config=config)
        self.num_labels = config.num_labels
        self.crf = CRF(config.num_labels, batch_first=True)
        self.init_weights()

    def _init_weights(self, module):
        """Initialize custom CRF parameters after HF weight materialization."""
        super()._init_weights(module)
        if isinstance(module, CRF):
            module.reset_parameters()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                valid_ids=None, label_masks=None):
        seq_outputs = self.roberta(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=None,
        )[0]
        if not torch.isfinite(seq_outputs).all():
            raise FloatingPointError("XLM-R encoder produced non-finite hidden states.")

        batch_size = seq_outputs.shape[0]
        range_vector = torch.arange(
            batch_size, dtype=torch.long, device=seq_outputs.device
        ).unsqueeze(1)
        seq_outputs = seq_outputs[range_vector, valid_ids]
        logits = self.classifier(self.dropout(seq_outputs))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("XLM-R classifier produced non-finite CRF emissions.")
        # Match the mask representation accepted by the installed torchcrf
        # implementation. Decoding accepts booleans, while its loss path
        # expects the legacy uint8 mask used by the PhoBERT implementation.
        decode_mask = label_masks != 0
        loss_mask = label_masks.type(torch.uint8)
        seq_tags = self.crf.decode(logits, mask=decode_mask)

        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=loss_mask)
            if not torch.isfinite(log_likelihood):
                lengths = loss_mask.sum(dim=1).detach().cpu().tolist()
                label_min = labels.min().detach().item()
                label_max = labels.max().detach().item()
                transitions_finite = torch.isfinite(self.crf.transitions).all().item()
                raise FloatingPointError(
                    "XLM-R CRF produced a non-finite likelihood "
                    f"(mask_lengths={lengths}, labels=[{label_min}, {label_max}], "
                    f"finite_transitions={transitions_finite})."
                )
            return NerOutput(loss=-log_likelihood, tags=seq_tags)
        return NerOutput(tags=seq_tags)
