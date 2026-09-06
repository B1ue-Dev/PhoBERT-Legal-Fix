from typing import List, Union
from pathlib import Path
from torch.utils.data import Dataset

from vphoberttagger.processor import NerFeatures
from vphoberttagger.constant import PROCESSOR_MAPPING

import os
import torch


class NerDataset(Dataset):
    def __init__(self, features: List[NerFeatures], device: str = 'cpu'):
        self.examples = features
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return {key: val.to(self.device) for key, val in self.examples[index].__dict__.items()}


def build_dataset(data_dir: Union[str, os.PathLike],
                  tokenizer,
                  label2id: List[str],
                  header: List[str],
                  dtype: str = 'train',
                  max_seq_len: int = 256,
                  device: str = 'cpu',
                  use_crf: bool = True,
                  overwrite_data: bool = False) -> NerDataset:
    data_path = Path(data_dir)
    # Support both train_data.txt and train.txt naming conventions
    dfile_candidates = [
        data_path / f"{dtype}_data.txt",
        data_path / f"{dtype}.txt"
    ]
    dfile_path = None
    for candidate in dfile_candidates:
        if candidate.exists():
            dfile_path = candidate
            break
            
    if dfile_path is None:
        raise FileNotFoundError(f"Could not find {dtype} dataset file in {data_dir}. Looked for {[str(c) for c in dfile_candidates]}")

    cached_path = dfile_path.with_suffix('.cached')
    process_key = tokenizer.name_or_path
    if not os.path.exists(cached_path) or overwrite_data:
        features = PROCESSOR_MAPPING[process_key](dfile_path, tokenizer, label2id, header, max_seq_len, use_crf=use_crf)
        torch.save(features, cached_path)
    else:
        features = torch.load(cached_path)
        
    return NerDataset(features=features, device=device)
