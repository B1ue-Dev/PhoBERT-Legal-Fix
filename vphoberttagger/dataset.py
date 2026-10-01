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

    process_key = tokenizer.name_or_path
    if process_key not in PROCESSOR_MAPPING:
        raise ValueError(f"Unsupported model '{process_key}'. Supported models: {list(PROCESSOR_MAPPING)}")
    cache_key = process_key.replace('/', '_').replace('\\', '_')
    architecture_key = 'crf' if use_crf else 'token'
    cached_path = dfile_path.with_suffix(f'.{cache_key}.{max_seq_len}.{architecture_key}.cached')
    if not os.path.exists(cached_path) or overwrite_data:
        features = PROCESSOR_MAPPING[process_key](dfile_path, tokenizer, label2id, header, max_seq_len, use_crf=use_crf)
        torch.save(features, cached_path)
    else:
        # Dataset caches contain NerFeatures instances created locally above.
        # PyTorch 2.6 defaults to weights_only=True, which rejects that class.
        features = torch.load(cached_path, weights_only=False)
        
    return NerDataset(features=features, device=device)
