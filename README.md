# <div align="center">🏛️ PhoBERT-CRF cho PAP_NER</div>

<div align="center">

[![Paper: PLOS ONE](https://img.shields.io/badge/Paper-PLOS%20ONE-blue.svg)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0353166)
[![DOI: 10.1371/journal.pone.0353166](https://img.shields.io/badge/DOI-10.1371%2Fjournal.pone.0353166-brightgreen.svg)](https://doi.org/10.1371/journal.pone.0353166)
[![Dataset: Zenodo](https://img.shields.io/badge/Dataset-Zenodo-orange.svg)](https://doi.org/10.5281/zenodo.18044019)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

</div>

Mô hình học sâu kết hợp **PhoBERT (Semantic Engine)** và **CRF (Structural Engine)** để nhận diện thực thể có tên (NER) chuyên sâu cho thủ tục hành chính công và văn bản pháp luật tiếng Việt. Nghiên cứu và bộ dữ liệu được công bố chính thức trên tạp chí **PLOS ONE**: 

> 📄 **Paper:** [*PAP_NER: A large-scale vietnamese administrative named entity recognition corpus and hybrid deep learning architecture*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0353166)  
> 🔗 **DOI:** [10.1371/journal.pone.0353166](https://doi.org/10.1371/journal.pone.0353166)

---

## 🏗️ Kiến trúc mô hình (Dual-Engine Architecture)

```text
                  Input Sentence (Văn bản hành chính)
                               │
                               ▼
        Vietnamese Word Segmentation & BPE Subwords
                               │
                               ▼
               ┌───────────────────────────────┐
               │    Semantic Engine: PhoBERT   │  (vinai/phobert-base, 12 layers)
               │    Contextual Embeddings H    │  H ∈ ℝ^(n × 768)
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │    Linear / Emission Layer    │  (768 → 11 nhãn BIO)
               │    Emission Scores E          │  E ∈ ℝ^(n × 11)
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │   Structural Engine: CRF      │  (Học ma trận chuyển dịch A)
               │   Negative Log-Likelihood     │  (Forward Algorithm khi train)
               └───────────────┬───────────────┘
                               │
                               ▼
                     Viterbi Decoding (Inference)
                               │
                               ▼
                    Chuỗi nhãn BIO tối ưu
               (B-CQ, I-CQ, B-ĐT, B-VBPL, B-NG, B-SL, O)
```

---

## 📚 Bộ dữ liệu PAP_NER (`Dataset/`)

Dữ liệu được lưu trong thư mục `Dataset/` với cấu trúc:
```text
Dataset/
├── train_data.txt   (142,218 câu)
├── dev_data.txt     (10,305 câu)
├── test_data.txt    (10,278 câu)
├── docs.md          (Tài liệu mô tả chi tiết nhãn và thống kê)
└── img/             (Bảng thống kê Table 2 & Table 3)
```

### Hệ thống 11 nhãn thực thể BIO:
* **CQ (Cơ quan):** `B-CQ`, `I-CQ` (*UBND tỉnh, Sở Kế hoạch...*)
* **ĐT (Đối tượng):** `B-ĐT`, `I-ĐT` (*doanh nghiệp, công dân...*)
* **VBPL (Văn bản pháp luật):** `B-VBPL`, `I-VBPL` (*Nghị định 123/2020/NĐ-CP...*)
* **NG (Ngày giờ):** `B-NG`, `I-NG` (*15/03/2024, ngày 15 tháng 10 năm 2020...*)
* **SL (Số lượng):** `B-SL`, `I-SL` (*10 ngày, 5 triệu đồng, 100%...*)
* **O (Outside):** `O`

---

## 🚀 Cài đặt & Khởi chạy

### 1. Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

### 2. Huấn luyện (Training):
Lệnh chạy fine-tuning **PhoBERT-CRF** trên tập dữ liệu PAP_NER:
```bash
python main.py train \
    --task pap_ner \
    --data_dir ./Dataset \
    --model_name_or_path vinai/phobert-base \
    --model_arch crf \
    --output_dir outputs \
    --max_seq_length 256 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --classifier_learning_rate 1e-3 \
    --epochs 20 \
    --early_stop 3 \
    --run_test
```
*(Hoặc chạy script: `bash ./train.sh`)*

### Benchmark XLM-RoBERTa

XLM-R uses the same PAP_NER train/dev/test split, but requires a separate
word-aligned cache. The training command below writes independent benchmark
artifacts and leaves the PhoBERT experiment untouched:

```bash
python main.py train \
    --task pap_ner \
    --data_dir ./Dataset \
    --model_name_or_path xlm-roberta-base \
    --model_arch crf \
    --output_dir outputs/xlmr-base \
    --max_seq_length 256 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 2e-5 \
    --classifier_learning_rate 1e-4 \
    --epochs 4 \
    --early_stop 3 \
    --run_test
```

The final test run writes `best_model.pt`, `classification_report.json`, and
`confusion_matrix.png` under the selected output directory. Model selection is
performed on `dev_data.txt`; `test_data.txt` is reserved for final reporting.

### PAP_NER benchmark outputs

Every `test` run also creates these comparison-friendly artifacts in the same
directory:

- `benchmark_summary.md`: the headline **strict entity Micro F1** and Macro F1.
- `benchmark_results.json`: machine-readable model metadata, strict entity
  metrics, BIO metrics, and per-entity results.
- `entity_metrics.csv` and `entity_f1.png`: precision, recall, F1, and support
  for `CQ`, `ĐT`, `VBPL`, `NG`, and `SL`.

For a fair PhoBERT-CRF versus XLM-R-CRF comparison, use the same
`train_data.txt` / `dev_data.txt` / `test_data.txt`, seed, maximum sequence
length, and training budget. Hardware may differ; record it, but compare the
final test strict entity Micro F1 rather than training speed.

### 3. Đánh giá trên tập Test (Testing):
```bash
python main.py test --data_dir ./Dataset --model_path outputs/best_model.pt
```

### 4. Dự đoán câu mới (Inference):
```bash
python main.py predict --model_path outputs/best_model.pt
```

### 5. Giao diện Web Demo (Streamlit):
```bash
python main.py demo --model_path outputs/best_model.pt
```

---

## 📖 Tài liệu kỹ thuật chi tiết (Documentation)

Hệ thống tài liệu chuyên sâu được lưu trữ tại thư mục [`docs/`](docs/):

| Nhóm tài liệu | File | Nội dung chính |
| :--- | :--- | :--- |
| **Kiến trúc hệ thống** | [`docs/model-architecture.md`](docs/model-architecture.md) | Phân tích chi tiết kiến trúc và nguyên lý hoạt động của mô hình |
| | [`docs/dual-engine-design.md`](docs/dual-engine-design.md) | Cơ chế phối hợp Dual-Engine: Semantic Engine + Structural Engine |
| | [`docs/architecture-overview.md`](docs/architecture-overview.md) | Sơ đồ luồng dữ liệu từ câu đầu vào đến chuỗi BIO |
| **Động cơ cốt lõi** | [`docs/semantic-engine.md`](docs/semantic-engine.md) | Cơ chế Self-Attention và trích xuất ngữ cảnh của PhoBERT |
| | [`docs/crf-mechanism.md`](docs/crf-mechanism.md) | Chi tiết tầng CRF: Transition Matrix, Forward Algorithm & Viterbi |
| **Quy trình dữ liệu** | [`docs/pipeline-workflow.md`](docs/pipeline-workflow.md) | Quy trình End-to-End từ tokenization đến inference |
| | [`docs/system-workflow.md`](docs/system-workflow.md) | Diễn giải luồng chạy toàn diện của hệ thống |
| | [`docs/input-formatting.md`](docs/input-formatting.md) | Định dạng input CoNLL, xử lý subwords và nhãn BIO |
| **Thực nghiệm & Dữ liệu** | [`docs/experiments-and-results.md`](docs/experiments-and-results.md) | Báo cáo kết quả huấn luyện, ma trận nhầm lẫn & metric F1/Accuracy |
| | [`Dataset/docs.md`](Dataset/docs.md) | Báo cáo cấu trúc và thống kê bộ ngữ liệu PAP_NER |

---

## 📑 Trích dẫn khoa học (Citation)

Nếu bạn sử dụng mã nguồn, kiến trúc mô hình hoặc bộ dữ liệu **PAP_NER** trong các công trình nghiên cứu khoa học, vui lòng trích dẫn bài báo chính thức:

```bibtex
@article{pap_ner_2026,
  title     = {PAP\_NER: A large-scale vietnamese administrative named entity recognition corpus and hybrid deep learning architecture},
  author    = {La, Dinh-Dien and Tran, Tien-Bang and Du, Ngoc-Huy and Dang, Ngoc-Hung and Phung, Trung-Nghia and Tran, Van-Khanh},
  journal   = {PLOS ONE},
  volume    = {21},
  number    = {7},
  pages     = {e0353166},
  year      = {2026},
  publisher = {Public Library of Science},
  doi       = {10.1371/journal.pone.0353166},
  url       = {https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0353166}
}
```

