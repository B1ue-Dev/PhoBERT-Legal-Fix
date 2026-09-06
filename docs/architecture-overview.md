kiến trúc PhoBERT-CRF :   
                 Input sentence
                       │
                       ▼
              ┌─────────────────┐
              │     PhoBERT     │
              │ Semantic Engine │
              └────────┬────────┘
                       │
                  contextual
                  embeddings
                       │
                       ▼
                 Linear Layer
                       │
                 emission scores
                       │
                       ▼
              ┌─────────────────┐
              │      CRF        │
              │Structural Engine│
              └────────┬────────┘
                       │
                Viterbi decoding
                       │
                       ▼
                  BIO sequence

PhoBERT hiểu ngữ nghĩa.

CRF quyết định chuỗi BIO nào hợp lý nhất. 
5. Linear layer làm gì?

Sau PhoBERT:

hᵢ ∈ R⁷⁶⁸

được đưa vào Linear:

Linear(768 → |Y|)

Nếu paper có 11 labels:

Linear(768 → 11)

Kết quả:

eᵢ = W hᵢ + b

với:

eᵢ ∈ R¹¹

Đây gọi là:

Emission scores

Ví dụ:

Token = "Nghị"

B-VBPL     4.8
I-VBPL     1.2
B-CQ       0.3
I-CQ       0.1
O          0.7
...

Softmax có thể chọn:

B-VBPL

nhưng CRF sẽ đi xa hơn.