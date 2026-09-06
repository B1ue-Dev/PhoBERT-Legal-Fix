1. Tổng thể kiến trúc

Có thể đọc Hình 1 như sau:

              INPUT
                │
                ▼
       ┌─────────────────┐
       │  Stage 1        │
       │  PhoBERT        │
       │ Semantic Engine │
       └────────┬────────┘
                │
                ▼
       Contextual embeddings
                │
                ▼
       ┌─────────────────┐
       │  Stage 2        │
       │ Linear / Emission│
       │     Layer       │
       └────────┬────────┘
                │
                ▼
          Emission scores
                │
                ▼
       ┌─────────────────┐
       │  Stage 3        │
       │      CRF        │
       │ Structural      │
       │     Engine      │
       └────────┬────────┘
                │
                ▼
          BIO predictions

Có thể hiểu:

PhoBERT hiểu token đang nói về cái gì → Linear chuyển representation thành điểm cho từng label → CRF chọn chuỗi label hợp lý nhất.

2. Stage 1 — Semantic Engine: PhoBERT

Đây là phần Transformer.

Input:

Nghị định 123/2020/NĐ-CP được ban hành

được đưa qua PhoBERT:

Tokens
  ↓
Embedding
  ↓
Transformer layers
  ↓
Contextual representations

Ví dụ:

h₁ = representation("Nghị_định")
h₂ = representation("123/2020")
h₃ = representation("NĐ-CP")
...

Nhưng h₁ không chỉ chứa thông tin của "Nghị_định".

Nó đã được contextualized bởi các token xung quanh.

3. Tác giả nói PhoBERT học 3 loại thông tin
A. Lexical semantics

Ví dụ:

ủy ban + nhân dân

PhoBERT hiểu đây là một cụm có ý nghĩa hành chính:

Ủy ban nhân dân
       ↓
      CQ

Thay vì coi:

ủy ban → một danh từ
nhân dân → một danh từ

một cách độc lập.

B. Long-range dependencies

Ví dụ:

Nghị định ............... NĐ-CP
   ↑                         ↑
 position i              position i+3

PhoBERT có self-attention nên có khả năng liên kết các token ở xa nhau.

Đây là điểm quan trọng đối với:

Nghị định số 123/2020/NĐ-CP

vì entity VBPL có thể kéo dài nhiều token.

C. Contextual disambiguation

Ví dụ từ:

cá nhân

có thể xuất hiện trong nhiều ngữ cảnh.

PhoBERT sử dụng context:

Theo quy định, cá nhân phải...

để xác định vai trò của "cá nhân" trong câu.

Đây là lý do họ gọi PhoBERT là:

Semantic Engine

4. Sau PhoBERT là gì?

Đây là chỗ nhiều người dễ bỏ qua.

Không phải:

PhoBERT → CRF

trực tiếp theo nghĩa representation đi thẳng vào CRF.

Thông thường pipeline là:

PhoBERT
   ↓
hidden states
   ↓
Linear classifier
   ↓
emission scores
   ↓
CRF

Ví dụ PhoBERT tạo:

H ∈ R^(n × hidden_size)

với PhoBERT-base:

hidden_size = 768

Sau đó Linear:

768 → 11

vì PAP_NER có 11 BIO labels.

Ta nhận được:

Emission ∈ R^(n × 11)

Ví dụ:

                O     B-CQ   I-CQ   B-VBPL ...
Nghị_định      0.2    0.1    0.3    8.7
123/2020       0.1    0.1    0.2    7.9
NĐ-CP          0.1    0.1    0.3    8.5

Đây là emission scores.

5. Stage 3 — Structural Engine: CRF

Đây là phần tác giả nhấn mạnh nhất.

CRF không chỉ hỏi:

"Token này là B-VBPL hay B-CQ?"

mà hỏi:

“Toàn bộ chuỗi label nào là hợp lý nhất?”

Ví dụ:

B-VBPL → I-VBPL → I-VBPL → O

CRF đánh giá cả sequence.

6. “BIO validity” nghĩa là gì?

Ví dụ model Softmax có thể dự đoán:

O
I-VBPL
B-CQ

Nhưng:

O → I-VBPL

không hợp lệ theo BIO.

Vì I-VBPL phải tiếp nối một entity VBPL.

Đúng phải là:

B-VBPL → I-VBPL

hoặc:

B-VBPL → I-VBPL → I-VBPL

CRF học transition giữa các labels để ưu tiên sequence hợp lý.

7. “Entity coherence” là gì?

Giả sử model dự đoán:

Nghị định
123/2020
NĐ-CP

Đây phải là:

B-VBPL
I-VBPL
I-VBPL

Không nên:

B-VBPL
I-VBPL
I-CQ

Tức là giữa một entity không nên đột nhiên đổi:

VBPL → CQ

CRF giúp mô hình học được pattern này.

8. “Boundary detection” là gì?

CRF cũng học các pattern về điểm bắt đầu/kết thúc entity.

Ví dụ:

Nghị định số 123/2020/NĐ-CP về thủ tục hành chính
^^^^^^^^^^^^^^^^^^^^^^^^
       VBPL

Entity có thể kết thúc trước:

về

nên sequence:

B-VBPL
I-VBPL
I-VBPL
O
O
...

CRF có thể học các transition/boundary pattern xuất hiện thường xuyên trong PAP_NER.

9. “Dual-engine” thực chất nghĩa là gì?

Đây là cách tác giả conceptualize architecture:

Engine 1 — Semantic
PhoBERT

Giải quyết:

Token này có ý nghĩa gì trong context?

Engine 2 — Structural
CRF

Giải quyết:

Các nhãn của toàn câu nên liên kết với nhau thế nào?

Hai phần bổ trợ:

              PhoBERT
                 │
       "hiểu nội dung"
                 │
                 ▼
             Emission
                 │
                 ▼
                CRF
                 │
        "hiểu cấu trúc label"
                 │
                 ▼
             NER output
10. “Modular optimization” nghĩa là gì?

Câu cuối:

Semantic Engine is initialized from pre-trained PhoBERT weights, while Structural Engine is learned from scratch on PAP_NER...

có nghĩa:

PhoBERT

Không train từ đầu.

Pretrained PhoBERT
       ↓
   Fine-tuning
       ↓
PAP_NER

PhoBERT đã có kiến thức tiếng Việt từ pretraining.

CRF

Không có pretrained weights tương tự.

Transition matrix được học từ PAP_NER:

CRF transition matrix
        ↓
learn from PAP_NER

Ví dụ nó học:

B-VBPL → I-VBPL    thường xuyên
I-VBPL → I-VBPL    thường xuyên
O → I-VBPL         hiếm/không hợp lệ
B-CQ → I-VBPL      không hợp lý