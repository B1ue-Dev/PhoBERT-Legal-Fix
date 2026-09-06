Có thể hiểu toàn bộ pipeline như sau:

Văn bản hành chính
       ↓
Tokenization
       ↓
     PhoBERT
       ↓
Contextual Embeddings H
       ↓
   Linear Layer
       ↓
Emission Scores
       ↓
      CRF
       ↓
Viterbi Decoding
       ↓
Chuỗi BIO tối ưu
1. PhoBERT = Semantic Engine

Phần 4.3 giải thích rằng PhoBERT chịu trách nhiệm hiểu ngữ nghĩa của từng token trong ngữ cảnh toàn câu.

Ví dụ:

Ủy ban nhân dân tỉnh Hà Giang

PhoBERT không chỉ nhìn từng từ độc lập:

Ủy ban
nhân dân
tỉnh
Hà Giang

mà Self-Attention cho phép nó tạo representation có tính đến các từ xung quanh.

Kết quả:

x1     x2       x3       x4
↓      ↓        ↓        ↓
PhoBERT Transformer
↓      ↓        ↓        ↓
h1     h2       h3       h4

Mỗi hi là contextual embedding của token thứ i.

2. Linear Layer = chuyển semantic → label scores

Phần 4.4 là bước:

H = PhoBERT output
        ↓
Linear(H)
        ↓
Emission scores

Ví dụ có 11 nhãn:

B-CQ
I-CQ
B-VBPL
I-VBPL
B-NG
I-NG
...
O

thì với mỗi token, Linear Layer tạo ra:

token "Nghị định"

B-VBPL    8.2
I-VBPL    5.7
B-CQ      1.2
I-CQ      0.8
O        -2.1
...

Đây là emission score.

Quan trọng: PhoBERT + Linear chưa quan tâm nhiều đến quan hệ giữa các nhãn liên tiếp.

Ví dụ nó có thể tạo:

O → I-VBPL → B-CQ

mặc dù:

O → I-VBPL

là BIO không hợp lệ.

3. CRF = Structural Engine

Phần 4.5 giải quyết vấn đề đó.

CRF không chỉ hỏi:

Token này nên là nhãn gì?

mà hỏi:

Toàn bộ chuỗi nhãn nào có điểm cao nhất?

Nó kết hợp:

Emission score
      +
Transition score
      ↓
Sequence score

Ví dụ:

B-VBPL → I-VBPL

có transition score cao.

Trong khi:

O → I-VBPL

có transition score rất thấp.

Do đó:

PhoBERT:
O → I-VBPL → B-CQ

có thể ban đầu được đánh giá khá cao về mặt semantic.

Nhưng CRF sẽ nói:

O → I-VBPL
      ↓
 transition score rất thấp
      ↓
 toàn sequence bị giảm score

và tìm một sequence hợp lý hơn.

4. Điểm quan trọng nhất: CRF không đơn giản là "rule BIO"

Đây là chỗ đoạn paper này cần đọc rất kỹ.

Họ nói:

pytorch-crf learns transition scores entirely from data without explicit hard masking.

Tức là họ không hard-code:

O → I-CQ = forbidden
B-CQ → I-VBPL = forbidden

thành -∞.

Thay vào đó CRF học transition matrix từ dữ liệu:

              next label
           O    B-CQ   I-CQ   B-VBPL   I-VBPL
previous
O          ↑     ↑      ↓       ↑        ↓↓↓
B-CQ       ↑     ↓      ↑↑      ↑        ↓↓↓
I-CQ       ↑     ↓      ↑↑      ↑        ↓↓↓
B-VBPL     ↑     ↓      ↓       ↑        ↑↑
I-VBPL     ↑     ↓      ↓       ↑        ↑↑

Trong đó:

↑↑ = transition thường xảy ra
↓↓↓ = transition rất ít xảy ra

Sau training, họ quan sát thấy các transition BIO bất hợp lệ có score dưới khoảng -8.

Nhưng cần lưu ý: CRF thuần túy như pytorch-crf không đảm bảo về mặt toán học rằng mọi sequence BIO-invalid đều bị cấm. Nó chỉ học để những transition đó có xác suất/score rất thấp. Việc paper báo cáo 0 sequence BIO-invalid trên test set là một kết quả thực nghiệm, không phải bằng chứng rằng CRF đã trở thành hard constraint.

5. Forward Algorithm dùng để làm gì?

Đây là phần cuối 4.5.2.

CRF cần tính:

P(y | x)

và:

P(y|x) =
 exp(score(x,y))
 -------------------------
 Σ exp(score(x,y'))

Vấn đề là nếu sentence có n token và K nhãn thì có:

K^n

sequence.

Ví dụ:

n = 50
K = 11

thì:

11^50

là con số khổng lồ.

Không thể enumerate tất cả.

Vì vậy CRF dùng Forward Algorithm / Dynamic Programming.

Thay vì:

liệt kê toàn bộ K^n sequences

nó giữ lại:

alpha[i][y]

nghĩa là:

Tổng score của tất cả các sequence kết thúc tại token i với label y.

Sau đó recurrence:

alpha[i][y]
=
logsumexp(
    alpha[i-1][y']
    + transition[y', y]
    + emission[i, y]
)

Nhờ vậy complexity giảm xuống khoảng:

O(n × K²)

thay vì:

O(K^n)

Đây chính là lý do CRF có thể train trên những câu dài.

6. Viterbi khác Forward Algorithm

Hai thuật toán trong CRF có mục đích khác nhau:

Algorithm	Mục đích
Forward	Tính partition function Z(x)
Viterbi	Tìm sequence có score cao nhất

Training:

PhoBERT
   ↓
Emission
   ↓
CRF
   ↓
Forward Algorithm
   ↓
Partition Function
   ↓
CRF Loss

Inference:

PhoBERT
   ↓
Emission
   ↓
CRF
   ↓
Viterbi
   ↓
Best BIO sequence
7. Toàn bộ PhoBERT-CRF trong paper

Bạn có thể ghi nhớ bằng sơ đồ này:

                    INPUT
                      │
                      ▼
             Vietnamese tokens
                      │
                      ▼
                 ┌─────────┐
                 │ PhoBERT │
                 │ 12 layers
                 └────┬────┘
                      │
                      ▼
          Contextual Embeddings H
                      │
                      ▼
               Linear Projection
                      │
                      ▼
              Emission Scores
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
       Label fitness      Transition Matrix
             │                 │
             └────────┬────────┘
                      ▼
                     CRF
                      │
          ┌───────────┴───────────┐
          │                       │
       Training                Inference
          │                       │
          ▼                       ▼
      Forward                  Viterbi
          │                       │
          ▼                       ▼
      CRF Loss              Best sequence
          │                       │
          └───────────┬───────────┘
                      ▼
                 BIO labels
Ý tưởng cốt lõi

PhoBERT trả lời:

"Token này, dựa trên toàn bộ ngữ cảnh, có vẻ thuộc entity nào?"

CRF trả lời:

"Nhưng xét cả chuỗi thì cách gán nhãn nào hợp lý nhất?"

Do đó:

PhoBERT = hiểu ngữ nghĩa/context
Linear = tạo emission scores
CRF = mô hình hóa dependency giữa các labels
Forward = train CRF
Viterbi = decode sequence tốt nhất

Và đây chính là kiến trúc mà bạn có thể dùng trực tiếp để xây dựng PAP_NER: PhoBERT → Linear → CRF, thay vì dùng PhoBERT → Linear → Softmax như NER thông thường. 