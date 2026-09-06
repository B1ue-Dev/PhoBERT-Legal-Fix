1. Pipeline tổng thể
Vietnamese sentence
        │
        ▼
   Tokenization
        │
        ▼
   PhoBERT Encoder
        │
        │ H = contextual embeddings
        ▼
 Linear Layer
        │
        │ Emission scores
        ▼
       CRF
        │
        ├── Transition scores
        │
        ├── Forward algorithm → Training
        │
        └── Viterbi → Inference
        │
        ▼
   BIO label sequence

Ví dụ:

Input:
"Ủy ban nhân dân xã Đồng Văn"

             PhoBERT
                ↓
     contextual representations
                ↓
        Linear classifier
                ↓
      emission scores
                ↓
              CRF
                ↓
B-CQ  I-CQ  I-CQ  I-CQ  I-CQ
2. PhoBERT làm nhiệm vụ gì?

PhoBERT là Semantic Engine.

Nó không trực tiếp quyết định toàn bộ chuỗi BIO. Nó tạo ra vector biểu diễn cho từng token, nhưng vector này đã chứa ngữ cảnh của toàn câu.

Ví dụ:

"Ủy ban nhân dân xã Đồng Văn"

PhoBERT nhìn toàn bộ câu và tạo:

Ủy ban       → h1
nhân dân     → h2
xã           → h3
Đồng Văn     → h4

Mỗi hi là một vector, ví dụ:

h_i ∈ R^768

với PhoBERT-base.

Điểm quan trọng:

h_i không chỉ biểu diễn bản thân token i mà còn chứa thông tin từ các token xung quanh.

Đây chính là lợi thế của Transformer đối với NER.

3. Linear layer tạo Emission Score

Sau PhoBERT:

H ∈ R^(n × 768)

Linear layer chuyển:

768 dimensions
      ↓
number of BIO labels

Ví dụ giả sử có 13 labels:

W ∈ R^(13 × 768)
b ∈ R^13

Ta có:

E = HW + b

Kết quả:

          O     B-CQ   I-CQ   B-VBPL   I-VBPL ...
token 1   1.2    4.8    1.1     0.3      0.2
token 2   0.2    2.1    5.7     0.1      0.5
token 3   0.1    0.2    4.9     0.3      0.4

Đây gọi là emission scores.

Nó trả lời:

"Xét riêng token này và ngữ cảnh của nó, label nào có vẻ phù hợp?"

4. Đây là điểm Softmax và CRF khác nhau
PhoBERT + Softmax

Softmax làm:

token 1 → chọn label tốt nhất
token 2 → chọn label tốt nhất
token 3 → chọn label tốt nhất
...

Tức là:

ŷ_i = argmax_y P(y_i | x)

Từng token độc lập.

Vấn đề:

O → I-VBPL

có thể xảy ra.

Ví dụ:

Nghị định 34/2016/NĐ-CP

O
B-VBPL
I-VBPL
O
I-VBPL    ← sai BIO

Softmax không quan tâm nhiều đến việc label trước đó là gì.

5. CRF bổ sung Transition Score

CRF thêm một thứ mà Softmax không có:

Transition Matrix A

Ví dụ:

Previous → Current

B-VBPL → I-VBPL     +3.5
I-VBPL → I-VBPL     +2.8
O      → B-VBPL     +1.2
O      → I-VBPL     -8.5
I-CQ   → I-VBPL     -7.9

Nó học:

"Label trước và label sau có hợp lý không?"

Do đó CRF không chỉ nhìn:

Emission(token)

mà nhìn:

Emission
   +
Transition
6. CRF thực sự tối ưu cái gì?

Đây là phần quan trọng nhất.

Giả sử có một câu:

Nghị định 34 / 2016 / NĐ-CP

Có hai chuỗi ứng viên:

Sequence A
B-VBPL I-VBPL I-VBPL I-VBPL
Sequence B
B-VBPL O I-VBPL O

PhoBERT có thể cho emission tương đối cao cho cả hai.

Nhưng CRF tính:

Score(sequence)
=
Emission scores
+
Transition scores

Ví dụ:

Score(A) = 15.8
Score(B) = 9.2

→ CRF chọn A.

Nói cách khác:

PhoBERT đánh giá từng token phù hợp với label nào; CRF đánh giá toàn bộ chuỗi label có hợp lý hay không.

7. Công thức CRF

Công thức cốt lõi:

$$ S(x,y)= \sum_{i=1}^{n} E_{i,y_i} + \sum_{i=1}^{n} A_{y_{i-1},y_i} $$

Trong đó:

Emission
$$ E_{i,y_i} $$

→ token i có phù hợp với label yi không?

Transition
$$ A_{y_{i-1},y_i} $$

→ chuyển từ label trước sang label hiện tại có hợp lý không?

8. Training CRF như thế nào?

Model phải học để:

Gold sequence

có score cao.

Ví dụ:

Gold:

B-CQ I-CQ I-CQ O

Model muốn:

Score(Gold) ↑

đồng thời:

Score(wrong sequences) ↓

Loss:

$$ L = -\log P(y|x) $$

hay:

$$ L = -[S(x,y)-\log Z(x)] $$

Trong đó:

S(x,y) = score của sequence đúng
Z(x) = partition function
log Z(x) = tổng hợp score của tất cả sequence có thể
9. Tại sao cần Forward Algorithm?

Giả sử:

n = 50 tokens
L = 13 labels

Số sequence:

$$ 13^{50} $$

là cực kỳ lớn.

Không thể:

for every possible sequence:
    calculate score

CRF dùng Dynamic Programming.

Forward algorithm giúp tính:

$$ Z(x) $$

mà không cần duyệt toàn bộ L^n sequence.

Độ phức tạp:

$$ O(nL^2) $$

Ví dụ:

50 × 13²
= 8,450

thay vì:

13^50

Đây là lý do CRF có thể sử dụng được trong thực tế.

10. Khi inference thì không dùng Forward

Training:

Forward Algorithm
       ↓
Partition function
       ↓
Negative Log Likelihood

Inference:

Viterbi Algorithm
       ↓
tìm sequence có score cao nhất

Tức là:

TRAINING

PhoBERT
   ↓
Emission
   ↓
CRF
   ↓
Forward
   ↓
Loss
   ↓
Backpropagation

Còn:

INFERENCE

PhoBERT
   ↓
Emission
   ↓
CRF
   ↓
Viterbi
   ↓
Best BIO sequence
11. Viterbi là gì?

Viterbi tìm:

$$ \hat y = \arg\max_y S(x,y) $$

Ví dụ:

Token:

Nghị định | 34/2016 | /NĐ-CP

Các label có thể là:

O
B-VBPL
I-VBPL
...

Viterbi không chọn:

token 1 → best label
token 2 → best label
token 3 → best label

mà tìm:

BEST GLOBAL PATH

B-VBPL → I-VBPL → I-VBPL

dựa trên tổng score của cả đường đi.

12. Một điểm rất quan trọng trong paper này

Paper nói:

CRF "enforces" BIO constraints.

Nhưng implementation thực tế mà đoạn bạn gửi mô tả là:

không hard-mask các transition illegal.

Tức là họ không làm:

O → I-VBPL = -∞

mà để CRF tự học transition score từ dữ liệu.

Ví dụ nó học được:

O → I-VBPL = -8
B-VBPL → I-VBPL = +3
I-VBPL → I-VBPL = +2

Sau training, đường đi sai có score rất thấp.

Đây là soft constraint, không phải hard constraint.

Paper còn báo cáo rằng trên test set:

10,278 sentences
~334,000 tokens

Viterbi tạo ra:

0 invalid BIO sequences
13. Nhưng có một điểm cần bạn đặc biệt lưu ý

Có một cách diễn đạt trong paper hơi dễ gây hiểu nhầm:

"CRF strictly enforces BIO constraints"

Nếu implementation dùng pytorch-crf mà không explicit transition masking, thì về mặt toán học CRF không đảm bảo tuyệt đối rằng transition illegal không bao giờ xảy ra.

Nó chỉ học để làm những transition đó có score rất thấp.

Do đó chính xác hơn nên hiểu:

Hard constraint:
O → I-VBPL = -∞
→ tuyệt đối không thể xảy ra

còn paper này:

O → I-VBPL ≈ -8
→ cực kỳ không ưu tiên

Đây là learned/soft structural constraint.