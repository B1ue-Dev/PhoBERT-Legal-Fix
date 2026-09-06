6. CRF làm gì?

CRF thêm một transition matrix.

Ví dụ:

             next label
             B-CQ  I-CQ  B-VBPL I-VBPL O
previous
B-CQ          ...   2.5    ...    ...  ...
I-CQ          ...   3.1    ...    ...  ...
B-VBPL        ...   ...     ...    4.8  ...
I-VBPL        ...   ...     ...    5.2  ...
O             ...   ...     ...    -4.0 ...

Nó học:

Transition(yᵢ₋₁ → yᵢ)

Ví dụ:

B-VBPL → I-VBPL

có score cao.

Trong khi:

O → I-VBPL

có score thấp/không hợp lệ.

7. CRF không chỉ nhìn từng token

Đây là khác biệt quan trọng:

Softmax
token 1 → label 1
token 2 → label 2
token 3 → label 3
CRF
token 1 ─┐
         ├→ toàn bộ sequence → best labels
token 2 ─┤
         │
token 3 ─┘

CRF đánh giá toàn bộ chuỗi.

Công thức ý tưởng:

Score(X,Y)
=
Σ emission score
+
Σ transition score

hay:

Score(X,Y)
=
Σᵢ emission(i, yᵢ)
+
Σᵢ transition(yᵢ₋₁, yᵢ)

Sau đó chọn:

Y* = argmax_Y Score(X,Y)

Tức là:

tìm chuỗi BIO có tổng điểm cao nhất.

8. Viterbi nằm ở đâu?

Trong inference:

PhoBERT
   ↓
Linear
   ↓
Emission
   ↓
CRF
   ↓
Viterbi
   ↓
Best sequence

Viterbi tìm:

Y*

có score cao nhất.

Ví dụ Softmax tạo ra:

O
I-VBPL
I-VBPL
B-CQ

CRF có thể đánh giá rằng:

B-VBPL
I-VBPL
I-VBPL
B-CQ

có tổng score cao hơn.

→ chọn sequence thứ hai.

9. "CRF enforces BIO constraint" cần hiểu chính xác

Có một điểm mình muốn làm rõ.

Paper nói CRF enforces BIO constraints, nhưng về mặt implementation có hai cách:

Cách 1 — học transition penalty

Model học transition:

O → I-VBPL

có score rất thấp.

Cách 2 — hard constraint

Ta explicitly cấm:

O → I-VBPL
B-CQ → I-VBPL

bằng cách đặt transition score:

-inf

Paper đoạn bạn đưa chủ yếu mô tả theo hướng CRF học transition potentials, chứ không đủ thông tin để khẳng định implementation đã hard-code tất cả transition bất hợp lệ thành -∞.

Vì vậy khi reproduce paper, không nên tự thêm hard constraints nếu muốn bám sát paper.
15. Tại sao CRF chỉ tăng F1 khoảng +0.4% nhưng vẫn có ý nghĩa?

Đây là điểm paper muốn nhấn mạnh.

Kết quả:

PhoBERT Softmax ≈ 97.55/97.51%
PhoBERT-CRF    ≈ 97.95%

Gain tổng thể không lớn:

≈ +0.4%

Nhưng với một số entity phức tạp:

VBPL: +0.96%
ĐT:   +0.78%
SL:   +0.78%

Trong khi:

NG: gần như không tăng

Điều này rất hợp lý về mặt kiến trúc:

Entity đơn giản
      ↓
PhoBERT đã đủ tốt
      ↓
CRF ít tác dụng

Entity phức tạp
      ↓
boundary / sequence khó
      ↓
CRF có tác dụng lớn hơn
16. Quan trọng nhất: Paper này không đề xuất kiến trúc mới

Paper nói khá rõ:

PhoBERT-CRF là một kiến trúc hybrid đã được sử dụng phổ biến.

Đóng góp chính nằm ở:

PAP_NER dataset
        +
Benchmark
        +
PhoBERT-CRF evaluation
        +
Phân tích semantic vs structural
        +
Deployment analysis

Chứ không phải:

"Chúng tôi phát minh ra CRF mới"

Đây là điểm cần nhớ nếu bạn đang đọc paper để làm nghiên cứu NER văn bản pháp luật.
PhoBERT quyết định “token này có vẻ là entity gì?” → CRF quyết định “toàn bộ chuỗi nhãn nào hợp lý nhất?”

1. Pipeline tổng thể

Với câu:

Căn cứ Nghị định 34/2016/NĐ-CP của Chính phủ

Ta có:

Input tokens
   ↓
PhoBERT
   ↓
Contextual embeddings H
   ↓
Linear Layer
   ↓
Emission scores
   ↓
CRF
   ↓
Transition scores + Emission scores
   ↓
Viterbi decoding
   ↓
BIO sequence

Ví dụ:

Nghị định     34/2016/NĐ-CP      của       Chính phủ
   ↓               ↓              ↓             ↓
B-VBPL         I-VBPL           O           B-CQ

Điểm quan trọng là PhoBERT không trực tiếp quyết định sequence cuối cùng. Nó tạo ra emission score, sau đó CRF sử dụng thêm thông tin về quan hệ giữa các nhãn.

2. Emission score là gì?

Sau khi đi qua PhoBERT, mỗi token có một vector:

Nghị định → h₁
34/2016/NĐ-CP → h₂
của → h₃
Chính phủ → h₄

PhoBERT Base có hidden size:

768

nên:

hᵢ ∈ R⁷⁶⁸

Sau đó Linear Layer biến:

768 dimensions
      ↓
num_labels dimensions

Ví dụ có 13 nhãn:

hᵢ [768]
   ↓ Linear
scores [13]

Ta có:

Emission[i][label]

Ví dụ giả định:

Token	B-VBPL	I-VBPL	B-CQ	O
Nghị định	8.5	2.1	0.3	-2.0
34/2016/NĐ-CP	1.2	9.1	0.4	-3.0
của	-1.2	-2.1	-0.5	7.8
Chính phủ	0.2	-1.1	8.9	-2.0

Như vậy PhoBERT nói:

Nghị định → B-VBPL rất phù hợp
34/2016/NĐ-CP → I-VBPL rất phù hợp
của → O rất phù hợp
Chính phủ → B-CQ rất phù hợp

Nhưng đây chỉ là đánh giá từng token.

3. Vấn đề của Softmax

PhoBERT + Softmax thường làm:

token 1 → chọn label tốt nhất
token 2 → chọn label tốt nhất
token 3 → chọn label tốt nhất
...

Tức là:

$$ \hat y_i = \arg\max_y P(y_i|x) $$

Mỗi token gần như được quyết định độc lập.

Ví dụ Softmax có thể tạo:

Nghị định       → O
34/2016/NĐ-CP   → I-VBPL
của             → O

Đây là:

O → I-VBPL → O

❌ Sai BIO.

Vì:

I-VBPL

không được phép xuất hiện ngay sau:

O

Softmax bản thân nó không biết rằng chuỗi này bất hợp lệ.

4. CRF thêm một thứ rất quan trọng: Transition Score

CRF học một ma trận:

$$ A_{y_{i-1},y_i} $$

Nó trả lời:

Nếu token trước có nhãn X, thì token hiện tại có nhãn Y có hợp lý không?

Ví dụ:

Previous → Current	Score
B-VBPL → I-VBPL	+3.5
I-VBPL → I-VBPL	+2.8
O → B-VBPL	+1.5
O → O	+2.0
O → I-VBPL	-8.5
B-CQ → I-VBPL	-7.9

CRF học những giá trị này từ dataset.

5. Đây chính là “Structural Engine”

Ta có hai loại thông tin:

PhoBERT
Emission score

trả lời:

Token này giống entity nào?

CRF
Transition score

trả lời:

Nhãn này có hợp lý khi đứng cạnh nhãn trước không?

Kết hợp:

$$ Score(x,y) = \sum_i Emission(i,y_i) + \sum_i Transition(y_{i-1},y_i) $$

Đây là công thức cốt lõi của PhoBERT-CRF.

6. Ví dụ cực kỳ quan trọng

Giả sử PhoBERT tạo ra:

Token 1: Nghị định
Token 2: 34/2016/NĐ-CP

Emission:

             B-VBPL    I-VBPL
Nghị định      8.5       2.0
34/...         3.0       9.0

Có hai sequence:

Sequence A
B-VBPL → I-VBPL

Emission:

8.5 + 9.0 = 17.5

Transition:

B-VBPL → I-VBPL = +3

Tổng:

17.5 + 3 = 20.5
Sequence B
O → I-VBPL

Emission:

2.0 + 9.0 = 11.0

Transition:

O → I-VBPL = -8

Tổng:

11 - 8 = 3

CRF sẽ chọn:

B-VBPL → I-VBPL

mặc dù nó phải đánh giá toàn bộ sequence.

7. Vậy CRF có phải hard constraint không?

Theo chính paper này: không phải hard constraint ngay từ đầu.

Đây là điểm rất quan trọng.

Paper nói implementation sử dụng:

pytorch-crf

và không explicit hard-mask các transition bất hợp lệ.

Thay vào đó:

CRF học transition score từ data

Ví dụ:

O → I-VBPL

được học thành score rất thấp:

≈ -8

Trong khi:

B-VBPL → I-VBPL

có score cao hơn.

Do đó paper gọi đây là:

soft constraints

Sau khi train đủ dữ liệu, soft constraint này trở thành gần giống:

de facto hard constraint

Theo paper, họ kiểm tra test set khoảng:

10,278 sentences
≈ 334,000 tokens

và Viterbi decoding tạo:

0 structurally invalid BIO sequences
8. Forward Algorithm dùng để làm gì?

Đây là phần dễ bị nhầm.

CRF cần tính:

$$ P(y|x) = \frac{e^{Score(x,y)}}{Z(x)} $$

Trong đó:

Z(x)

là tổng score của tất cả các sequence có thể xảy ra.

Ví dụ có:

n = 50 tokens
K = 10 labels

thì số sequence:

$$ 10^{50} $$

Không thể duyệt từng sequence.

CRF dùng:

Forward Algorithm + Dynamic Programming

để tính tổng này hiệu quả.

Độ phức tạp:

$$ O(nK^2) $$

Trong đó:

n = số token
K = số label
9. Còn Viterbi dùng để làm gì?

Có hai bài toán khác nhau:

Training

Cần:

Partition function Z
        ↓
Forward algorithm

để tính loss.

Inference

Cần:

Sequence tốt nhất
        ↓
Viterbi algorithm

Tức là:

TRAINING
PhoBERT
   ↓
Emission
   ↓
CRF
   ↓
Forward Algorithm
   ↓
Loss

Còn inference:

PhoBERT
   ↓
Emission
   ↓
CRF
   ↓
Viterbi
   ↓
Best BIO sequence
10. Công thức loss

CRF thường tối ưu negative log-likelihood:

$$ \mathcal L = -\log P(y|x) $$

hay:

$$ \mathcal L = -\left[ Score(x,y) - \log Z(x) \right] $$

Trong đó:

Score(x,y)

→ điểm của ground-truth sequence

và:

log Z(x)

→ tổng hợp điểm của mọi sequence có thể.

Training sẽ cố làm:

Score(correct sequence)
        ↑
        càng cao càng tốt

và:

Score(wrong sequences)
        ↓
        càng thấp càng tốt
11. Tại sao CRF đặc biệt hữu ích cho PAP_NER?

Theo phần thảo luận của paper, CRF không cải thiện mọi entity giống nhau.

Nó đặc biệt hữu ích với:

ĐT – Object

Có tính mơ hồ ngữ nghĩa.

Ví dụ một noun phrase có thể là:

Object entity

hoặc:

generic noun

PhoBERT mạnh ở phần này.

VBPL – Legal Document

Đây là nơi CRF thể hiện rõ nhất.

Ví dụ:

Nghị định 34/2016/NĐ-CP

là một span dài.

Softmax có thể:

B-VBPL
I-VBPL
O
I-VBPL

hoặc cắt entity sai boundary.

CRF giúp duy trì:

B-VBPL
I-VBPL
I-VBPL
I-VBPL
NG – Datetime

Ví dụ:

12/05/2025

pattern rất rõ.

PhoBERT đã gần như hoàn hảo nên:

PhoBERT ≈ 99.99 F1

CRF thêm vào cũng không cải thiện đáng kể.

Điều này dẫn đến một kết luận quan trọng:

CRF không phải “càng thêm càng tốt”; nó đặc biệt có giá trị khi entity có boundary phức tạp hoặc sequence structure mạnh.

12. Có một điểm trong paper bạn nên đặc biệt lưu ý

Paper gọi CRF là:

“strictly enforcing BIO constraints”

nhưng phần implementation lại nói:

pytorch-crf không hard-mask illegal transitions.

Vì vậy về mặt kỹ thuật, nên hiểu chính xác là:

PhoBERT
   │
   │ semantic information
   ▼
Emission
   │
   ├──────────────┐
   │              │
   ▼              ▼
Token evidence   Transition matrix
                  │
                  │ structural preference
                  ▼
                 CRF
                  │
                  ▼
               Viterbi
                  │
                  ▼
            Best sequence

CRF không “biết luật BIO” bằng một bộ luật hard-coded. Nó học xác suất/score của các transition từ dữ liệu.