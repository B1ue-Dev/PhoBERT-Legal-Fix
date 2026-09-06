. Ý tưởng chính của Section 4

Tác giả chia mô hình thành 2 phần:

                  PhoBERT-CRF
                      │
          ┌───────────┴───────────┐
          ↓                       ↓
   Semantic Engine         Structural Engine
          │                       │
       PhoBERT                   CRF
          │                       │
   hiểu ngữ cảnh            ràng buộc BIO
          │                       │
          └───────────┬───────────┘
                      ↓
                 Legal/Admin NER
PhoBERT = Semantic Engine

PhoBERT chịu trách nhiệm:

Hiểu ý nghĩa và ngữ cảnh của từng token.

Ví dụ:

UBND tỉnh Hà Giang

PhoBERT nhìn toàn bộ context để hiểu rằng:

UBND
tỉnh
Hà Giang

có khả năng tạo thành một Agency (CQ).

CRF = Structural Engine

CRF không chủ yếu "hiểu nghĩa" như Transformer.

Nó tập trung vào:

Các label có quan hệ với nhau như thế nào trong một chuỗi?

Ví dụ:

B-VBPL → I-VBPL → I-VBPL → O

hợp lệ.

Trong khi:

O → I-VBPL

là một sequence không hợp lệ theo BIO chuẩn.

2. Section 4.1 đang formalize bài toán NER

Tác giả bắt đầu bằng việc định nghĩa input.

Giả sử câu có n token:

x = (x₁, x₂, ..., xₙ)

Ví dụ:

x₁        x₂       x₃       x₄
Nghị_định số       123      ...

Mỗi token cần được gán một label:

y = (y₁, y₂, ..., yₙ)

Ví dụ:

B-VBPL
I-VBPL
I-VBPL
O
3. Tập label là gì?

PAP_NER có 5 loại entity:

CQ
ĐT
SL
NG
VBPL

Kết hợp với BIO:

O

B-CQ
I-CQ

B-ĐT
I-ĐT

B-SL
I-SL

B-NG
I-NG

B-VBPL
I-VBPL

=> tổng cộng:

11 labels

|Y| = 11

Đây chính là num_labels khi bạn implement model.

4. Mục tiêu của model là gì?

Model không chỉ muốn dự đoán:

token 1 → label gì?
token 2 → label gì?
token 3 → label gì?

mà muốn tìm toàn bộ chuỗi label tốt nhất:

y* = argmax_y P(y | x)

Nói đơn giản:

Trong tất cả các chuỗi BIO có thể có, tìm chuỗi có xác suất cao nhất.

Ví dụ:

Input:
Nghị định 123/2020/NĐ-CP được ban hành...

Candidates:

O O O O O ...
B-VBPL I-VBPL I-VBPL O ...
B-CQ I-CQ O O ...
...

CRF sẽ đánh giá cả chuỗi, thay vì từng token độc lập.

5. Đây chính là khác biệt giữa Softmax và CRF

Đây là phần quan trọng nhất của đoạn bạn đưa.

Softmax

Với token i:

PhoBERT
   ↓
hᵢ
   ↓
Linear
   ↓
Softmax
   ↓
P(yᵢ | x)

Mỗi token được dự đoán gần như độc lập.

Ví dụ:

Token          Prediction

Nghị_định      B-VBPL
123/2020       I-VBPL
được           O
...

Vấn đề là model có thể tạo:

O
I-VBPL
B-CQ

CRF không có.

6. Tại sao Softmax có thể tạo O → I-VBPL?

Giả sử:

Token 1 = "ban"
Token 2 = "hành"
Token 3 = "Nghị_định"

Softmax xử lý từng vị trí:

token 1 → O
token 2 → I-VBPL
token 3 → B-CQ

Nếu xét riêng từng token thì các prediction này có thể có probability cao.

Nhưng xét toàn bộ sequence:

O → I-VBPL

không hợp lệ theo BIO.

Softmax không có cơ chế tự nhiên để nói:

"Không được chuyển từ O sang I-VBPL."

7. CRF giải quyết bằng transition score

CRF thêm một thành phần rất quan trọng:

Transition Score

Có thể hình dung:

                CRF
                 │
       ┌─────────┴─────────┐
       ↓                   ↓
Emission score       Transition score
"token này là gì?"   "label trước → label sau?"

Ví dụ:

B-VBPL → I-VBPL     score cao
I-VBPL → I-VBPL     score cao

O → I-VBPL          score rất thấp
B-CQ → I-VBPL       score rất thấp

Sau đó CRF tìm sequence có tổng score cao nhất.

8. Công thức quan trọng nhất

Về trực giác, CRF tính:

Score(x, y)
=
Emission scores
+
Transition scores

Hay:

$$ Score(x,y) = \sum_i E_{i,y_i} + \sum_i T_{y_{i-1},y_i} $$

Trong đó:

Emission
$$ E_{i,y_i} $$

trả lời:

Token i phù hợp với label yᵢ đến mức nào?

Ví dụ:

"Nghị_định"
       ↓
PhoBERT
       ↓
Linear
       ↓
B-VBPL = 8.2
B-CQ   = 1.1
O      = 0.3
Transition
$$ T_{y_{i-1},y_i} $$

trả lời:

Label trước chuyển sang label hiện tại có hợp lý không?

Ví dụ:

B-VBPL → I-VBPL     +2.5
I-VBPL → I-VBPL     +1.8
O → I-VBPL          -5.0
B-CQ → I-VBPL       -4.2
9. Sau đó CRF chọn sequence tốt nhất

Ví dụ có 2 candidate:

Sequence A
B-VBPL → I-VBPL → I-VBPL → O

Emission:

8 + 7 + 6 + 9

Transition:

+2 +2 +1

Total:

35
Sequence B
O → I-VBPL → I-VBPL → O

Emission có thể vẫn cao:

9 + 8 + 6 + 9

nhưng transition:

-5 +2 +1

Total thấp hơn.

CRF chọn:

Sequence A
10. Một điểm rất quan trọng: CRF không "cứng" BIO constraint theo cách đơn giản

Đoạn paper nói:

"subject to the structural constraint that y forms a valid BIO sequence"

và ví dụ:

I-VBPL phải được preceded bởi B-VBPL hoặc I-VBPL.

Về implementation, có hai cách:

Cách 1 — Learned transition

CRF học transition matrix:

11 × 11

Ví dụ:

T[B-VBPL, I-VBPL]
T[I-VBPL, I-VBPL]
T[O, I-VBPL]
...
Cách 2 — Hard constraints

Có thể mask các transition không hợp lệ:

O → I-VBPL       ❌
B-CQ → I-VBPL    ❌
B-VBPL → I-VBPL  ✅
I-VBPL → I-VBPL  ✅

Nếu code của paper chỉ dùng torchcrf/CRF thông thường mà không xây transition mask, thì không nên nói CRF "guarantees" BIO validity một cách tuyệt đối.

Nó học để ưu tiên sequence hợp lệ, còn hard constraint cần được implement rõ ràng.