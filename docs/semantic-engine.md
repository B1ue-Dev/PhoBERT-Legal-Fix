mô tả phần Semantic Engine của kiến trúc PhoBERT-CRF, và quan trọng nhất là hiểu được luồng:

Text → PhoBERT → contextual embedding → Linear layer → Emission scores → CRF → BIO labels

Mình làm rõ từng phần theo đúng paper, đồng thời chỉ ra Softmax nằm ở đâu và CRF thay thế nó như thế nào.

1. Toàn bộ kiến trúc

Có thể hình dung:

Vietnamese sentence
        │
        ▼
   Tokenization
        │
        ▼
      PhoBERT
   12 Transformer
      layers
        │
        ▼
Contextual embeddings H
        │
        ▼
   Linear layer
        │
        ▼
 Emission scores E
        │
        ├──────────────┐
        │              │
        ▼              ▼
   Semantic score   Transition score
        │              │
        └──────┬───────┘
               ▼
              CRF
               │
         Viterbi decoding
               │
               ▼
        BIO entity labels

Ví dụ:

Nghị định 34/2016/NĐ-CP

Sau tokenization:

[Nghị, định, 34, /, 2016, /, NĐ, -, CP]

PhoBERT không chỉ nhìn từng token riêng biệt.

Ví dụ embedding của:

34

sẽ phụ thuộc vào context:

Nghị định 34/2016/NĐ-CP
^^^^^^^^^^^^^^^^^^^^^^^^

Do đó 34 có representation khác với 34 trong:

Tôi mua 34 sản phẩm

Đây chính là vai trò của contextual embedding.

2. PhoBERT là Semantic Engine

Paper gọi PhoBERT là:

Semantic Engine

Nó chịu trách nhiệm trả lời câu hỏi:

"Token này có ý nghĩa gì trong ngữ cảnh hiện tại?"

Ví dụ:

Ủy ban nhân dân xã Đồng Văn

PhoBERT nhìn toàn bộ context và tạo representation cho từng token:

Ủy ban       → h₁
nhân dân     → h₂
xã           → h₃
Đồng Văn     → h₄

Mỗi hᵢ là một vector.

Với PhoBERT-Base:

hᵢ ∈ R^768

Nếu câu có n token:

H ∈ R^(n × 768)

Ví dụ:

n = 10

H:
10 × 768
3. Tại sao PhoBERT hiểu được context?

Paper nói PhoBERT có:

12 Transformer layers
12 attention heads
hidden size 768
khoảng 135M parameters

Một token được contextualize thông qua Self-Attention.

Ví dụ:

Nghị định 34/2016/NĐ-CP

Representation của:

34

có thể attention đến:

Nghị định
2016
NĐ
CP

Do đó PhoBERT học được quan hệ:

Nghị định
    ↓
34
    ↓
2016
    ↓
NĐ-CP

Đây là phần semantic understanding.

4. Input embedding

Paper viết:

token embedding + position embedding + segment embedding

Có thể hiểu:

Embedding_i
=
TokenEmbedding_i
+
PositionEmbedding_i
+
SegmentEmbedding_i

Ví dụ:

Nghị

sẽ có:

Token embedding
      +
Position embedding
      +
Segment embedding
      ↓
final embedding

Sau đó đưa vào Transformer.

5. 12 Transformer layers làm gì?

Luồng đơn giản:

Input embeddings
       ↓
Transformer Layer 1
       ↓
Transformer Layer 2
       ↓
...
       ↓
Transformer Layer 12
       ↓
H

Mỗi Transformer layer gồm hai phần chính:

        Input
          │
          ▼
   Self-Attention
          │
     Residual + LN
          │
          ▼
         FFN
          │
     Residual + LN
          │
          ▼
        Output
Self-Attention

Giúp token nhìn các token khác.

FFN

Biến đổi representation của từng token sau khi attention.

Kết quả cuối cùng:

H = [h₁, h₂, ..., hₙ]

Trong đó:

hᵢ = representation của token i

sau khi đã hiểu context.

6. Sau PhoBERT chưa phải là NER

Đây là điểm rất quan trọng.

PhoBERT chỉ tạo:

H = contextual embeddings

Nó chưa trực tiếp nói:

B-VBPL
I-VBPL
O
B-CQ
...

Paper tiếp tục sử dụng:

Linear layer

để chuyển:

768 dimensions

thành:

số lượng NER labels
7. Emission Score là gì?

Giả sử PAP_NER có:

K = số BIO labels

Ta có:

W ∈ R^(K × 768)
b ∈ R^K

Với token i:

hᵢ ∈ R^768

Linear:

eᵢ = W hᵢ + b

Kết quả:

eᵢ ∈ R^K

Đây gọi là:

Emission scores

Ví dụ giả sử có 6 labels:

O
B-CQ
I-CQ
B-VBPL
I-VBPL
B-SL

thì:

Token = "34"

Emission:

O        → -2.1
B-CQ     → -1.5
I-CQ     → -0.8
B-VBPL   →  4.8
I-VBPL   →  3.2
B-SL     → -1.2

Model đang nói:

Với riêng token "34", B-VBPL có semantic score cao nhất.

8. Đây chính là chỗ Softmax và CRF khác nhau
PhoBERT + Softmax

Softmax làm:

eᵢ
 ↓
Softmax
 ↓
P(label | token)

Sau đó chọn:

argmax

cho từng token độc lập.

Ví dụ:

Nghị       → B-VBPL
định       → I-VBPL
34         → B-VBPL
/          → I-VBPL
2016       → I-VBPL

Nhưng token tiếp theo có thể bị dự đoán sai:

O
I-VBPL
B-CQ

Trong đó:

I-VBPL

không nên xuất hiện ngay sau:

O

Softmax không thực sự biết rằng chuỗi này có vấn đề về cấu trúc.

9. CRF giải quyết vấn đề gì?

CRF không chỉ hỏi:

Token này nên là label gì?

Mà hỏi:

Toàn bộ chuỗi label nào có xác suất tốt nhất?

Ví dụ:

Token:

Nghị | định | 34 | / | 2016 | / | NĐ | - | CP

PhoBERT tạo emission:

        B-VBPL I-VBPL B-VBPL I-VBPL ...
Nghị      5      1      0      ...
định      1      5      0      ...
34        4      3      1      ...
...

CRF thêm:

Transition scores

Ví dụ:

B-VBPL → I-VBPL      +3.5
I-VBPL → I-VBPL      +4.1

O → I-VBPL           -10
B-CQ → I-VBPL        -8
I-CQ → I-VBPL        -7

Như vậy CRF biết:

B-VBPL → I-VBPL → I-VBPL

là chuỗi hợp lý.

Trong khi:

O → I-VBPL

là chuỗi không hợp lý.

10. Công thức quan trọng nhất của PhoBERT-CRF

CRF tính score cho toàn bộ sequence:

$$ Score(X,Y) = \sum_{i=1}^{n} E_{i,y_i} + \sum_{i=1}^{n} T_{y_{i-1},y_i} $$

Trong đó:

Emission
$$ E_{i,y_i} $$

là:

PhoBERT cho rằng token i phù hợp với label yᵢ đến mức nào.

Transition
$$ T_{y_{i-1},y_i} $$

là:

label trước chuyển sang label hiện tại có hợp lý không.

Do đó:

PhoBERT
   ↓
Semantic score
   +
CRF
   ↓
Structural score
   ↓
Best sequence
11. Ví dụ cực kỳ dễ hiểu

Câu:

Nghị định 34/2016/NĐ-CP

Giả sử PhoBERT tạo:

Nghị      → B-VBPL: 5.0
định      → I-VBPL: 4.5
34        → I-VBPL: 4.0
/         → I-VBPL: 3.5
2016      → I-VBPL: 4.2
/         → I-VBPL: 3.7
NĐ        → I-VBPL: 4.0
-         → I-VBPL: 3.8
CP        → I-VBPL: 4.1

Softmax có xu hướng chọn từng token:

B-VBPL
I-VBPL
I-VBPL
I-VBPL
...

Nhưng giả sử tại / model nhầm:

O

thì:

B-VBPL I-VBPL O I-VBPL ...

CRF nhìn transition:

I-VBPL → O → I-VBPL

và nhận thấy sequence này có score thấp.

Nó sẽ ưu tiên:

B-VBPL
I-VBPL
I-VBPL
I-VBPL
...

nếu tổng score cao hơn.

12. Viterbi Decoder nằm ở đâu?

Khi inference:

PhoBERT
   ↓
Emission
   ↓
CRF
   ↓
Viterbi
   ↓
Best BIO sequence

Viterbi tìm:

$$ Y^* = \arg\max_Y Score(X,Y) $$

Tức là:

tìm chuỗi BIO có tổng score cao nhất.

Ví dụ:

Candidate 1:

B-VBPL I-VBPL I-VBPL O
Score = 25.4

Candidate 2:

B-VBPL I-VBPL O I-VBPL
Score = 12.7

CRF chọn:

Candidate 1
13. Vậy "Semantic Engine + Structural Engine" thực sự nghĩa là gì?

Đây là cách paper diễn giải:

Thành phần	Nhiệm vụ
PhoBERT	Hiểu nghĩa/context
Linear layer	Chuyển embedding thành emission scores
CRF	Mô hình hóa quan hệ giữa các label
Transition matrix	Học label nào nên đi sau label nào
Viterbi	Tìm chuỗi label tốt nhất

Có thể nhớ cực ngắn:

PhoBERT = "Token này có ý nghĩa gì?"

CRF = "Chuỗi label này có hợp lý không?"

Viterbi = "Chuỗi hợp lý nhất là chuỗi nào?"
14. Một điểm rất quan trọng: CRF không "hiểu tiếng Việt"

Đừng hiểu nhầm rằng:

CRF → hiểu pháp luật

Không phải.

CRF chủ yếu học:

label → label

Ví dụ:

B-VBPL → I-VBPL
I-VBPL → I-VBPL
I-VBPL → O

Còn việc hiểu:

"Nghị định"
"Ủy ban nhân dân"
"cá nhân"

là PhoBERT đảm nhiệm.

Vì vậy:

PhoBERT
   │
   │ semantic/context
   ▼
Emission
   │
   │ structural constraints
   ▼
CRF