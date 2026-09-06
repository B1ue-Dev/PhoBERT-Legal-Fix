1. 5.1 Experimental setup — Thiết lập thí nghiệm

Tác giả mô tả môi trường và cách train model để người khác có thể tái lập thí nghiệm.

Hardware
PyTorch
   ↓
NVIDIA A100 40GB
   ↓
Training PhoBERT-CRF

Họ cố định:

random seed = 42

để giảm ảnh hưởng của random initialization.

Hyperparameter

Họ nói rằng đã search hyperparameter trên Development set rồi chọn configuration cuối cùng.

Ví dụ các hyperparameter thường gồm:

Learning rate
Batch size
Epoch
Weight decay
Max sequence length
Dropout
...

Điểm quan trọng:

Không được dùng Test set để chọn hyperparameter.

Pipeline đúng là:

Train
  ↓
Training set
  ↓
Train model
  ↓
Dev set
  ↓
Chọn hyperparameter / model
  ↓
Test set
  ↓
Đánh giá cuối cùng
2. 5.2 Evaluation metrics — Đánh giá NER

Đây là phần rất quan trọng nếu bạn đang xây dựng PAP_NER.

Tác giả dùng:

Precision
Recall
F1

và đặc biệt nhấn mạnh Strict Matching.

3. Strict Matching nghĩa là gì?

Giả sử ground truth:

Tỉnh Quảng Trị
B-CQ I-CQ I-CQ

Model dự đoán:

Tỉnh Quảng Trị
B-CQ I-CQ I-CQ

→ TP

Nhưng nếu model dự đoán:

Quảng Trị
B-CQ I-CQ

thì dù "Quảng Trị" nằm trong entity thật, nó không được tính là TP trong Strict evaluation, bởi vì boundary không khớp hoàn toàn.

Hoặc:

Tỉnh Quảng Trị
B-VBPL I-VBPL I-VBPL

boundary đúng nhưng entity type sai → cũng không phải TP.

Do đó Strict yêu cầu:

        Span đúng
           +
       Type đúng
           ↓
          TP
4. Precision / Recall / F1
Precision

Trong các entity model dự đoán:

Bao nhiêu entity là đúng?

Precision = TP / (TP + FP)

Ví dụ:

Model dự đoán 100 entity
80 đúng

→ Precision = 80%.

Recall

Trong tất cả entity thật:

Model tìm được bao nhiêu?

Recall = TP / (TP + FN)

Ví dụ:

Dataset có 100 entity
Model tìm được 80

→ Recall = 80%.

F1

Kết hợp Precision và Recall:

F1 = 2 × Precision × Recall
     --------------------------
       Precision + Recall

F1 càng cao → model càng tốt.

Trong NER, F1 thường là metric chính.

5. Micro F1 và Macro F1

Đây là phần bạn nên đặc biệt chú ý vì PAP_NER có 5 loại entity và phân bố không cân bằng.

Ví dụ:

CQ       62,915
ĐT       67,001
NG       35,647
VBPL     25,522
SL       14,722
Micro F1

Gộp tất cả:

CQ + ĐT + NG + VBPL + SL
              ↓
        TP / FP / FN
              ↓
          Micro F1

Entity xuất hiện nhiều sẽ có ảnh hưởng lớn hơn.

Macro F1

Tính riêng:

F1_CQ
F1_ĐT
F1_NG
F1_VBPL
F1_SL

sau đó:

Macro F1 =
(F1_CQ + F1_ĐT + F1_NG + F1_VBPL + F1_SL) / 5

Mỗi entity type có trọng số ngang nhau.

6. Bốn chế độ evaluation

Đây là phần khá hay của paper.

Tác giả không chỉ dùng Strict mà còn phân tích:

Strict
Partial
Entity Type
Exact

Có thể hiểu:

Evaluation	Boundary	Type
Strict	✅ đúng	✅ đúng
Partial	⚠️ overlap	✅ đúng
Entity Type	❌ không quan tâm	✅ đúng
Exact	✅ đúng	❌ không quan tâm
Strict
Ground truth:
[Ủy ban nhân dân tỉnh Quảng Trị] = CQ

Prediction:
[Ủy ban nhân dân tỉnh Quảng Trị] = CQ

→ Correct
Partial

Ví dụ:

Ground truth:
[Ủy ban nhân dân tỉnh Quảng Trị]

Prediction:
[nhân dân tỉnh Quảng Trị]

Có overlap.

→ Partial có thể ghi nhận một phần credit.

Entity Type

Chỉ quan tâm:

CQ?
VBPL?
NG?
ĐT?
SL?

không quan tâm boundary.

Điều này giúp trả lời:

Model có hiểu entity thuộc loại nào không?

Exact

Chỉ quan tâm boundary:

Ground truth:
[Ủy ban nhân dân tỉnh Quảng Trị]

Prediction:
[Ủy ban nhân dân tỉnh Quảng Trị]

→ boundary đúng.

Nhưng nếu model dự đoán:

B-VBPL I-VBPL ...

thay vì:

B-CQ I-CQ ...

thì vẫn được tính đúng ở Exact, bởi vì Exact chỉ đánh giá boundary.

7. Tại sao 4 metrics này hữu ích?

Nó giúp phân biệt model sai ở đâu.

Ví dụ:

Strict F1       = 88%
Entity Type F1  = 95%
Exact F1        = 91%

Có thể suy luận:

95% Entity Type
        ↓
Semantic classification khá tốt

91% Exact
        ↓
Boundary detection khá tốt

88% Strict
        ↓
Khi kết hợp boundary + type
        ↓
vẫn có lỗi

Đây là cách phân tích tốt hơn việc chỉ nói:

"F1 = 88%."

8. McNemar's Test

Cuối cùng tác giả muốn trả lời:

PhoBERT-CRF tốt hơn PhoBERT-Softmax thật sự hay chỉ tình cờ?

Ví dụ:

PhoBERT + Softmax     F1 = 91.2
PhoBERT + CRF         F1 = 92.5

Chênh lệch:

+1.3 F1

Nhưng có thể do random variation.

McNemar's Test dùng để kiểm tra hai model được đánh giá trên cùng các mẫu test có khác biệt đáng kể hay không.

Ý tưởng:

                Model B
              Correct  Wrong
Model A
Correct          a       b
Wrong            c       d

McNemar chủ yếu quan tâm:

b = A đúng, B sai
c = A sai, B đúng

Nếu b và c chênh lệch đủ lớn → có bằng chứng rằng hai model có hành vi khác nhau có ý nghĩa thống kê.