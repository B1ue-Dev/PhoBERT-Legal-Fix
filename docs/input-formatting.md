iả sử câu:

Ủy ban nhân dân tỉnh Quảng Trị ban hành Nghị định 34/2016/NĐ-CP

Sau preprocessing/tokenization, ta có một chuỗi token:

x = [x₁, x₂, ..., xₙ]

Trong đó:

n: số token
xᵢ: token thứ i

Ground truth tương ứng:

y = [y₁, y₂, ..., yₙ]

Ví dụ:

Ủy       B-CQ
ban      I-CQ
nhân     I-CQ
dân      I-CQ
tỉnh     I-CQ
Quảng    I-CQ
Trị      I-CQ
ban      O
hành     O
Nghị     B-VBPL
định     I-VBPL
...

Paper sử dụng BIO tagging.
4. PhoBERT làm gì?

PhoBERT nhận input:

x₁, x₂, ..., xₙ

và tạo contextual representation:

H = [h₁, h₂, ..., hₙ]

Trong đó:

hᵢ ∈ R^768

với PhoBERT-Base.

Điểm quan trọng:

hᵢ

không chỉ chứa thông tin của token i.

Nó chứa thông tin ngữ cảnh xung quanh.

Ví dụ từ:

cá nhân

có thể được hiểu khác nhau tùy câu:

"cá nhân có quyền..."

so với:

"người nộp hồ sơ là cá nhân..."

PhoBERT dùng contextual representation để phân biệt các trường hợp này.

Vì vậy paper gọi PhoBERT là:

Semantic Engine