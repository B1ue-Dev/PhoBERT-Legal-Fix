# Bộ dữ liệu PAP_NER (Public Administration Procedures Named Entity Recognition)

Tài liệu mô tả chi tiết về bộ dữ liệu **PAP_NER** dùng trong bài toán gán nhãn thực thể tên riêng (NER) phục vụ thủ tục hành chính công và chính phủ điện tử (e-Government) tại Việt Nam (công bố trên tạp chí *PLOS ONE*, DOI: [10.1371/journal.pone.0353166](https://doi.org/10.1371/journal.pone.0353166)).

---

## 1. Hệ thống nhãn thực thể (The Semantic Labeling Scheme)

Khác với các hệ thống NER tổng quát (chỉ dùng các nhãn chung chung như `PER`, `LOC`, `ORG`), **PAP_NER** được thiết kế dựa trên **chức năng nghiệp vụ hành chính và pháp lý** với 5 nhóm thực thể chính:

| Nhãn | Tên đầy đủ (Entity Type) | Ý nghĩa nghiệp vụ | Ví dụ tiêu biểu |
| :--- | :--- | :--- | :--- |
| **CQ** | **Agency** (Cơ quan) | Cơ quan, đơn vị có thẩm quyền thực hiện hoặc ban hành | *Ủy ban nhân dân tỉnh Hà Giang, Sở Kế hoạch và Đầu tư, Bộ Công an...* |
| **ĐT** | **Object** (Đối tượng) | Đối tượng thực hiện, thụ hưởng hoặc chịu tác động của thủ tục | *công dân, doanh nghiệp tư nhân, cá nhân, người nước ngoài...* |
| **VBPL** | **Legal Document** (Văn bản pháp luật) | Tên văn bản, luật, nghị định, thông tư làm căn cứ pháp lý | *Nghị định 123/2020/NĐ-CP, Bộ luật Hình sự, Thông tư 01/2021/TT-BTP...* |
| **NG** | **Datetime** (Ngày giờ) | Mốc thời gian ban hành, ngày có hiệu lực, thời hạn thực hiện | *15/03/2024, ngày 15 tháng 10 năm 2020, 08 giờ 30 phút...* |
| **SL** | **Quantity** (Số lượng) | Định lượng, số ngày giải quyết, lệ phí, phần trăm | *10 ngày, 5 triệu đồng, 100% vốn điều lệ, 03 bộ hồ sơ...* |
| **O** | **Outside** | Từ thông thường, không nằm trong thực thể | *căn cứ, theo quy định, ban hành, tiếp nhận...* |

---

## 2. Chuẩn biểu diễn BIO (Begin, Inside, Outside)

Do các thực thể trong văn bản hành chính thường gồm nhiều từ ghép kéo dài, dữ liệu sử dụng định dạng nhãn **BIO**:
* **B- [Entity]** (Begin): Từ đầu tiên bắt đầu một thực thể.
* **I- [Entity]** (Inside): Các từ tiếp theo nằm bên trong cùng thực thể đó.
* **O** (Outside): Không thuộc bất kỳ thực thể nào.

**Ví dụ:**
```text
UBND             B-CQ
tỉnh             I-CQ
Hà_Giang         I-CQ
ban_hành         O
Nghị_định        B-VBPL
123/2020/NĐ-CP   I-VBPL
```

Tổng cộng hệ thống có **11 nhãn phân loại**: `B-CQ`, `I-CQ`, `B-ĐT`, `I-ĐT`, `B-VBPL`, `I-VBPL`, `B-NG`, `I-NG`, `B-SL`, `I-SL`, và `O`.

---

## 3. Cấu trúc File Dữ liệu (CoNLL Format)

Dữ liệu được lưu trong 3 file chính:
* `train_data.txt` (Tập huấn luyện)
* `dev_data.txt` (Tập kiểm thử trong quá trình train / validation)
* `test_data.txt` (Tập đánh giá cuối cùng)

Mỗi dòng đại diện cho một từ (đã tách từ), các thuộc tính được phân tách bằng khoảng trắng / tab. Giữa các câu được phân tách bằng một dòng trống (`\n`):

```text
Đăng_ký                    N   B-NP   O
thành_lập                  V   B-VP   O
doanh_nghiệp               V   I-VP   B-ĐT
tư_nhân                    A   O      I-ĐT
phòng_đăng_ký_kinh_doanh   N   B-NP   B-CQ
sở_kế_hoạch_và_đầu_tư      N   I-NP   B-CQ
```

### Quy tắc sử dụng các cột:
1. **Cột 1 (Word):** Từ đã tách từ tiếng Việt $\to$ **Đầu vào chính của mô hình (Input)**.
2. **Cột 2 (POS):** Từ loại (Part-of-Speech: N, V, A, Np, M, E...).
3. **Cột 3 (Chunk):** Cụm cú pháp (B-NP, I-NP, B-VP...).
4. **Cột 4 (NER):** Nhãn thực thể $\to$ **Nhãn mục tiêu cần dự đoán (Target Label)**.

> [!NOTE]
> Trong thí nghiệm chuẩn của paper, mô hình chỉ sử dụng **Cột 1 (Word)** làm đầu vào và **Cột 4 (NER)** làm nhãn giám sát. Hai cột POS và Chunk được cung cấp để hỗ trợ các nghiên cứu mở rộng.

---

## 4. Thống kê Phân bổ Dữ liệu (Bảng 2 - Table 2)

Bộ dữ liệu có quy mô lớn với **162,801 câu** và **205,807 thực thể**, được phân chia theo tỷ lệ chuẩn:

| Loại thực thể (Entity Type) | Train (87.4%) | Dev (6.3%) | Test (6.3%) | Toàn bộ (All) | Tỷ lệ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ĐT** (Đối tượng) | 52,500 | 7,322 | 7,179 | **67,001** | **32.6%** |
| **CQ** (Cơ quan) | 48,418 | 7,186 | 7,311 | **62,915** | **30.6%** |
| **NG** (Ngày giờ) | 27,642 | 3,984 | 4,021 | **35,647** | **17.3%** |
| **VBPL** (Văn bản pháp luật) | 23,827 | 867 | 828 | **25,522** | **12.4%** |
| **SL** (Số lượng) | 11,060 | 2,024 | 1,638 | **14,722** | **7.1%** |
| **Tổng số thực thể (# Entities)** | **163,447** | **21,383** | **20,977** | **205,807** | **100%** |
| **Tổng số mẫu / câu (# Samples)** | **142,218** | **10,305** | **10,278** | **162,801** | — |

---

## 5. Thống kê Tần suất Nhãn Token (Bảng 3 - Table 3)

Phân bổ số lượng token cụ thể trên toàn bộ tập dữ liệu (hơn 3.5 triệu token):

| Nhãn Token | Số lượng Token (Count) | Tỷ lệ / Ý nghĩa |
| :--- | :---: | :--- |
| **O** | **3,292,690** | Các từ ngữ thông thường ngoài thực thể |
| **I-VBPL** | **84,604** | **Nhãn thực thể có số lượng token lớn nhất!** |
| **B-ĐT** | 67,047 | Điểm bắt đầu của Đối tượng |
| **B-CQ** | 63,408 | Điểm bắt đầu của Cơ quan |
| **I-ĐT** | 48,346 | Phần thân của Đối tượng |
| **B-NG** | 35,647 | Điểm bắt đầu của Ngày giờ |
| **I-NG** | 35,647 | Phần thân của Ngày giờ (tỷ lệ 1:1 với B-NG) |
| **B-VBPL** | 25,522 | Điểm bắt đầu của Văn bản pháp luật |
| **I-CQ** | 20,373 | Phần thân của Cơ quan |
| **I-SL** | 16,564 | Phần thân của Số lượng |
| **B-SL** | 14,722 | Điểm bắt đầu của Số lượng |

---

## 6. Phân tích Kỹ thuật & Giá trị đối với Kiến trúc PhoBERT-CRF

1. **Hiện tượng thực thể rất dài (Long Entity Spans):**
   * Số lượng token `I-VBPL` (**84,604**) gấp hơn **3.3 lần** số lượng `B-VBPL` (**25,522**).
   * Điều này phản ánh thực tế rằng tên các văn bản pháp luật, nghị định, thông tư của Việt Nam thường rất dài:
     ```text
     Nghị định (B-VBPL) số (I-VBPL) 123/2020/NĐ-CP (I-VBPL) ngày (I-VBPL) 15 (I-VBPL) tháng (I-VBPL) 10 (I-VBPL) năm (I-VBPL) 2020 (I-VBPL)
     ```
   * **Vai trò của CRF:** Nếu chỉ dùng Softmax độc lập, mô hình rất dễ bị dự đoán ngắt quãng (chèn nhãn `O` hoặc `B-` vào giữa). Tầng **CRF** học ma trận chuyển dịch trạng thái để duy trì liên tục chuỗi `I-VBPL` và triệt tiêu lỗi cú pháp BIO.

2. **Hiện tượng mất cân bằng nhãn (Class Imbalance):**
   * Nhóm **ĐT** (32.6%) và **CQ** (30.6%) chiếm tới **63.2%** tổng số thực thể.
   * Nhóm **SL** (7.1%) và **VBPL** (12.4%) có tần suất xuất hiện ít hơn đáng kể.
   * Cần chú ý theo dõi chỉ số F1 riêng cho từng nhãn trong quá trình huấn luyện và đánh giá.