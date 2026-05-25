# TÀI LIỆU MÔ TẢ CẤU TRÚC HỆ THỐNG CHATBOT RAG PTIT

**Mô tả chức năng các folder và file Python trong hệ thống**

- **Phiên bản tài liệu:** local development
- **Mục đích:** phục vụ quản lý mã nguồn, báo cáo kỹ thuật và triển khai web local/public

**Tóm tắt:** Hệ thống gồm backend FastAPI, RAG retrieval bằng BM25 + FAISS + RRF, model BARTpho-LoRA, tầng kiểm soát độ tin cậy, SQLite database, workflow hỏi thầy cô/admin, và giao diện web local cho user/admin.

---

## 1. Cấu trúc tổng thể hệ thống

```
PythonProject5/
|
|-- app.py
|-- build_index.py
|-- retriever_rrf.py
|-- rag_answer.py
|-- rag_pipeline.py
|
|-- control/
|-- knowledge/
|-- workflow/
|-- static/
|-- index/
|-- models/
|-- hf_models/
|
|-- admin_pending.py
|-- admin_answer_pending.py
|-- admin_corrections.py
|
|-- test_db.py
|-- test_store.py
|-- test_control.py
`-- check_db_records.py
```

**Các thành phần trên phối hợp để tạo thành chatbot RAG có khả năng:**
- Trả lời từ tài liệu
- Kiểm tra độ tin cậy
- Lưu câu hỏi chưa chắc vào hàng chờ
- Tiếp nhận câu trả lời từ thầy cô/admin
- Tạo correction cho câu trả lời cũ

---

## 2. Các folder chính

### `control/`
**Chức năng:**
- Kiểm tra context có đủ liên quan không
- Kiểm tra answer có bám context không
- Quyết định trả lời thẳng, trả lời thận trọng, hoặc chuyển hỏi thầy cô
- Nhận diện câu hỏi về quyền/được phép/ngoại lệ
- Nhận diện ngành và học kỳ trong câu hỏi
- Rerank chunk theo ngành + học kỳ để giảm nhiễu retrieval

### `knowledge/`
**Chức năng:**
- Tạo database `rag_chatbot.db`
- Lưu câu trả lời đã xác nhận bởi thầy cô/admin
- Lưu câu hỏi pending cần xác minh
- Lưu lịch sử `answer_logs`
- Lưu corrections cho câu trả lời cũ
- Cung cấp các hàm thêm/sửa/tìm dữ liệu

### `workflow/`
**Chức năng:**
- Quản lý hàng chờ câu hỏi cần hỏi thầy cô
- Lưu câu trả lời thầy cô vào `confirmed_answers`
- Tạo correction cho `answer_logs` cũ có liên quan
- Đồng bộ tri thức mới vào hệ thống

---

## 3. Các file Python chính ở thư mục gốc

### `app.py`
**Chức năng:**
- Khởi động web server local bằng FastAPI
- Load retriever một lần khi server startup
- Load BARTpho-LoRA một lần khi server startup
- Tạo API POST `/api/ask` cho chatbot
- Tạo API admin để xem pending, trả lời pending, xem corrections và mark notified
- Phục vụ trang `/chat` và `/admin`

**Lệnh chạy thường dùng:**
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

### `build_index.py`
**Chức năng:**
- Đọc file chunks JSON
- Tokenize nội dung bằng PyVi cho BM25
- Encode chunks bằng Vietnamese bi-encoder
- Tạo FAISS vector index
- Lưu `bm25.pkl`, `faiss.index`, `embeddings.npy`, `chunks.json`, `meta.json` vào folder `index/`
- Mỗi lần đổi file chunks cần chạy lại file này

**Lệnh chạy thường dùng:**
```bash
python build_index.py
```

### `retriever_rrf.py`
**Chức năng:**
- Load index đã build từ folder `index/`
- Tìm kiếm BM25
- Tìm kiếm FAISS
- Kết hợp hai nguồn bằng Reciprocal Rank Fusion
- Trả về top_k chunks để pipeline dùng làm context

### `rag_answer.py`
**Chức năng:**
- Load BARTpho tokenizer theo cách thủ công để tránh lỗi tokenizer
- Load BARTpho base model
- Load LoRA adapter
- Cung cấp hàm `generate_answer()` cho `rag_pipeline.py` và `app.py` sử dụng

### `rag_pipeline.py`
**Chức năng:**
- Nhận câu hỏi user
- Tìm confirmed_answers trước khi gọi model
- Nếu chưa có confirmed answer thì gọi retriever
- Lấy nhiều chunks top_k, sau đó rerank theo ngành + học kỳ nếu cần
- Ghép multi-context cho model
- Với câu hỏi danh sách môn học theo ngành/học kỳ, trả lời trực tiếp từ chunk bảng
- Với câu hỏi khác, gọi BARTpho-LoRA sinh answer
- Chạy context_checker, answer_verifier và confidence_gate
- Lưu answer_logs
- Nếu không chắc, tạo pending_questions

---

## 4. Các file trong folder `control/`

### `control/text_utils.py`
**Các hàm quan trọng:**
- `normalize_text()` - Chuẩn hóa văn bản
- `extract_keywords()` - Trích xuất từ khóa
- `keyword_overlap_score()` - Tính điểm trùng lặp từ khóa
- `infer_topic()` - Suy luận chủ đề
- `is_permission_question()` - Nhận diện câu hỏi về quyền
- `is_curriculum_table_context()` - Kiểm tra context là bảng chương trình học
- `has_policy_evidence()` - Kiểm tra bằng chứng chính sách

### `control/context_checker.py`
**Các hàm quan trọng:**
- `check_context()` - Kiểm tra độ liên quan của context

### `control/answer_verifier.py`
**Các hàm quan trọng:**
- `verify_answer()` - Xác minh câu trả lời có bám sát context

### `control/confidence_gate.py`
**Các hàm quan trọng:**
- `generate_teacher_questions()` - Tạo câu hỏi gửi thầy cô
- `decide_action()` - Quyết định hành động (trả lời/hỏi thầy)
- `build_user_response()` - Xây dựng phản hồi cho user

### `control/query_constraints.py`
**Các hàm quan trọng:**
- `extract_program()` - Trích xuất ngành học
- `extract_semester()` - Trích xuất học kỳ
- `extract_query_constraints()` - Trích xuất ràng buộc truy vấn
- `is_curriculum_subject_list_question()` - Nhận diện câu hỏi danh sách môn học
- `rerank_hits_by_constraints()` - Xếp hạng lại kết quả theo ràng buộc

---

## 5. Các file trong folder `knowledge/`

### `knowledge/db.py`
**Chức năng:**
- Tạo `rag_chatbot.db`
- Tạo bảng `confirmed_answers`
- Tạo bảng `pending_questions`
- Tạo bảng `answer_logs`
- Tạo bảng `corrections`
- Bật WAL mode cho SQLite để đọc/ghi ổn hơn khi có nhiều request nhỏ
- Tạo index cho các cột thường truy vấn

**Các hàm chính:**
- `get_connection()` - Lấy kết nối database
- `init_db()` - Khởi tạo database

### `knowledge/init_db.py`
**Chức năng:**
- Gọi `init_db()`
- In đường dẫn database để kiểm tra

**Lệnh chạy:**
```bash
python knowledge/init_db.py
```

### `knowledge/store.py`
**Chức năng:**
- Thêm và tìm confirmed answer
- Thêm và liệt kê pending questions
- Thêm và liệt kê answer logs
- Tạo và liệt kê corrections
- Lưu câu trả lời thầy cô cho một pending question
- Tìm các answer_logs có thể cần correction

**Các hàm quan trọng:**
- `add_confirmed_answer()` - Thêm câu trả lời đã xác nhận
- `search_confirmed_answers()` - Tìm kiếm câu trả lời đã xác nhận
- `add_pending_question()` - Thêm câu hỏi chờ xử lý
- `list_pending_questions()` - Liệt kê câu hỏi chờ xử lý
- `add_answer_log()` - Thêm log câu trả lời
- `list_answer_logs_by_user()` - Liệt kê log theo user
- `create_correction()` - Tạo correction
- `list_corrections()` - Liệt kê corrections
- `save_teacher_answer_for_pending()` - Lưu câu trả lời thầy cô
- `find_logs_that_may_need_correction()` - Tìm log cần correction

---

## 6. Các file trong folder `workflow/`

### `workflow/teacher_queue.py`
**Chức năng:**
- Lấy pending question theo ID
- Format pending question để hiển thị trong CLI hoặc admin web
- In danh sách pending questions
- Nhận câu trả lời từ thầy cô/admin
- Lưu câu trả lời xác nhận vào confirmed_answers
- Gọi knowledge_sync để tạo correction nếu cần

**Các hàm chính:**
- `get_pending_question_by_id()` - Lấy câu hỏi pending theo ID
- `format_pending_question()` - Format câu hỏi pending
- `print_pending_questions()` - In danh sách câu hỏi pending
- `answer_pending_question()` - Trả lời câu hỏi pending

### `workflow/knowledge_sync.py`
**Chức năng:**
- Khi có confirmed answer mới, lấy các answer_logs cùng topic
- Đánh giá log nào có khả năng cần correction
- Tạo nội dung correction message
- Lưu correction vào database

**Các hàm chính:**
- `should_create_correction()` - Kiểm tra có nên tạo correction
- `build_correction_message()` - Xây dựng nội dung correction
- `sync_corrections_from_confirmed_answer()` - Đồng bộ correction từ confirmed answer

---

## 7. Các file admin CLI

**Các lệnh admin:**
```bash
python admin_pending.py
python admin_answer_pending.py --pending_id 1 --answer "..."
python admin_corrections.py
```

---

## 8. Các file test/debug

**Các file test:**
- `test_db.py` - Test database
- `test_store.py` - Test store functions
- `test_control.py` - Test control functions
- `check_db_records.py` - Kiểm tra records trong database

---

## 9. Các file web trong `static/`

**Cấu trúc:**
- `chat.html` - Giao diện chat cho user
- `admin.html` - Giao diện admin
- `chat.js` - Logic xử lý chat
- `admin.js` - Logic xử lý admin

---

## 10. Luồng hoạt động hiện tại

### 10.1. Khi user hỏi

```
chat.html
  ↓
chat.js
  ↓
POST /api/ask
  ↓
app.py
  ↓
rag_pipeline.py
  ↓
confirmed_answers?
  ↓
retriever_rrf.py
  ↓
control/query_constraints.py
  ↓
direct curriculum answer hoặc BARTpho-LoRA
  ↓
context_checker
  ↓
answer_verifier
  ↓
confidence_gate
  ↓
lưu answer_logs
  ↓
nếu không chắc: lưu pending_questions
  ↓
trả kết quả về web
```

### 10.2. Khi admin xử lý pending

```
admin.html
  ↓
admin.js
  ↓
GET /api/admin/pending
  ↓
admin nhập câu trả lời
  ↓
POST /api/admin/answer-pending
  ↓
workflow/teacher_queue.py
  ↓
knowledge/store.py
  ↓
lưu confirmed_answers
  ↓
workflow/knowledge_sync.py
  ↓
tạo corrections
```

---

## 11. Vai trò từng phần trong một câu ngắn

| Thành phần | Vai trò |
|------------|---------|
| **app.py** | Web server FastAPI, điểm vào chính của hệ thống |
| **build_index.py** | Xây dựng BM25 + FAISS index từ chunks |
| **retriever_rrf.py** | Tìm kiếm và kết hợp BM25 + FAISS bằng RRF |
| **rag_answer.py** | Load và sử dụng BARTpho-LoRA để sinh câu trả lời |
| **rag_pipeline.py** | Điều phối toàn bộ luồng RAG từ câu hỏi đến câu trả lời |
| **control/** | Kiểm tra độ tin cậy và quyết định hành động |
| **knowledge/** | Quản lý database và lưu trữ tri thức |
| **workflow/** | Xử lý hàng chờ và đồng bộ tri thức từ thầy cô |
| **static/** | Giao diện web cho user và admin |

---

## 12. Phần nên bổ sung tiếp theo

**Bước tiếp theo** để chatbot hoạt động tự nhiên hơn trong hội thoại nhiều lượt là bổ sung module lịch sử chat và viết lại câu hỏi follow-up.

### Cấu trúc đề xuất:
```
chat/
|-- history_store.py
`-- question_rewriter.py
```

### Chức năng:
- Lưu lịch sử hội thoại theo `conversation_id`
- Hiểu các câu hỏi follow-up như: "còn học kỳ 2 thì sao?", "vậy ngành đó thì thế nào?"
- Rewrite câu hỏi phụ thuộc ngữ cảnh thành câu hỏi đầy đủ trước khi retrieval
- Không dùng lịch sử chat làm nguồn sự thật; nguồn sự thật vẫn là `confirmed_answers` hoặc retrieved chunks

---

**Kết thúc tài liệu**

