# test_store.py

from knowledge.db import init_db
from knowledge.store import (
    add_answer_log,
    add_pending_question,
    list_pending_questions,
    save_teacher_answer_for_pending,
    search_confirmed_answers,
    create_correction,
    list_corrections
)

init_db()

# 1. Lưu một câu trả lời chatbot đã đưa ra
log_id = add_answer_log(
    user_id="user_001",
    question="Em có được học ít hơn 18 tín chỉ HK7 không?",
    answer="Theo ngữ cảnh hiện có, mình chưa đủ thông tin để kết luận.",
    topic="dang_ky_tin_chi",
    chunk_id=28,
    page=0,
    confidence="low",
    decision="ask_teacher"
)

print("Created answer log:", log_id)

# 2. Đưa câu hỏi vào pending
pending_id = add_pending_question(
    user_id="user_001",
    question="Em có được học ít hơn 18 tín chỉ HK7 không?",
    topic="dang_ky_tin_chi",
    context_snapshot="Context chỉ cho biết HK7 có 18 tín chỉ.",
    retrieved_chunk_id=28,
    reason="Context không nêu rõ có được học ít hơn hay không.",
    teacher_questions=[
        "Sinh viên có được đăng ký ít hơn số tín chỉ chuẩn không?",
        "Điều kiện để được học ít hơn là gì?",
        "Có cần cố vấn học tập hoặc phòng đào tạo duyệt không?"
    ]
)

print("Created pending question:", pending_id)

# 3. Xem pending
pending = list_pending_questions()
print("Pending questions:")
for p in pending:
    print(p["id"], p["question"], p["teacher_questions"])

# 4. Giả lập thầy cô trả lời
confirmed_id = save_teacher_answer_for_pending(
    pending_id=pending_id,
    canonical_answer=(
        "Sinh viên có thể đăng ký ít hơn số tín chỉ trong kế hoạch chuẩn "
        "nếu có lý do phù hợp và được cố vấn học tập hoặc đơn vị đào tạo xem xét."
    ),
    verified_by="teacher_demo"
)

print("Created confirmed answer:", confirmed_id)

# 5. Tìm confirmed answer
results = search_confirmed_answers("học ít hơn tín chỉ")
print("Confirmed search results:")
for r in results:
    print(r["id"], r["canonical_question"], "=>", r["canonical_answer"])

# 6. Tạo correction cho answer cũ
correction_id = create_correction(
    answer_log_id=log_id,
    new_answer=(
        "Mình cập nhật lại câu trả lời trước đó: sinh viên có thể đăng ký ít hơn "
        "số tín chỉ chuẩn nếu có lý do phù hợp và được cố vấn học tập hoặc đơn vị đào tạo xem xét."
    ),
    reason="Đã có câu trả lời xác nhận từ thầy cô.",
    source_confirmed_answer_id=confirmed_id
)

print("Created correction:", correction_id)

# 7. Xem corrections
corrections = list_corrections()
print("Corrections:")
for c in corrections:
    print(c["id"], c["status"], c["new_answer"])