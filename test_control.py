# test_control.py

from control.context_checker import check_context
from control.answer_verifier import verify_answer
from control.confidence_gate import decide_action, build_user_response


question = "Học kì 1 năm nhất ngành công nghệ thông tin gồm những môn học nào?"

chunk = {
    "chunk_id": 28,
    "page": 0,
    "metadata": {
        "title": "KẾ HOẠCH VÀ TIẾN TRÌNH HỌC TẬP CHUẨN - Công Nghệ Thông Tin - HK1"
    },
    "content": """
KẾ HOẠCH VÀ TIẾN TRÌNH HỌC TẬP CHUẨN
Ngành: Công Nghệ Thông Tin
Khóa: 2025
Học kỳ: HK1

Danh sách môn học/học phần:
1. Triết học Mác Lênin - 3 TC
2. Giải tích 1 - 3 TC
3. Nhập môn tin học và lập trình - 3 TC
4. Đại số - 3 TC
5. Nhập môn công nghệ số và ứng dụng AI - 2 TC
6. Giáo dục thể chất 1
7. Giáo dục quốc phòng
8. Kỹ năng mềm 1

Tổng số tín chỉ có ghi trong bảng: 14 TC
"""
}

answer = """
Học kỳ 1 năm nhất ngành Công nghệ thông tin gồm các môn: Triết học Mác Lênin,
Giải tích 1, Nhập môn tin học và lập trình, Đại số, Nhập môn công nghệ số và ứng dụng AI,
Giáo dục thể chất 1, Giáo dục quốc phòng và Kỹ năng mềm 1.
"""

context_check = check_context(
    question=question,
    chunk=chunk,
    retrieval_score=0.015,
    bm25_rank=1,
    faiss_rank=1
)

answer_check = verify_answer(
    question=question,
    context=chunk["content"],
    answer=answer
)

gate = decide_action(
    question=question,
    context_check=context_check,
    answer_check=answer_check
)

final_response = build_user_response(answer, gate)

print("CONTEXT CHECK:")
print(context_check)

print("\nANSWER CHECK:")
print(answer_check)

print("\nGATE:")
print(gate)

print("\nFINAL RESPONSE:")
print(final_response)