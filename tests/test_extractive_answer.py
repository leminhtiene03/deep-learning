from control.extractive_answer import (
    count_structured_items,
    is_enumeration_question,
    should_use_extractive_answer,
    format_extractive_answer,
)


SAMPLE_DIEU = """
Điều 33. Đăng ký học phần
1. Sinh viên đăng ký học phần theo kế hoạch.
2. Sinh viên được đăng ký bổ sung khi còn chỗ.
khoản 3. Trường hợp đặc biệt do Hiệu trưởng quyết định.
khoản 4. Sinh viên không đăng ký quá hạn mức tín chỉ.
khoản 5. Vi phạm quy định sẽ bị xử lý theo quy chế.
"""


def test_enumeration_question_detected():
    assert is_enumeration_question("Điều 33 quy định gì?")
    assert is_enumeration_question("Nội dung điều 5 là gì?")
    assert not is_enumeration_question("Điểm a khoản 3 điều 33 nói gì?")


def test_should_use_extractive():
    assert should_use_extractive_answer("Điều 33 quy định gì?", SAMPLE_DIEU)
    assert not should_use_extractive_answer("Học viện ở đâu?", "Một đoạn ngắn.")


def test_format_extractive_keeps_all_khoan():
    answer = format_extractive_answer("Điều 33 quy định gì?", SAMPLE_DIEU)
    assert "khoản 3" in answer.lower()
    assert "khoản 5" in answer.lower()
    assert count_structured_items(answer) >= 2
