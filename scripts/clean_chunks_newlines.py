# -*- coding: utf-8 -*-
"""
Script làm sạch nội dung trong chunks_table_aware.json

Chức năng:
- Thay ký tự xuống dòng \n bằng khoảng trắng
- Thay \r\n, \t bằng khoảng trắng
- Thay literal "\\n" nếu tồn tại
- Gom nhiều khoảng trắng liên tiếp thành 1 khoảng trắng
- Tạo backup trước khi ghi đè file gốc
"""

import json
import re
import shutil
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Làm sạch text:
    - Thay xuống dòng thật bằng dấu cách
    - Thay literal \\n bằng dấu cách
    - Gom nhiều dấu cách thành một
    """
    if not isinstance(text, str):
        return text

    text = text.replace("\\n", " ")   # xử lý chuỗi literal backslash + n
    text = text.replace("\r\n", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    # Gom nhiều khoảng trắng liên tiếp thành 1
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_newlines_in_chunks():
    input_path = Path(__file__).resolve().parent.parent / "data" / "chunks" / "chunks_table_aware.json"

    if not input_path.exists():
        print(f"Không tìm thấy file: {input_path}")
        return

    backup_path = input_path.with_suffix(".backup.json")

    print(f"Đọc file: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        print("File JSON không phải dạng list chunks.")
        return

    print(f"Số lượng chunks: {len(chunks)}")

    # Tạo backup trước khi ghi đè
    shutil.copy2(input_path, backup_path)
    print(f"Đã tạo backup: {backup_path}")

    cleaned_count = 0

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        if "content" in chunk and isinstance(chunk["content"], str):
            old_content = chunk["content"]
            new_content = clean_text(old_content)

            if new_content != old_content:
                chunk["content"] = new_content
                cleaned_count += 1

    print(f"Số chunks đã được làm sạch: {cleaned_count}")

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Đã ghi đè file gốc: {input_path}")
    print("Hoàn tất!")


if __name__ == "__main__":
    clean_newlines_in_chunks()