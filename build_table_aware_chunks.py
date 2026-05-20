# build_table_aware_chunks.py
# Create table-aware chunks from chunks_new.json
# Original chunks_new.json is NOT modified.

import json
import re
import copy
from pathlib import Path


INPUT_PATH = "chunks_new.json"
OUTPUT_PATH = "chunks_table_aware.json"
OUTPUT_TABLE_ONLY_PATH = "chunks_table_only.json"
# Nếu False: chunk bảng gốc sẽ được thay bằng các chunk con.
# Nếu True : giữ cả chunk bảng gốc + thêm chunk con.
# Khuyên dùng False để tránh retriever vẫn lấy bảng dài gây nhiễu.
KEEP_PARENT_TABLE_CHUNKS = False


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_space(text):
    text = str(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_curriculum_table_chunk(chunk):
    meta = chunk.get("metadata", {}) or {}
    title = str(meta.get("title", "")).lower()
    content = str(chunk.get("content", "")).lower()

    signals = [
        "kế hoạch và tiến trình học tập chuẩn",
        "tên môn học/học phần",
        "số tc",
        "học kỳ",
        "hk1",
        "hk2",
    ]

    text = title + "\n" + content

    score = sum(1 for s in signals if s in text)
    return score >= 3


def extract_program_and_cohort(content):
    """
    Extract:
    NGÀNH CÔNG NGHỆ THÔNG TIN KHÓA 2025
    """
    text = normalize_space(content)

    program = ""
    cohort = ""

    m = re.search(
        r"NGÀNH\s+(.+?)\s+KHÓA\s+(\d{4})",
        text,
        flags=re.IGNORECASE
    )

    if m:
        program = m.group(1).strip().title()
        cohort = m.group(2).strip()

    return program, cohort


def clean_course_name(name):
    name = str(name)

    # Remove leading row number
    name = re.sub(r"^\s*\d+\s+", "", name)

    # Remove table headers if accidentally captured
    bad_prefixes = [
        "TT Tên môn học/học phần Số TC Học kỳ",
        "TT Tên môn học/học phần",
        "Tên môn học/học phần",
        "Số TC",
        "Học kỳ",
    ]

    for p in bad_prefixes:
        name = name.replace(p, "")

    name = normalize_space(name)

    # Remove duplicated spaces/newlines
    name = name.strip(" -–—:;,.")
    return name


def looks_like_bad_course_name(name):
    if not name:
        return True

    low = name.lower()

    bad_phrases = [
        "kế hoạch và tiến trình",
        "ngành công nghệ",
        "năm học thứ",
        "tt tên môn",
        "số tc",
        "học kỳ",
        "tổng học phí",
    ]

    if any(p in low for p in bad_phrases):
        return True

    if len(name) < 3:
        return True

    return False


def extract_courses_by_semester(content):
    """
    Extract course records of form:
    <course name> <credits> HKx

    This works better than line-based parsing because the original table
    may have two semesters side by side in the same line.
    """
    text = normalize_space(content)

    # Put space around HK markers
    text = re.sub(r"(HK\s*\d+)", lambda m: m.group(1).replace(" ", ""), text, flags=re.IGNORECASE)

    # Regex idea:
    # optional row number + course name + credits + HKx
    #
    # Use non-greedy match until a number before HK.
    pattern = re.compile(
        r"(?:^|\s)(?:\d+\s+)?"
        r"(?P<name>[A-ZÀ-Ỹa-zà-ỹ0-9\(\)\+\/\.,&\- ]{3,}?)"
        r"\s+(?P<credits>\d+)\s+HK(?P<semester>\d+)",
        flags=re.IGNORECASE
    )

    semesters = {}

    for m in pattern.finditer(text):
        name = clean_course_name(m.group("name"))
        credits = m.group("credits")
        semester = f"HK{m.group('semester')}"

        if looks_like_bad_course_name(name):
            continue

        semesters.setdefault(semester, [])

        item = {
            "name": name,
            "credits": credits,
            "semester": semester,
            "source": "regex"
        }

        # Avoid duplicates
        if not any(x["name"].lower() == name.lower() for x in semesters[semester]):
            semesters[semester].append(item)

    return semesters


def add_known_non_credit_courses(content, semesters):
    """
    Some curriculum PDF/table rows are broken, e.g.
    'Giáo dục thể chất 1' may appear without explicit 'HK1' marker.
    This helper adds common non-credit/special rows into the right semester
    if they exist in the parent chunk.
    """
    text = content.lower()

    known = [
        ("HK1", "Giáo dục thể chất 1"),
        ("HK1", "Giáo dục quốc phòng"),
        ("HK1", "Kỹ năng mềm 1"),
        ("HK2", "Giáo dục thể chất 2"),
        ("HK2", "Kỹ năng mềm 2"),
        ("HK3", "Giáo dục thể chất 3"),
        ("HK3", "Kỹ năng mềm 3"),
        ("HK4", "Giáo dục thể chất 4"),
        ("HK4", "Kỹ năng mềm 4"),
    ]

    for sem, name in known:
        if name.lower() in text:
            semesters.setdefault(sem, [])

            already = any(
                x["name"].lower() == name.lower()
                for x in semesters[sem]
            )

            if not already:
                semesters[sem].append({
                    "name": name,
                    "credits": "",
                    "semester": sem,
                    "source": "known_non_credit"
                })

    return semesters


def sort_semester_key(sem):
    m = re.search(r"\d+", sem)
    return int(m.group()) if m else 999


def estimate_total_credits(courses):
    total = 0
    for c in courses:
        if c.get("credits", "").isdigit():
            total += int(c["credits"])
    return total


def build_semester_chunk(parent_chunk, semester, courses, new_chunk_id):
    parent_meta = parent_chunk.get("metadata", {}) or {}
    parent_title = parent_meta.get("title", "") or "KẾ HOẠCH VÀ TIẾN TRÌNH HỌC TẬP CHUẨN"

    parent_content = parent_chunk.get("content", "")
    program, cohort = extract_program_and_cohort(parent_content)

    total_credits = estimate_total_credits(courses)

    lines = []
    lines.append(parent_title)
    if program:
        lines.append(f"Ngành: {program}")
    if cohort:
        lines.append(f"Khóa: {cohort}")

    lines.append(f"Học kỳ: {semester}")
    lines.append("")
    lines.append("Danh sách môn học/học phần:")

    for i, c in enumerate(courses, start=1):
        name = c["name"]
        credits = c.get("credits", "")

        if credits:
            lines.append(f"{i}. {name} - {credits} TC")
        else:
            lines.append(f"{i}. {name}")

    if total_credits > 0:
        lines.append("")
        lines.append(f"Tổng số tín chỉ có ghi trong bảng: {total_credits} TC")

    lines.append("")
    lines.append(
        f"Nguồn: parent_chunk_id={parent_chunk.get('chunk_id')}, "
        f"page={parent_chunk.get('page')}"
    )

    content = "\n".join(lines).strip()

    metadata = copy.deepcopy(parent_meta)
    metadata.update({
        "doc_type": "curriculum_table_semester",
        "parent_chunk_id": parent_chunk.get("chunk_id"),
        "parent_title": parent_title,
        "program": program,
        "cohort": cohort,
        "semester": semester,
        "title": f"{parent_title} - {program} - {semester}".strip(" -")
    })

    return {
        "chunk_id": new_chunk_id,
        "content": content,
        "char_count": len(content),
        "word_count": len(content.split()),
        "page": parent_chunk.get("page"),
        "priority": "high",
        "metadata": metadata
    }


def split_curriculum_chunk(parent_chunk, start_chunk_id):
    content = parent_chunk.get("content", "")

    semesters = extract_courses_by_semester(content)
    semesters = add_known_non_credit_courses(content, semesters)

    child_chunks = []
    next_id = start_chunk_id

    for semester in sorted(semesters.keys(), key=sort_semester_key):
        courses = semesters[semester]

        if not courses:
            continue

        child = build_semester_chunk(
            parent_chunk=parent_chunk,
            semester=semester,
            courses=courses,
            new_chunk_id=next_id
        )

        child_chunks.append(child)
        next_id += 1

    return child_chunks, next_id


def rebuild_chunk_ids(chunks):
    """
    Ensure chunk_id is continuous and unique.
    Also keep original_chunk_id for traceability.
    """
    rebuilt = []

    for new_id, chunk in enumerate(chunks):
        c = copy.deepcopy(chunk)

        old_id = c.get("chunk_id")
        meta = c.get("metadata", {}) or {}

        if "original_chunk_id" not in meta:
            meta["original_chunk_id"] = old_id

        c["chunk_id"] = new_id
        c["metadata"] = meta

        # Recalculate counts
        content = c.get("content", "")
        c["char_count"] = len(content)
        c["word_count"] = len(content.split())

        rebuilt.append(c)

    return rebuilt


def main():
    input_path = Path(INPUT_PATH)

    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_PATH}")

    chunks = load_json(INPUT_PATH)

    print("Input chunks:", len(chunks))

    new_chunks = []
    table_parent_count = 0
    table_child_count = 0

    temp_new_id = 10_000_000

    for chunk in chunks:
        if is_curriculum_table_chunk(chunk):
            table_parent_count += 1

            child_chunks, temp_new_id = split_curriculum_chunk(
                parent_chunk=chunk,
                start_chunk_id=temp_new_id
            )

            if KEEP_PARENT_TABLE_CHUNKS:
                parent_copy = copy.deepcopy(chunk)
                parent_meta = parent_copy.get("metadata", {}) or {}
                parent_meta["doc_type"] = "curriculum_table_parent"
                parent_meta["table_split"] = True
                parent_copy["metadata"] = parent_meta
                parent_copy["priority"] = "low"
                new_chunks.append(parent_copy)

            if child_chunks:
                new_chunks.extend(child_chunks)
                table_child_count += len(child_chunks)
            else:
                # Nếu tách bảng thất bại, giữ lại chunk gốc để không mất dữ liệu.
                fallback = copy.deepcopy(chunk)
                fallback_meta = fallback.get("metadata", {}) or {}
                fallback_meta["table_split_failed"] = True
                fallback_meta["doc_type"] = "curriculum_table_split_failed"
                fallback["metadata"] = fallback_meta
                new_chunks.append(fallback)
        else:
            new_chunks.append(chunk)

    # Rebuild lại chunk_id cho file full output
    new_chunks = rebuild_chunk_ids(new_chunks)

    # Lọc riêng các chunk dạng bảng sau khi đã rebuild ID
    # Như vậy chunk_id trong chunks_table_only.json khớp với chunks_table_aware.json
    table_only_chunks = []

    for c in new_chunks:
        meta = c.get("metadata", {}) or {}
        doc_type = meta.get("doc_type", "")

        if doc_type in {
            "curriculum_table_semester",
            "curriculum_table_parent",
            "curriculum_table_split_failed"
        }:
            table_only_chunks.append(c)

    save_json(new_chunks, OUTPUT_PATH)
    save_json(table_only_chunks, OUTPUT_TABLE_ONLY_PATH)

    print("Table parent chunks detected:", table_parent_count)
    print("Table child chunks created :", table_child_count)
    print("Output chunks:", len(new_chunks))
    print("Table-only chunks:", len(table_only_chunks))
    print("Saved full output:", OUTPUT_PATH)
    print("Saved table-only :", OUTPUT_TABLE_ONLY_PATH)

    # In thử vài chunk bảng để kiểm tra nhanh
    print("\nSample table chunks:")
    shown = 0

    for c in table_only_chunks:
        meta = c.get("metadata", {}) or {}

        print("\n" + "=" * 80)
        print("chunk_id:", c["chunk_id"])
        print("doc_type:", meta.get("doc_type"))
        print("title:", meta.get("title"))
        print(c["content"][:1200])

        shown += 1
        if shown >= 5:
            break


if __name__ == "__main__":
    main()