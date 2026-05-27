# core/pipeline.py

import argparse
from typing import Dict, Any, Optional, List

from .retriever import RRFHybridRetriever

from .answer import (
    load_lora_model,
    generate_answer,
    ADAPTER_DIR
)

from control.context_checker import check_context
from control.answer_verifier import verify_answer
from control.confidence_gate import decide_action, build_user_response
from control.text_utils import infer_topic
from control.extractive_answer import (
    should_use_extractive_answer,
    format_extractive_answer,
    is_enumeration_question,
)

from knowledge.db import init_db
from knowledge.store import (
    search_confirmed_answers,
    add_answer_log,
    add_pending_question
)

from chat.history_store import ChatHistoryStore
from chat.question_rewriter import rewrite_question, rewrite_question_simple


def get_hit_attr(hit, name: str, default=None):
    return getattr(hit, name, default)


def build_multi_context_from_hits(
        hits,
        tokenizer,
        base_token_count: int = 0,
        max_total_tokens: int = 1024
):
    context_parts = []
    sources = []
    current_tokens = base_token_count

    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        metadata = chunk.get("metadata", {}) or {}

        chunk_id = chunk.get("chunk_id")
        page = chunk.get("page")
        title = metadata.get("title", "")

        content = chunk.get("content", "") or ""
        content = content.strip()

        if not content:
            continue

        part = (
            f"[Nguồn {i}]\n"
            f"chunk_id: {chunk_id}\n"
            f"page: {page}\n"
            f"title: {title}\n"
            f"content:\n{content}"
        )

        part_token_count = len(tokenizer.encode(part))

        if current_tokens + part_token_count > max_total_tokens:
            if len(context_parts) == 0:
                allowed_tokens = max_total_tokens - current_tokens
                if allowed_tokens > 50:
                    encoded_part = tokenizer.encode(part)[:allowed_tokens]
                    part = tokenizer.decode(encoded_part, skip_special_tokens=True)
                    context_parts.append(part)
                    sources.append({
                        "rank": i,
                        "chunk_id": chunk_id,
                        "page": page,
                        "title": title,
                        "content": content,
                        "retrieval_score": get_hit_attr(hit, "score", None),
                        "rrf_score": get_hit_attr(hit, "rrf_score", None),
                        "bm25_rank": get_hit_attr(hit, "bm25_rank", None),
                        "faiss_rank": get_hit_attr(hit, "faiss_rank", None),
                        "bm25_score": get_hit_attr(hit, "bm25_score", None),
                        "faiss_score": get_hit_attr(hit, "faiss_score", None),
                    })
            break

        context_parts.append(part)
        current_tokens += part_token_count

        sources.append({
            "rank": i,
            "chunk_id": chunk_id,
            "page": page,
            "title": title,
            "content": content,
            "retrieval_score": get_hit_attr(hit, "score", None),
            "rrf_score": get_hit_attr(hit, "rrf_score", None),
            "bm25_rank": get_hit_attr(hit, "bm25_rank", None),
            "faiss_rank": get_hit_attr(hit, "faiss_rank", None),
            "bm25_score": get_hit_attr(hit, "bm25_score", None),
            "faiss_score": get_hit_attr(hit, "faiss_score", None),
        })

    return "\n\n" + ("=" * 60 + "\n\n").join(context_parts), sources


def find_confirmed_answer(question: str, topic: Optional[str] = None) -> Optional[Dict[str, Any]]:
    results = search_confirmed_answers(
        query=question,
        topic=topic,
        limit=1
    )

    if results:
        return results[0] if isinstance(results, list) else results

    if topic:
        results = search_confirmed_answers(
            query=question,
            topic=None,
            limit=1
        )
        if results:
            return results[0] if isinstance(results, list) else results

    return None


def answer_question_controlled(
        question: str,
        retriever: RRFHybridRetriever,
        tokenizer,
        model,
        device: str,
        user_id: str = "anonymous",
        candidate_k: int = 30,
        top_k: int = 1,
        conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    init_db()
    topic = infer_topic(question)

    # Query rewriting (multi-turn) — trước cache và retrieval
    search_query = question
    history_store = ChatHistoryStore()

    if conversation_id:
        history = history_store.get_recent_history(conversation_id, k=3)
        if history:
            try:
                search_query = rewrite_question(
                    history=history,
                    current_query=question,
                    llm_model=model,
                    tokenizer=tokenizer,
                )
            except Exception as e:
                import logging
                logging.warning(
                    "Query rewriting failed: %s, using rule-based fallback", e
                )
                search_query = rewrite_question_simple(history, question)

    # 1. Ưu tiên câu trả lời đã xác nhận (dùng câu đã viết lại nếu có)
    confirmed = find_confirmed_answer(search_query, topic=topic)
    if confirmed is not None:
        final_answer = confirmed["canonical_answer"]
        log_id = add_answer_log(
            user_id=user_id,
            question=question,
            answer=final_answer,
            topic=topic,
            confidence="high",
            decision="use_confirmed_cache",
            status="active"
        )

        if conversation_id:
            history_store.add_message(conversation_id, "user", question)
            history_store.add_message(conversation_id, "assistant", final_answer)

        return {
            "question": question,
            "answer": final_answer,
            "raw_model_answer": None,
            "source_type": "confirmed_answer",
            "confirmed_answer_id": confirmed["id"],
            "answer_log_id": log_id,
            "topic": topic,
            "decision": "use_confirmed_cache",
            "confidence": "high",
            "context_check": None,
            "answer_check": None,
            "gate": {
                "decision": "use_confirmed_cache",
                "confidence": "high",
                "topic": topic,
                "reasons": ["Tìm thấy câu trả lời đã được xác nhận phù hợp với câu hỏi."]
            },
            "chunk": None,
            "pending_id": None
        }

    # 2. Retrieve top chunks (search_query đã được viết lại nếu có conversation_id)
    hits = retriever.search(
        search_query,
        top_k=top_k,
        candidate_k=candidate_k
    )

    if not hits:
        fallback_answer = (
            "Mình chưa tìm thấy ngữ cảnh phù hợp trong dữ liệu hiện tại. "
            "Câu hỏi này nên được chuyển cho thầy cô hoặc đơn vị phụ trách xác minh thêm."
        )
        context_check = {
            "context_status": "no_context",
            "context_score": 0.0,
            "reasons": ["Không tìm thấy chunk nào phù hợp với câu hỏi."]
        }
        answer_check = {
            "answer_status": "bad_answer",
            "answer_score": 0.0,
            "reasons": ["Không có context để sinh câu trả lời."]
        }
        gate = decide_action(
            question=question,
            context_check=context_check,
            answer_check=answer_check
        )
        log_id = add_answer_log(
            user_id=user_id,
            question=question,
            answer=fallback_answer,
            topic=topic,
            confidence=gate["confidence"],
            decision=gate["decision"],
            status="active"
        )
        pending_id = add_pending_question(
            user_id=user_id,
            question=question,
            topic=topic,
            context_snapshot=None,
            retrieved_chunk_id=None,
            reason="Không tìm thấy context phù hợp.",
            teacher_questions=gate.get("teacher_questions", ),
            status="pending"
        )

        if conversation_id:
            history_store.add_message(conversation_id, "user", question)
            history_store.add_message(conversation_id, "assistant", fallback_answer)

        return {
            "question": question,
            "answer": fallback_answer,
            "raw_model_answer": None,
            "source_type": "no_context",
            "answer_log_id": log_id,
            "topic": topic,
            "decision": gate["decision"],
            "confidence": gate["confidence"],
            "context_check": context_check,
            "answer_check": answer_check,
            "gate": gate,
            "chunk": None,
            "pending_id": pending_id
        }

    question_token_count = len(tokenizer.encode(question))

    context, sources = build_multi_context_from_hits(
        hits=hits,
        tokenizer=tokenizer,
        base_token_count=question_token_count,
        max_total_tokens=1024
    )

    top_hit = hits[0]
    chunk = top_hit.chunk

    chunk_id = chunk.get("chunk_id")
    page = chunk.get("page")

    retrieval_score = get_hit_attr(top_hit, "score", None)
    rrf_score = get_hit_attr(top_hit, "rrf_score", None)
    bm25_rank = get_hit_attr(top_hit, "bm25_rank", None)
    faiss_rank = get_hit_attr(top_hit, "faiss_rank", None)
    bm25_score = get_hit_attr(top_hit, "bm25_score", None)
    faiss_score = get_hit_attr(top_hit, "faiss_score", None)

    # 3. Check context trước
    context_check = check_context(
        question=question,
        chunk=chunk,
        retrieval_score=retrieval_score,
        bm25_rank=bm25_rank,
        faiss_rank=faiss_rank
    )

    # 4. Sinh câu trả lời: extractive (đủ ý) hoặc BARTpho-LoRA
    use_extractive = should_use_extractive_answer(question, context)
    prefer_complete = is_enumeration_question(question) and not use_extractive

    if use_extractive:
        raw_answer = format_extractive_answer(question, context)
        source_type = "extractive_passage"
    else:
        raw_answer = generate_answer(
            question=question,
            context=context,
            tokenizer=tokenizer,
            model=model,
            device=device,
            prefer_complete=prefer_complete,
        )
        source_type = "rag_generation"

    # 5. Verify answer
    answer_check = verify_answer(
        question=question,
        context=context,
        answer=raw_answer
    )

    # 6. Confidence Gate
    gate = decide_action(
        question=question,
        context_check=context_check,
        answer_check=answer_check
    )

    final_answer = build_user_response(
        answer=raw_answer,
        gate_result=gate
    )

    # 7. Lưu answer log
    log_id = add_answer_log(
        user_id=user_id,
        question=question,
        answer=final_answer,
        topic=gate["topic"],
        chunk_id=chunk_id,
        page=page,
        context_snapshot=context,
        retrieval_score=rrf_score if rrf_score is not None else retrieval_score,
        bm25_rank=bm25_rank,
        faiss_rank=faiss_rank,
        confidence=gate["confidence"],
        decision=gate["decision"],
        status="active"
    )

    # 8. Nếu không chắc thì lưu pending question
    pending_id = None

    if gate["decision"] == "ask_teacher":
        pending_id = add_pending_question(
            user_id=user_id,
            question=question,
            topic=gate["topic"],
            context_snapshot=context,
            retrieved_chunk_id=chunk_id,
            reason="; ".join(gate.get("reasons", )),
            teacher_questions=gate.get("teacher_questions", ),
            status="pending"
        )

    if conversation_id:
        history_store.add_message(conversation_id, "user", question)
        history_store.add_message(conversation_id, "assistant", final_answer)

    return {
        "question": question,
        "answer": final_answer,
        "raw_model_answer": raw_answer,
        "source_type": source_type,
        "answer_log_id": log_id,
        "topic": gate["topic"],
        "decision": gate["decision"],
        "confidence": gate["confidence"],
        "context_check": context_check,
        "answer_check": answer_check,
        "gate": gate,
        "chunk": {
            "chunk_id": chunk_id,
            "page": page,
            "title": (chunk.get("metadata", {}) or {}).get("title"),
            "content": context,
            "retrieval_score": retrieval_score,
            "rrf_score": rrf_score,
            "bm25_rank": bm25_rank,
            "faiss_rank": faiss_rank,
            "bm25_score": bm25_score,
            "faiss_score": faiss_score
        },
        "chunks": sources,
        "pending_id": pending_id
    }


def print_pipeline_result(result: Dict[str, Any], show_debug: bool = False, show_context: bool = False):
    print("\n" + "=" * 80)
    print("QUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nDECISION:")
    print("topic     :", result.get("topic"))
    print("decision  :", result.get("decision"))
    print("confidence:", result.get("confidence"))

    print("\nLOG:")
    print("answer_log_id:", result.get("answer_log_id"))
    print("pending_id   :", result.get("pending_id"))

    chunks = result.get("chunks") or []
    if chunks:
        print("\nSOURCES:")
        print("Total sources:", len(chunks))

        for c in chunks:
            print("-" * 80)
            print("rank      :", c.get("rank"))
            print("chunk_id  :", c.get("chunk_id"))
            print("page      :", c.get("page"))
            print("title     :", c.get("title"))
            print("rrf_score :", c.get("rrf_score"))
            print("bm25_rank :", c.get("bm25_rank"))
            print("faiss_rank:", c.get("faiss_rank"))

            if show_context:
                print("\nCONTENT:")
                print(c.get("content", "")[:1800])
    else:
        chunk = result.get("chunk")
        if chunk:
            print("\nSOURCE:")
            print("chunk_id :", chunk.get("chunk_id"))
            print("page     :", chunk.get("page"))
            print("title    :", chunk.get("title"))
            print("rrf_score:", chunk.get("rrf_score"))
            print("bm25_rank:", chunk.get("bm25_rank"))
            print("faiss_rank:", chunk.get("faiss_rank"))

            if show_context:
                print("\nCONTEXT:")
                print(chunk.get("content"))

    if show_debug:
        print("\nCONTEXT CHECK:")
        print(result.get("context_check"))
        print("\nANSWER CHECK:")
        print(result.get("answer_check"))
        print("\nGATE:")
        print(result.get("gate"))
        if result.get("raw_model_answer") is not None:
            print("\nRAW MODEL ANSWER:")
            print(result.get("raw_model_answer"))

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("question", type=str)
    parser.add_argument("--user_id", type=str, default="anonymous")

    parser.add_argument("--adapter_dir", type=str, default=ADAPTER_DIR)

    parser.add_argument("--candidate_k", type=int, default=30)
    parser.add_argument("--bm25_weight", type=float, default=0.2)
    parser.add_argument("--faiss_weight", type=float, default=0.8)
    parser.add_argument("--rrf_k", type=int, default=60)

    parser.add_argument("--show_debug", action="store_true")
    parser.add_argument("--show_context", action="store_true")

    args = parser.parse_args()

    init_db()

    print("Loading retriever...")
    retriever = RRFHybridRetriever(
        bm25_weight=args.bm25_weight,
        faiss_weight=args.faiss_weight,
        rrf_k=args.rrf_k
    )

    print("Loading BARTpho-LoRA...")
    tokenizer, model, device = load_lora_model(args.adapter_dir)

    result = answer_question_controlled(
        question=args.question,
        retriever=retriever,
        tokenizer=tokenizer,
        model=model,
        device=device,
        user_id=args.user_id,
        candidate_k=args.candidate_k,
        top_k=args.top_k
    )

    print_pipeline_result(
        result,
        show_debug=args.show_debug,
        show_context=args.show_context
    )


if __name__ == "__main__":
    main()