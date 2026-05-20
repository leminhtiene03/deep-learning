# rag_answer.py
# RAG pipeline: RRF retriever top-1 chunk -> BARTpho-LoRA answer generation
from huggingface_hub import snapshot_download
from transformers import BartphoTokenizer
import os
import sys
import argparse
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

from retriever_rrf import RRFHybridRetriever


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


BASE_MODEL = "vinai/bartpho-syllable"

# Sửa đường dẫn này theo nơi bạn lưu LoRA adapter
# Ví dụ Colab/Drive:
# ADAPTER_DIR = "/content/drive/MyDrive/PTIT/lora_bartpho_rag_train_final/best_lora_adapter"
#
# Ví dụ Windows local:
# ADAPTER_DIR = "./models/bartpho_lora_adapter"
ADAPTER_DIR = "./models/bartpho_lora_adapter"

MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 256


def load_lora_model(adapter_dir: str):
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(
            f"Không tìm thấy LoRA adapter tại: {adapter_dir}\n"
            "Hãy sửa ADAPTER_DIR hoặc truyền --adapter_dir."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Downloading/loading BARTpho files...")
    base_dir = snapshot_download(
        repo_id=BASE_MODEL,
        local_dir="./hf_models/bartpho-syllable",
        local_dir_use_symlinks=False
    )

    vocab_file = os.path.join(base_dir, "sentencepiece.bpe.model")
    monolingual_vocab_file = os.path.join(base_dir, "dict.txt")

    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Missing vocab_file: {vocab_file}")

    if not os.path.exists(monolingual_vocab_file):
        raise FileNotFoundError(f"Missing monolingual_vocab_file: {monolingual_vocab_file}")

    print("Loading BartphoTokenizer manually...")
    tokenizer = BartphoTokenizer(
        vocab_file=vocab_file,
        monolingual_vocab_file=monolingual_vocab_file
    )

    print("Loading base model from:", base_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_dir)

    print("Loading LoRA adapter:", adapter_dir)
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    model.config.decoder_start_token_id = model.config.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.to(device)
    model.eval()

    print("Device:", device)

    return tokenizer, model, device

def build_source(question: str, context: str) -> str:
    # Giữ format giống train_final:
    # "câu hỏi: ... context: ..."
    return f"câu hỏi: {question.strip()} context: {context.strip()}"


def generate_answer(
    question: str,
    context: str,
    tokenizer,
    model,
    device: str,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    source = build_source(question, context)

    inputs = tokenizer(
        source,
        return_tensors="pt",
        max_length=max_input_tokens,
        truncation=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            early_stopping=True
        )

    answer = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    ).strip()

    return answer


def answer_question(
    question: str,
    retriever: RRFHybridRetriever,
    tokenizer,
    model,
    device: str,
    candidate_k: int = 30,
):
    # Chỉ lấy top 1 chunk
    hits = retriever.search(
        question,
        top_k=1,
        candidate_k=candidate_k
    )

    if not hits:
        return {
            "question": question,
            "answer": "Mình chưa tìm thấy ngữ cảnh phù hợp để trả lời chắc chắn.",
            "chunk": None
        }

    hit = hits[0]
    chunk = hit.chunk

    context = chunk.get("content", "")

    answer = generate_answer(
        question=question,
        context=context,
        tokenizer=tokenizer,
        model=model,
        device=device
    )

    return {
        "question": question,
        "answer": answer,
        "chunk": {
            "chunk_id": chunk.get("chunk_id"),
            "page": chunk.get("page"),
            "title": (chunk.get("metadata", {}) or {}).get("title", ""),
            "score": hit.score,
            "rrf_score": hit.rrf_score,
            "bm25_rank": hit.bm25_rank,
            "faiss_rank": hit.faiss_rank,
            "bm25_score": hit.bm25_score,
            "faiss_score": hit.faiss_score,
            "content": context
        }
    }


def print_result(result: dict, show_context: bool = False):
    print("\n" + "=" * 80)
    print("QUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])

    chunk = result.get("chunk")

    if chunk is None:
        print("\nSOURCE: None")
        return

    print("\nSOURCE:")
    print(f"chunk_id   : {chunk['chunk_id']}")
    print(f"page       : {chunk['page']}")
    print(f"title      : {chunk['title']}")
    print(f"score      : {chunk['score']:.6f}")
    print(f"rrf_score  : {chunk['rrf_score']:.6f}")
    print(f"bm25_rank  : {chunk['bm25_rank']}")
    print(f"faiss_rank : {chunk['faiss_rank']}")

    if show_context:
        print("\nCONTEXT:")
        print(chunk["content"])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "question",
        type=str,
        help="Câu hỏi cần trả lời"
    )

    parser.add_argument(
        "-c",
        "--candidate_k",
        type=int,
        default=30,
        help="Số candidate lấy từ BM25/FAISS trước khi RRF"
    )

    parser.add_argument(
        "--bm25_weight",
        type=float,
        default=0.2,
        help="Trọng số BM25 trong RRF"
    )

    parser.add_argument(
        "--faiss_weight",
        type=float,
        default=0.8,
        help="Trọng số FAISS trong RRF"
    )

    parser.add_argument(
        "--rrf_k",
        type=int,
        default=60,
        help="RRF k"
    )

    parser.add_argument(
        "--show_context",
        action="store_true",
        help="In full top-1 context"
    )

    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=ADAPTER_DIR,
        help="Đường dẫn LoRA adapter"
    )

    args = parser.parse_args()

    print("Loading retriever...")
    retriever = RRFHybridRetriever(
        bm25_weight=args.bm25_weight,
        faiss_weight=args.faiss_weight,
        rrf_k=args.rrf_k
    )

    tokenizer, model, device = load_lora_model(args.adapter_dir)

    result = answer_question(
        question=args.question,
        retriever=retriever,
        tokenizer=tokenizer,
        model=model,
        device=device,
        candidate_k=args.candidate_k
    )

    print_result(result, show_context=args.show_context)


if __name__ == "__main__":
    main()