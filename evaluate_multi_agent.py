# evaluate_multi_agent.py
from datasets import load_dataset
from agents import (
    UserProxy,
    QueryRefiner,
    Retriever,
    Generator,
    Evaluator,
    DynamicPassageSelector,
)

from script_opt_ollama import (
    retrieve,
    chat_refiner,
    chat_generator,
    chat_evaluator,
    chat_selector,
    TOP_K,
    TOP_N_RETRIEVAL,
    TOP_K_MAX,
    TOP_K_MIN,
)

import time
import json
import csv
import os
import re
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

PDF_PATHS = [
    "pdfs/qasper_paper_1.pdf",
    "pdfs/qasper_paper_2.pdf",
    "pdfs/qasper_paper_3.pdf",
    "pdfs/qasper_paper_4.pdf",
    "pdfs/qasper_paper_5.pdf",
]

DATASET_INDICES = [0, 1, 2, 3, 4]


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

LOG_DIR = "eval_logs"
os.makedirs(LOG_DIR, exist_ok=True)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

JSON_LOG_PATH = os.path.join(LOG_DIR, f"multi_agent_{RUN_TIMESTAMP}.json")
CSV_LOG_PATH = os.path.join(LOG_DIR, f"multi_agent_{RUN_TIMESTAMP}.csv")

CSV_FIELDS = [
    "pdf_path",
    "paper_title",
    "question_index",
    "question",
    "ground_truth",
    "refined_query",
    "rag_answer",
    "attempts",
    "elapsed_seconds",
    "retrieval_score",
    "generation_score",
    "summary"
]


# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────

def extract_score(text):
    match = re.search(r"\b([0-9]|10)\b", text)
    if match:
        return int(match.group(1)) / 10.0
    return 0.0


def init_csv(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv_row(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def save_json_log(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────
# Judges (Ollama evaluator)
# ─────────────────────────────────────────────────────────────

def judge_answer(question, ground_truth, rag_answer):

    prompt = f"""
You are a strict RAG evaluator.

Question:
{question}

Ground Truth:
{ground_truth}

RAG Answer:
{rag_answer}

Score accuracy from 0 to 10.

Rules:
- exact match = 10
- partially correct = 5-9
- wrong = 0
- hallucination = 0

Return ONLY a number between 0 and 10.
"""

    response, _ = chat_evaluator(
        [{"role": "user", "content": prompt}]
    )

    generation_score = extract_score(response)
    print(f"generation score: {generation_score}")
    return generation_score


def judge_retrieval(question, ground_truth, retrieved_chunks):

    chunks_text = "\n---\n".join(
        c[0] if isinstance(c, (list, tuple)) else str(c)
        for c in retrieved_chunks
    )

    prompt = f"""
You are a strict RAG retrieval evaluator.

Question:
{question}

Ground Truth:
{ground_truth}

Retrieved Context:
{chunks_text}

Does the retrieved context contain the answer?

Score from 0 to 10.

10 = answer clearly present
5 = partially present
0 = not present

Return ONLY a number between 0 and 10.
"""

    response, _ = chat_evaluator(
        [{"role": "user", "content": prompt}]
    )

    retrieval_score = extract_score(response)
    print(f"retrieval score: {retrieval_score}")
    return retrieval_score


# ─────────────────────────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────────────────────────

print("Loading QASper dataset...")
dataset = load_dataset("allenai/qasper", split="validation", trust_remote_code=True)

init_csv(CSV_LOG_PATH)

global_total = 0
global_gen = 0
global_ret = 0
global_count = 0

full_log = {
    "run_timestamp": RUN_TIMESTAMP,
    "papers": [],
}


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────

for pdf_path, ds_idx in zip(PDF_PATHS, DATASET_INDICES):

    item = dataset[ds_idx]

    paper_title = item["title"]
    questions = item["qas"]["question"]
    answers_data = item["qas"]["answers"]

    print("\n" + "=" * 60)
    print("PDF:", pdf_path)
    print("Title:", paper_title)
    print("=" * 60)

    paper_total = 0
    paper_gen = 0
    paper_ret = 0

    paper_records = []

    for i in range(len(questions)):

        query = questions[i]

        ground_truth = "Unknown"
        if answers_data[i] and answers_data[i]["answer"]:
            ans = answers_data[i]["answer"][0]

            if ans.get("free_form_answer"):
                ground_truth = ans["free_form_answer"]

            elif ans.get("extractive_spans"):
                ground_truth = ", ".join(ans["extractive_spans"])

            elif ans.get("yes_no") is not None:
                ground_truth = str(ans["yes_no"])

        print("\nQuestion:", query)
        print("GT:", ground_truth)

        state = {
            "query": query,
            "answer": "",
            "score": 0.0,
            "attempts": 0,
            "chunks": [],
            "should_retry": False,
            "model_used": "",
            "refined_query": "",
        }

        proxy = UserProxy(
            refiner=QueryRefiner(chat_fn=chat_refiner),

            retriever=Retriever(
                retrieve_fn=retrieve,
                top_k=TOP_K,
                top_n=TOP_N_RETRIEVAL,
            ),

            selector=DynamicPassageSelector(
                chat_fn=chat_selector,
                max_passages=TOP_K_MAX,
                min_passages=TOP_K_MIN,
            ),

            generator=Generator(chat_fn=chat_generator),

            evaluator=Evaluator(
                chat_fn=chat_evaluator,
                min_score=0.75,
            ),
        )

        start = time.time()
        state = proxy.run(state)
        elapsed = time.time() - start

        num_chunks = len(state["chunks"])
        num_candidates = state.get("dps_n_candidates", "?")

        print(f"Retrieved chunks (after DPS): {num_chunks}")
        print(f"Candidates before DPS: {num_candidates}")

        print("Answer:", state["answer"])
        print("Attempts:", state["attempts"])




        retrieval_score = judge_retrieval(
            query,
            ground_truth,
            state["chunks"]
        )

        generation_score = judge_answer(
            query,
            ground_truth,
            state["answer"]
        )

        paper_ret += retrieval_score
        paper_gen += generation_score
        paper_total += generation_score

        record = {
            "pdf_path": pdf_path,
            "paper_title": paper_title,
            "question_index": i + 1,
            "question": query,
            "ground_truth": ground_truth,
            "refined_query": state["refined_query"],
            "rag_answer": state["answer"],
            "attempts": state["attempts"],
            "elapsed_seconds": round(elapsed, 2),
            "retrieval_score": retrieval_score,
            "generation_score": generation_score,
            "summary": ""
        }

        paper_records.append(record)
        append_csv_row(CSV_LOG_PATH, record)

    n = len(questions)

    avg_total = paper_total / n
    avg_gen = paper_gen / n
    avg_ret = paper_ret / n

    print("\nPaper Results:")
    print("Total:", avg_total)
    print("Generation:", avg_gen)
    print("Retrieval:", avg_ret)

    global_total += paper_total
    global_gen += paper_gen
    global_ret += paper_ret
    global_count += n

    full_log["papers"].append({
        "pdf_path": pdf_path,
        "paper_title": paper_title,
        "results": paper_records,
    })

    save_json_log(JSON_LOG_PATH, full_log)


# ─────────────────────────────────────────────────────────────
# Global summary
# ─────────────────────────────────────────────────────────────

print("\n===== FINAL =====")

avg_total = global_total / global_count
avg_gen = global_gen / global_count
avg_ret = global_ret / global_count

print("Total:", avg_total)
print("Generation:", avg_gen)
print("Retrieval:", avg_ret)

summary_row = {
    "pdf_path": "SUMMARY",
    "paper_title": "ALL",
    "question_index": "",
    "question": "",
    "ground_truth": "",
    "refined_query": "",
    "rag_answer": "",
    "attempts": "",
    "elapsed_seconds": "",
    "retrieval_score": avg_ret,
    "generation_score": avg_gen,
    "summary": f"accuracy={avg_gen:.3f} retrieval={avg_ret:.3f}"
}

append_csv_row(CSV_LOG_PATH, summary_row)

save_json_log(JSON_LOG_PATH, full_log)