"""
Baseline vs Fine-tuned Model Comparison

This script compares the base CodeT5 model (not fine-tuned) with the fine-tuned version
on all 4 tasks to measure the improvement from fine-tuning.
"""

import os
import json
import torch
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from tqdm import tqdm


# ===============================
# CONFIGURATION
# ===============================

BASE_MODEL = "Salesforce/codet5-base"  # Model without fine-tuning
FINETUNED_MODEL = "codet5_multitask_final"  # Fine-tuned model

TEST_PATH = "final_data/test.jsonl"
OUTPUT_DIR = "eval_outputs"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================
# LOAD MODELS
# ===============================

print("=" * 70)
print("🔍 Baseline vs Fine-tuned Model Comparison")
print("=" * 70)
print(f"Device: {device}")

print("\n📥 Loading models...")

# Load tokenizer (same for both)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model
print("   Loading base model (Salesforce/codet5-base)...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)
base_model.eval()
print("   ✅ Base model loaded")

# Load fine-tuned model
print("   Loading fine-tuned model...")
finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(FINETUNED_MODEL).to(device)
finetuned_model.eval()
print("   ✅ Fine-tuned model loaded")


# ===============================
# LOAD TEST DATASET
# ===============================

print(f"\n📂 Loading test dataset: {TEST_PATH}")
test_samples = []
with open(TEST_PATH, "r", encoding="utf8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        test_samples.append(json.loads(line))

# Separate by task
search_samples = [s for s in test_samples if s.get("task") == "search"]
summary_samples = [s for s in test_samples if s.get("task") == "summary"]
repair_samples = [s for s in test_samples if s.get("task") == "repair"]
detection_samples = [s for s in test_samples if s.get("task") == "detection"]

print(f"   ✅ Total: {len(test_samples)} samples")
print(f"   • Search: {len(search_samples)}, Summary: {len(summary_samples)}")
print(f"   • Repair: {len(repair_samples)}, Detection: {len(detection_samples)}")


# ===============================
# PREDICTION FUNCTIONS
# ===============================

def predict_text(model, inp: str, max_new_tokens: int = 32) -> str:
    """Generate prediction from model"""
    inputs = tokenizer(inp, return_tensors="pt", truncation=True, max_length=768).to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return text.strip()


def execute_tests(code: str, tests: list) -> dict:
    """Execute unit tests on generated code"""
    if not tests:
        return {"passed": 0, "failed": 0, "errors": 0, "total": 0}

    passed = 0
    exec_globals = {}

    try:
        exec(code, exec_globals)
    except Exception:
        return {"passed": 0, "failed": len(tests), "errors": len(tests), "total": len(tests)}

    for test in tests:
        try:
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(test, exec_globals)
            passed += 1
        except:
            pass

    return {"passed": passed, "total": len(tests)}


# ===============================
# EVALUATION FUNCTIONS
# ===============================

def evaluate_search(model, samples, desc=""):
    """Evaluate code search accuracy"""
    correct = 0
    for s in tqdm(samples, desc=f"Search {desc}", leave=False):
        pred = predict_text(model, s["input"], max_new_tokens=4)
        if pred.strip() == s["output"].strip():
            correct += 1
    return correct / len(samples) if samples else 0


def evaluate_detection(model, samples, desc=""):
    """Evaluate bug detection accuracy"""
    correct = 0
    for s in tqdm(samples, desc=f"Detection {desc}", leave=False):
        pred = predict_text(model, s["input"], max_new_tokens=8)
        pred_label = "BUGGY" if "BUGGY" in pred.upper() else "CORRECT" if "CORRECT" in pred.upper() else pred
        expected = s["output"].strip().upper()
        if pred_label.upper() == expected:
            correct += 1
    return correct / len(samples) if samples else 0


def evaluate_summary(model, samples, desc=""):
    """Evaluate code summarization with ROUGE"""
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    
    predictions = []
    references = []
    
    for s in tqdm(samples, desc=f"Summary {desc}", leave=False):
        pred = predict_text(model, s["input"], max_new_tokens=64)
        predictions.append(pred)
        references.append(s["output"])
    
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])
    
    return {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bleu": bleu_score["bleu"]
    }


def evaluate_repair(model, samples, desc=""):
    """Evaluate code repair with pass@1"""
    passed_samples = 0
    total_tests_passed = 0
    total_tests = 0
    
    for s in tqdm(samples, desc=f"Repair {desc}", leave=False):
        pred = predict_text(model, s["input"], max_new_tokens=512)
        tests = s.get("tests", [])
        
        if tests:
            result = execute_tests(pred, tests)
            total_tests += result["total"]
            total_tests_passed += result["passed"]
            if result["passed"] == result["total"] and result["total"] > 0:
                passed_samples += 1
    
    return {
        "pass_at_1": passed_samples / len(samples) if samples else 0,
        "test_pass_rate": total_tests_passed / total_tests if total_tests > 0 else 0
    }


# ===============================
# MAIN COMPARISON
# ===============================

print("\n" + "=" * 70)
print("🏁 Starting Evaluation...")
print("=" * 70)

results = {
    "base": {},
    "finetuned": {}
}

# Task 1: Code Search
print("\n📌 TASK 1: CODE SEARCH")
print("-" * 40)
results["base"]["search"] = evaluate_search(base_model, search_samples, "(base)")
results["finetuned"]["search"] = evaluate_search(finetuned_model, search_samples, "(fine-tuned)")
print(f"   Base model:      {results['base']['search']*100:.2f}%")
print(f"   Fine-tuned:      {results['finetuned']['search']*100:.2f}%")
print(f"   Improvement:     {(results['finetuned']['search'] - results['base']['search'])*100:+.2f}%")

# Task 2: Bug Detection
print("\n📌 TASK 2: BUG DETECTION")
print("-" * 40)
results["base"]["detection"] = evaluate_detection(base_model, detection_samples, "(base)")
results["finetuned"]["detection"] = evaluate_detection(finetuned_model, detection_samples, "(fine-tuned)")
print(f"   Base model:      {results['base']['detection']*100:.2f}%")
print(f"   Fine-tuned:      {results['finetuned']['detection']*100:.2f}%")
print(f"   Improvement:     {(results['finetuned']['detection'] - results['base']['detection'])*100:+.2f}%")

# Task 3: Code Summary
print("\n📌 TASK 3: CODE SUMMARIZATION")
print("-" * 40)
results["base"]["summary"] = evaluate_summary(base_model, summary_samples, "(base)")
results["finetuned"]["summary"] = evaluate_summary(finetuned_model, summary_samples, "(fine-tuned)")
print(f"   Base model:      ROUGE-L={results['base']['summary']['rougeL']*100:.2f}%  BLEU={results['base']['summary']['bleu']*100:.2f}%")
print(f"   Fine-tuned:      ROUGE-L={results['finetuned']['summary']['rougeL']*100:.2f}%  BLEU={results['finetuned']['summary']['bleu']*100:.2f}%")
print(f"   Improvement:     ROUGE-L={((results['finetuned']['summary']['rougeL'] - results['base']['summary']['rougeL'])*100):+.2f}%")

# Task 4: Code Repair
print("\n📌 TASK 4: CODE REPAIR")
print("-" * 40)
results["base"]["repair"] = evaluate_repair(base_model, repair_samples, "(base)")
results["finetuned"]["repair"] = evaluate_repair(finetuned_model, repair_samples, "(fine-tuned)")
print(f"   Base model:      pass@1={results['base']['repair']['pass_at_1']*100:.2f}%")
print(f"   Fine-tuned:      pass@1={results['finetuned']['repair']['pass_at_1']*100:.2f}%")
print(f"   Improvement:     {(results['finetuned']['repair']['pass_at_1'] - results['base']['repair']['pass_at_1'])*100:+.2f}%")


# ===============================
# SUMMARY TABLE
# ===============================

print("\n" + "=" * 70)
print("📊 COMPARISON SUMMARY")
print("=" * 70)

print(f"""
┌─────────────────────┬──────────────┬──────────────┬──────────────┐
│ Task                │ Base Model   │ Fine-tuned   │ Improvement  │
├─────────────────────┼──────────────┼──────────────┼──────────────┤
│ Code Search (Acc)   │ {results['base']['search']*100:>10.2f}% │ {results['finetuned']['search']*100:>10.2f}% │ {(results['finetuned']['search']-results['base']['search'])*100:>+10.2f}% │
│ Bug Detection (Acc) │ {results['base']['detection']*100:>10.2f}% │ {results['finetuned']['detection']*100:>10.2f}% │ {(results['finetuned']['detection']-results['base']['detection'])*100:>+10.2f}% │
│ Summary (ROUGE-L)   │ {results['base']['summary']['rougeL']*100:>10.2f}% │ {results['finetuned']['summary']['rougeL']*100:>10.2f}% │ {(results['finetuned']['summary']['rougeL']-results['base']['summary']['rougeL'])*100:>+10.2f}% │
│ Code Repair (pass@1)│ {results['base']['repair']['pass_at_1']*100:>10.2f}% │ {results['finetuned']['repair']['pass_at_1']*100:>10.2f}% │ {(results['finetuned']['repair']['pass_at_1']-results['base']['repair']['pass_at_1'])*100:>+10.2f}% │
└─────────────────────┴──────────────┴──────────────┴──────────────┘
""")

# Save results
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "baseline_comparison.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"💾 Results saved to: {OUTPUT_DIR}/baseline_comparison.json")

print("\n" + "=" * 70)
print("✨ Comparison completed!")
print("=" * 70)
