"""
Multi-Task CodeT5 Model Evaluation

This script evaluates a fine-tuned CodeT5 model on 4 tasks:
1. Code Search: Accuracy
2. Bug Detection: Accuracy
3. Code Summary: ROUGE + BLEU
4. Code Repair: pass@1 (unit test execution) + BLEU

The script uses the test split from final_data/test.jsonl
"""

import os
import json
import torch
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate


# ===============================
# CONFIGURATION
# ===============================

# Model checkpoint to evaluate
MODEL_DIR = "codet5_multitask_final"  # Adjust to your checkpoint path

# Base model for tokenizer
TOKENIZER_SOURCE = "Salesforce/codet5-base"

# Test data path
TEST_PATH = "final_data/test.jsonl"

# Output directory for results
OUTPUT_DIR = "eval_outputs"

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 70)
print("🔍 CodeT5 Multi-Task Model Evaluation")
print("=" * 70)
print(f"Device: {device}")


# ===============================
# LOAD MODEL AND TOKENIZER
# ===============================

print("\n📥 Loading model & tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SOURCE)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
model.eval()
print(f"   ✅ Model loaded from: {MODEL_DIR}")


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

print(f"   ✅ Total test samples: {len(test_samples)}")

# Separate by task
search_samples = [s for s in test_samples if s.get("task") == "search"]
summary_samples = [s for s in test_samples if s.get("task") == "summary"]
repair_samples = [s for s in test_samples if s.get("task") == "repair"]
detection_samples = [s for s in test_samples if s.get("task") == "detection"]

print(f"   • Code Search:     {len(search_samples)} samples")
print(f"   • Code Summary:    {len(summary_samples)} samples")
print(f"   • Code Repair:     {len(repair_samples)} samples")
print(f"   • Bug Detection:   {len(detection_samples)} samples")


# ===============================
# PREDICTION FUNCTION
# ===============================


def predict_text(inp: str, max_new_tokens: int = 32) -> str:
    """Generate prediction from model"""
    inputs = tokenizer(inp, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return text.strip()


# ===============================
# UTILITY: TEST EXECUTION
# ===============================


def execute_tests(code: str, tests: list) -> dict:
    """
    Execute unit tests on generated code

    Args:
        code: Generated Python code to test
        tests: List of assert statements

    Returns:
        dict with passed, failed, error counts and details
    """
    if not tests:
        return {"passed": 0, "failed": 0, "errors": 0, "total": 0, "details": []}

    passed = 0
    failed = 0
    errors = 0
    details = []

    # Create a safe execution environment
    exec_globals = {}

    # First, execute the code to define functions
    try:
        exec(code, exec_globals)
    except Exception as e:
        return {
            "passed": 0,
            "failed": len(tests),
            "errors": len(tests),
            "total": len(tests),
            "details": [
                {
                    "test": t,
                    "status": "error",
                    "message": f"Code execution failed: {str(e)}",
                }
                for t in tests
            ],
        }

    # Run each test
    for test in tests:
        try:
            # Capture output
            stdout_capture = StringIO()
            stderr_capture = StringIO()

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(test, exec_globals)

            passed += 1
            details.append({"test": test, "status": "passed", "message": ""})

        except AssertionError as e:
            failed += 1
            details.append({"test": test, "status": "failed", "message": str(e)})

        except Exception as e:
            errors += 1
            details.append({"test": test, "status": "error", "message": str(e)})

    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": len(tests),
        "details": details,
    }


# ===============================
# TASK 1: CODE SEARCH (ACCURACY)
# ===============================

print("\n" + "=" * 70)
print("📌 TASK 1: CODE SEARCH (Accuracy)")
print("=" * 70)

correct = 0
total = 0
errors_search = []

for s in search_samples:
    gold = s["output"].strip()
    pred_text = predict_text(s["input"], max_new_tokens=4)

    # Extract first digit from prediction
    digits = [c for c in pred_text if c.isdigit()]
    pred = digits[0] if digits else None

    total += 1
    if pred == gold:
        correct += 1
    else:
        errors_search.append(
            {
                "input": s["input"][:200],  # Truncate for readability
                "gold": gold,
                "pred_raw": pred_text,
                "pred_digit": pred,
            }
        )

accuracy_search = correct / total if total > 0 else 0.0
print(f"\n✅ Code Search Accuracy: {accuracy_search:.4f} ({correct}/{total})")

# Save errors
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "search_errors.json"), "w", encoding="utf8") as f:
    json.dump(errors_search, f, indent=2)
print(f"   💾 Errors saved to {OUTPUT_DIR}/search_errors.json")


# ===============================
# TASK 2: BUG DETECTION (ACCURACY)
# ===============================

print("\n" + "=" * 70)
print("📌 TASK 2: BUG DETECTION (Accuracy)")
print("=" * 70)


def normalize_bug_label(text: str) -> str:
    """Map prediction to 'BUGGY' or 'CORRECT'"""
    t = text.strip().upper()
    if "BUGGY" in t:
        return "BUGGY"
    if "CORRECT" in t or "CORRECTLY" in t:
        return "CORRECT"
    if "BUG" in t:
        return "BUGGY"
    if "OK" in t or "GOOD" in t:
        return "CORRECT"
    return t


correct = 0
total = 0
errors_detection = []

for s in detection_samples:
    gold = s["output"].strip().upper()
    pred_text = predict_text(s["input"], max_new_tokens=4)
    pred_norm = normalize_bug_label(pred_text)

    total += 1
    if pred_norm == gold:
        correct += 1
    else:
        errors_detection.append(
            {
                "input": s["input"][:200],
                "gold": gold,
                "pred_raw": pred_text,
                "pred_norm": pred_norm,
            }
        )

accuracy_detection = correct / total if total > 0 else 0.0
print(f"\n✅ Bug Detection Accuracy: {accuracy_detection:.4f} ({correct}/{total})")

with open(os.path.join(OUTPUT_DIR, "detection_errors.json"), "w", encoding="utf8") as f:
    json.dump(errors_detection, f, indent=2)
print(f"   💾 Errors saved to {OUTPUT_DIR}/detection_errors.json")


# ===============================
# TASK 3: CODE SUMMARY (ROUGE + BLEU)
# ===============================

print("\n" + "=" * 70)
print("📌 TASK 3: CODE SUMMARIZATION (ROUGE + BLEU)")
print("=" * 70)

summary_refs = [s["output"].strip() for s in summary_samples]
summary_inputs = [s["input"] for s in summary_samples]
summary_preds = []

print("   Generating summaries...")
for inp in summary_inputs:
    pred = predict_text(inp, max_new_tokens=64)
    summary_preds.append(pred)

# Compute ROUGE
rouge = evaluate.load("rouge")
rouge_results = rouge.compute(
    predictions=summary_preds, references=summary_refs, use_stemmer=True
)

# Compute BLEU
bleu = evaluate.load("bleu")
bleu_results = bleu.compute(
    predictions=summary_preds, references=[[ref] for ref in summary_refs]
)

print(f"\n✅ ROUGE Scores:")
print(f"   • ROUGE-1: {rouge_results['rouge1']:.4f}")
print(f"   • ROUGE-2: {rouge_results['rouge2']:.4f}")
print(f"   • ROUGE-L: {rouge_results['rougeL']:.4f}")

print(f"\n✅ BLEU Score: {bleu_results['bleu']:.4f}")

# Save predictions
with open(
    os.path.join(OUTPUT_DIR, "summary_predictions.json"), "w", encoding="utf8"
) as f:
    json.dump(
        [{"pred": p, "ref": r} for p, r in zip(summary_preds, summary_refs)],
        f,
        indent=2,
    )
print(f"\n   💾 Predictions saved to {OUTPUT_DIR}/summary_predictions.json")


# ===============================
# TASK 4: CODE REPAIR (pass@1 + BLEU)
# ===============================

print("\n" + "=" * 70)
print("📌 TASK 4: CODE REPAIR (pass@1 + BLEU)")
print("=" * 70)

repair_refs = [s["output"] for s in repair_samples]
repair_inputs = [s["input"] for s in repair_samples]
repair_preds = []
repair_results = []

print("   Generating code repairs...")
for i, (inp, sample) in enumerate(zip(repair_inputs, repair_samples)):
    pred = predict_text(inp, max_new_tokens=256)
    repair_preds.append(pred)

    # Execute tests if available
    tests = sample.get("tests", [])
    if tests:
        test_result = execute_tests(pred, tests)
        repair_results.append(
            {
                "sample_id": i,
                "input": inp[:200],
                "prediction": pred,
                "reference": sample["output"],
                "tests": tests,
                "test_results": test_result,
            }
        )

# Calculate pass@1 (percentage of samples that pass all tests)
samples_with_tests = [r for r in repair_results if r["test_results"]["total"] > 0]
if samples_with_tests:
    passed_all = sum(
        1
        for r in samples_with_tests
        if r["test_results"]["passed"] == r["test_results"]["total"]
    )
    pass_at_1 = passed_all / len(samples_with_tests)

    total_tests = sum(r["test_results"]["total"] for r in samples_with_tests)
    total_passed = sum(r["test_results"]["passed"] for r in samples_with_tests)

    print(
        f"\n✅ pass@1 Score: {pass_at_1:.4f} ({passed_all}/{len(samples_with_tests)} samples)"
    )
    print(f"   • Samples passing all tests: {passed_all}/{len(samples_with_tests)}")
    print(f"   • Individual tests passed: {total_passed}/{total_tests}")
else:
    print("\n⚠️  No test cases available for pass@1 evaluation")

# Exact Match
em_correct = sum(1 for p, r in zip(repair_preds, repair_refs) if p.strip() == r.strip())
em_score = em_correct / len(repair_refs) if repair_refs else 0.0
print(f"\n✅ Exact Match: {em_score:.4f} ({em_correct}/{len(repair_refs)})")

# BLEU
bleu_code = evaluate.load("bleu")
bleu_code_results = bleu_code.compute(
    predictions=repair_preds, references=[[ref] for ref in repair_refs]
)
print(f"✅ BLEU Score: {bleu_code_results['bleu']:.4f}")

# Save results
with open(
    os.path.join(OUTPUT_DIR, "repair_predictions.json"), "w", encoding="utf8"
) as f:
    json.dump(repair_results, f, indent=2)
print(
    f"\n   💾 Predictions with test results saved to {OUTPUT_DIR}/repair_predictions.json"
)


# ===============================
# SUMMARY REPORT
# ===============================

print("\n" + "=" * 70)
print("📊 EVALUATION SUMMARY")
print("=" * 70)

print(f"\n1️⃣  Code Search:")
print(f"   • Accuracy: {accuracy_search:.4f}")

print(f"\n2️⃣  Bug Detection:")
print(f"   • Accuracy: {accuracy_detection:.4f}")

print(f"\n3️⃣  Code Summarization:")
print(f"   • ROUGE-1: {rouge_results['rouge1']:.4f}")
print(f"   • ROUGE-2: {rouge_results['rouge2']:.4f}")
print(f"   • ROUGE-L: {rouge_results['rougeL']:.4f}")
print(f"   • BLEU: {bleu_results['bleu']:.4f}")

print(f"\n4️⃣  Code Repair:")
if samples_with_tests:
    print(f"   • pass@1: {pass_at_1:.4f}")
    print(f"   • Test pass rate: {total_passed}/{total_tests}")
print(f"   • Exact Match: {em_score:.4f}")
print(f"   • BLEU: {bleu_code_results['bleu']:.4f}")

print("\n" + "=" * 70)
print("✨ Evaluation completed!")
print("=" * 70)
print(f"\n📁 All results saved to: {OUTPUT_DIR}/")
