import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Para métricas
import evaluate  # pip install evaluate rouge-score sacrebleu


# ===============================
# CONFIGURACIÓN
# ===============================

# Carpeta del checkpoint que quieres evaluar
MODEL_DIR = "codet5_multitask_ckptt/checkpoint-4000"  # <-- AJUSTA si usas otro checkpoint

# Modelo base que usaste para entrenar (solo para el tokenizer)
TOKENIZER_SOURCE = "Salesforce/codet5-base"

# Ruta al archivo de test (ojo con el nombre de la carpeta: 'multitask_datasett')
TEST_PATH = "multitask_datasett/test.jsonl"  # <-- cámbialo si renombraste la carpeta

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# ===============================
# CARGAR MODELO Y TOKENIZER
# ===============================

print("\nLoading model & tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SOURCE)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
model.eval()


# ===============================
# CARGAR DATASET DE TEST
# ===============================

print("\nLoading test dataset:", TEST_PATH)
test_samples = []
with open(TEST_PATH, "r", encoding="utf8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        test_samples.append(json.loads(line))

print("Total test samples:", len(test_samples))

# Separar por tarea usando el campo "task"
search_samples = [s for s in test_samples if s.get("task") == "code_search"]
summary_samples = [s for s in test_samples if s.get("task") == "code_summary"]
repair_samples = [s for s in test_samples if s.get("task") == "code_repair"]
bugdet_samples = [s for s in test_samples if s.get("task") == "bug_detection"]

print(" Code Search samples :", len(search_samples))
print(" Code Summary samples:", len(summary_samples))
print(" Code Repair samples :", len(repair_samples))
print(" Bug Detection samples:", len(bugdet_samples))


# ===============================
# FUNCIÓN DE PREDICCIÓN
# ===============================

def predict_text(inp: str, max_new_tokens: int = 32) -> str:
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
# EVALUACIÓN: CODE SEARCH (ACCURACY)
# ===============================

print("\n=== Evaluating CODE SEARCH (Accuracy) ===\n")

correct = 0
total = 0
errors_search = []

for s in search_samples:
    gold = s["output"].strip()   # "0", "1", "2", ...
    pred_text = predict_text(s["input"], max_new_tokens=4)

    # Extraer primer dígito de la predicción
    digits = [c for c in pred_text if c.isdigit()]
    pred = digits[0] if digits else None

    total += 1
    if pred == gold:
        correct += 1
    else:
        errors_search.append({
            "input": s["input"],
            "gold": gold,
            "pred_raw": pred_text,
            "pred_digit": pred,
        })

accuracy_search = correct / total if total > 0 else 0.0
print(f"RESULT → Code Search accuracy: {accuracy_search:.3f} ({correct}/{total})")

os.makedirs("eval_outputss", exist_ok=True)
errors_search_path = os.path.join("eval_outputss", "search_errors.json")
with open(errors_search_path, "w", encoding="utf8") as f:
    json.dump(errors_search, f, indent=2)
print(f"Saved Code Search mispredictions to {errors_search_path}")


# ===============================
# EVALUACIÓN: BUG DETECTION (ACCURACY)
# ===============================

print("\n=== Evaluating BUG DETECTION (Accuracy) ===\n")

correct = 0
total = 0
errors_bugdet = []

def normalize_bug_label(text: str):
    """
    Intenta mapear la predicción a 'BUGGY' o 'CORRECT' usando heurísticas simples.
    """
    t = text.strip().upper()
    # Buscar palabras clave
    if "BUGGY" in t:
        return "BUGGY"
    if "CORRECT" in t or "CORRECTLY" in t:
        return "CORRECT"
    # fallback: si contiene 'BUG', lo contamos como BUGGY
    if "BUG" in t:
        return "BUGGY"
    # si contiene 'OK' o 'GOOD', lo contamos como CORRECT
    if "OK" in t or "GOOD" in t:
        return "CORRECT"
    # último recurso: devolver tal cual
    return t

for s in bugdet_samples:
    gold = s["output"].strip().upper()  # "BUGGY" o "CORRECT"
    pred_text = predict_text(s["input"], max_new_tokens=4)
    pred_norm = normalize_bug_label(pred_text)

    total += 1
    if pred_norm == gold:
        correct += 1
    else:
        errors_bugdet.append({
            "input": s["input"],
            "gold": gold,
            "pred_raw": pred_text,
            "pred_norm": pred_norm,
        })

accuracy_bugdet = correct / total if total > 0 else 0.0
print(f"RESULT → Bug Detection accuracy: {accuracy_bugdet:.3f} ({correct}/{total})")

errors_bugdet_path = os.path.join("eval_outputss", "bug_detection_errors.json")
with open(errors_bugdet_path, "w", encoding="utf8") as f:
    json.dump(errors_bugdet, f, indent=2)
print(f"Saved Bug Detection mispredictions to {errors_bugdet_path}")


# ===============================
# EVALUACIÓN: CODE SUMMARY (ROUGE + BLEU)
# ===============================

print("\n=== Evaluating CODE SUMMARY (ROUGE + BLEU) ===\n")

summary_refs = [s["output"].strip() for s in summary_samples]
summary_inputs = [s["input"] for s in summary_samples]
summary_preds = []

for inp in summary_inputs:
    pred = predict_text(inp, max_new_tokens=48)  # más largo para no truncar
    summary_preds.append(pred)

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

rouge_results = rouge.compute(
    predictions=summary_preds,
    references=summary_refs,
    use_stemmer=True
)

bleu_results = bleu.compute(
    predictions=summary_preds,
    references=[[ref] for ref in summary_refs]
)

print("ROUGE results:")
print(f"  ROUGE-1: {rouge_results['rouge1']:.4f}")
print(f"  ROUGE-2: {rouge_results['rouge2']:.4f}")
print(f"  ROUGE-L: {rouge_results['rougeL']:.4f}")

print("\nBLEU results:")
print(f"  BLEU-4: {bleu_results['bleu']:.4f}")

summary_eval_path = os.path.join("eval_outputss", "summary_preds_refs.json")
with open(summary_eval_path, "w", encoding="utf8") as f:
    json.dump(
        [{"pred": p, "ref": r} for p, r in zip(summary_preds, summary_refs)],
        f,
        indent=2
    )
print(f"\nSaved summary predictions & references to {summary_eval_path}")


# ===============================
# EVALUACIÓN: CODE REPAIR (Exact Match + BLEU)
# ===============================

print("\n=== Evaluating CODE REPAIR (Exact Match + BLEU) ===\n")

repair_refs = [s["output"] for s in repair_samples]   # código fixed
repair_inputs = [s["input"] for s in repair_samples]  # "fix a bug:\n<buggy/fixed_code>"
repair_preds = []

for inp in repair_inputs:
    # para reparación de código queremos permitir más tokens
    pred = predict_text(inp, max_new_tokens=256)
    repair_preds.append(pred)

# Exact match (string)
em_correct = sum(1 for p, r in zip(repair_preds, repair_refs) if p.strip() == r.strip())
em_total = len(repair_refs)
em_score = em_correct / em_total if em_total > 0 else 0.0

print(f"Exact Match (Code Repair): {em_score:.4f} ({em_correct}/{em_total})")

# BLEU sobre código (tokenizado por espacios; no es perfecto pero sirve como proxy)
bleu_code = evaluate.load("bleu")
bleu_code_results = bleu_code.compute(
    predictions=repair_preds,
    references=[[ref] for ref in repair_refs]
)

print(f"BLEU (Code Repair): {bleu_code_results['bleu']:.4f}")

repair_eval_path = os.path.join("eval_outputss", "code_repair_preds_refs.json")
with open(repair_eval_path, "w", encoding="utf8") as f:
    json.dump(
        [{"pred": p, "ref": r} for p, r in zip(repair_preds, repair_refs)],
        f,
        indent=2
    )
print(f"Saved code repair predictions & references to {repair_eval_path}")


# ===============================
# EJEMPLOS: CODE SUMMARY (GOLD vs PRED)
# ===============================

print("\n=== Showing 5 Code Summary examples ===\n")

num_examples = min(5, len(summary_samples))
for i in range(num_examples):
    gold = summary_refs[i]
    pred = summary_preds[i]

    print("=" * 80)
    print(f"Example {i}")
    print("- GOLD:")
    print(gold)
    print("- PRED:")
    print(pred)

print("\nDone.")
