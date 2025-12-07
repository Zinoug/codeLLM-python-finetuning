import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Para métricas
import evaluate  # asegúrate de instalar con: pip install evaluate rouge-score sacrebleu

# ===============================
# CONFIGURACIÓN
# ===============================

# Carpeta del checkpoint que quieres evaluar
MODEL_DIR = "codet5_multitask_ckpt/checkpoint-777"  # <-- AJUSTA si usas otro checkpoint

# Modelo base que usaste para entrenar (solo para el tokenizer)
TOKENIZER_SOURCE = "Salesforce/codet5-base"

# Ruta al archivo de test
TEST_PATH = "multitask_dataset/test.jsonl"

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

search_samples = [s for s in test_samples if s["input"].startswith("search code:")]
summary_samples = [s for s in test_samples if s["input"].startswith("summarize code:")]

print(" Code Search samples:", len(search_samples))
print(" Code Summary samples:", len(summary_samples))


# ===============================
# FUNCIÓN DE PREDICCIÓN
# ===============================

def predict_text(inp: str, max_new_tokens: int = 8) -> str:
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

print("\nEvaluating CODE SEARCH (Accuracy)...\n")

correct = 0
total = 0
errors = []

for s in search_samples:
    gold = s["output"].strip()   # "0", "1", "2"
    pred_text = predict_text(s["input"], max_new_tokens=4)

    # Extraer primer dígito de la predicción
    digits = [c for c in pred_text if c.isdigit()]
    pred = digits[0] if digits else None

    total += 1
    if pred == gold:
        correct += 1
    else:
        errors.append({
            "input": s["input"],
            "gold": gold,
            "pred_raw": pred_text,
            "pred_digit": pred,
        })

accuracy = correct / total if total > 0 else 0.0
print(f"RESULT → Code Search accuracy: {accuracy:.3f} ({correct}/{total})")

# Guardar errores por si quieres analizarlos
os.makedirs("eval_outputs", exist_ok=True)
errors_path = os.path.join("eval_outputs", "search_errors.json")
with open(errors_path, "w", encoding="utf8") as f:
    json.dump(errors, f, indent=2)
print(f"Saved mispredictions to {errors_path}")


# ===============================
# EVALUACIÓN: CODE SUMMARY (ROUGE + BLEU)
# ===============================

print("\nEvaluating CODE SUMMARY (ROUGE + BLEU)...\n")

# 1) Generar predicciones para TODOS los ejemplos de summary
summary_refs = [s["output"].strip() for s in summary_samples]
summary_inputs = [s["input"] for s in summary_samples]
summary_preds = []

for inp in summary_inputs:
    pred = predict_text(inp, max_new_tokens=48)  # un poco más largo para no truncar
    summary_preds.append(pred)

# 2) Cargar métricas
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

# 3) Calcular ROUGE
rouge_results = rouge.compute(
    predictions=summary_preds,
    references=summary_refs,
    use_stemmer=True
)

# 4) Calcular BLEU (necesita lista de listas de referencias)
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

# Opcional: guardar preds y refs por si quieres analizarlos
summary_eval_path = os.path.join("eval_outputs", "summary_preds_refs.json")
with open(summary_eval_path, "w", encoding="utf8") as f:
    json.dump(
        [{"pred": p, "ref": r} for p, r in zip(summary_preds, summary_refs)],
        f,
        indent=2
    )
print(f"\nSaved summary predictions & references to {summary_eval_path}")


# ===============================
# EJEMPLOS: CODE SUMMARY (GOLD vs PRED)
# ===============================

print("\nShowing 5 Code Summary examples:\n")

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
