import json
import os
import random

# ---------------------------
# CONFIG
# ---------------------------

INPUT_FILE = "output/CODE_NL_PAIRS_FILTERED.json"

OUT_DIR = "multitask_dataset"
TRAIN_FILE = os.path.join(OUT_DIR, "train.jsonl")
VAL_FILE = os.path.join(OUT_DIR, "val.jsonl")
TEST_FILE = os.path.join(OUT_DIR, "test.jsonl")

# fracción de split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15  # el resto será test

# número de elecciones para Code Search (1 correcta + 2 distractores)
NUM_CHOICES = 3
NUM_DISTRACTORS = NUM_CHOICES - 1

RNG_SEED = 42


# ---------------------------
# HELPERS
# ---------------------------

def load_pairs(path):
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data


def normalize_summary(docstring: str) -> str:
    """
    Saca un resumen corto de la docstring:
    - usa el primer párrafo
    - aplasta saltos de línea en espacios
    """
    doc = docstring.strip()
    if not doc:
        return ""

    # Primer párrafo (separado por doble salto de línea)
    paragraph = doc.split("\n\n")[0].strip()
    # Reemplazar multiple espacios/nuevas líneas por un espacio
    summary = " ".join(paragraph.split())
    return summary


def make_code_summary_samples(pairs):
    """
    Devuelve una lista de dicts con:
    {
      "input": "summarize code:\n<code>",
      "output": "<short natural language summary>"
    }
    """
    samples = []
    for p in pairs:
        code = p["code"]
        doc = p["docstring"]
        summary = normalize_summary(doc)

        # por si alguna docstring quedó vacía después de normalizar
        if not summary:
            continue

        input_text = f"summarize code:\n{code}"
        output_text = summary

        samples.append({
            "input": input_text,
            "output": output_text
        })

    return samples


def make_code_search_samples(pairs):
    """
    Genera ejemplos tipo:

    input:
        search code:
        <summary>
        Choices:
        0: <code0>
        1: <code1>
        2: <code2>

    output:
        "índice correcto"
    """
    random.seed(RNG_SEED)

    n = len(pairs)
    indices = list(range(n))
    samples = []

    for i, p in enumerate(pairs):
        doc = p["docstring"]
        query = normalize_summary(doc)
        if not query:
            # si no hay buen texto NL, no sirve para Code Search
            continue

        correct_code = p["code"]

        # elegir distractores
        candidates = [j for j in indices if j != i]
        if len(candidates) < NUM_DISTRACTORS:
            # dataset muy pequeño, saltar por seguridad
            continue

        distractor_indices = random.sample(candidates, NUM_DISTRACTORS)
        distractor_codes = [pairs[j]["code"] for j in distractor_indices]

        # juntar todos los códigos (correcto + distractores) y barajarlos
        all_codes = distractor_codes + [correct_code]
        random.shuffle(all_codes)

        # encontrar índice donde quedó el correcto
        correct_index = all_codes.index(correct_code)

        # construir texto de input con Choices
        choices_lines = []
        for idx, code_choice in enumerate(all_codes):
            # cada choice en una línea
            choices_lines.append(f"{idx}: {code_choice}")

        input_text = (
            "search code:\n"
            f"{query}\n"
            "Choices:\n" +
            "\n".join(choices_lines)
        )

        output_text = str(correct_index)

        samples.append({
            "input": input_text,
            "output": output_text
        })

    return samples


def split_dataset(samples, train_ratio=0.7, val_ratio=0.15, seed=RNG_SEED):
    random.seed(seed)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]

    return train, val, test


def save_jsonl(path, samples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


# ---------------------------
# MAIN
# ---------------------------

def main():
    print("=== PREPARING MULTI-TASK DATASET (CODE SUMMARY + CODE SEARCH) ===\n")

    print(f"Loading pairs from: {INPUT_FILE}")
    pairs = load_pairs(INPUT_FILE)
    print(f"Total code-docstring pairs: {len(pairs)}")

    # 1) Code Summary
    summary_samples = make_code_summary_samples(pairs)
    print(f"Code Summary samples: {len(summary_samples)}")

    # 2) Code Search
    search_samples = make_code_search_samples(pairs)
    print(f"Code Search samples: {len(search_samples)}")

    # 3) Mezclar ambos tasks en un solo dataset multitarea
    all_samples = summary_samples + search_samples
    print(f"Total multitask samples (summary + search): {len(all_samples)}")

    # 4) Split train/val/test
    train, val, test = split_dataset(all_samples, TRAIN_RATIO, VAL_RATIO)

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # 5) Guardar como JSONL
    save_jsonl(TRAIN_FILE, train)
    save_jsonl(VAL_FILE, val)
    save_jsonl(TEST_FILE, test)

    print(f"\nSaved:")
    print(f"  - {TRAIN_FILE}")
    print(f"  - {VAL_FILE}")
    print(f"  - {TEST_FILE}")
    print("\nDone ✓")


if __name__ == "__main__":
    main()
