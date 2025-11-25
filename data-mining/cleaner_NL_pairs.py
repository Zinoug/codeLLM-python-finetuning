import json

INPUT = "output/CODE_NL_PAIRS.json"
OUTPUT = "output/CODE_NL_PAIRS_FILTERED.json"

def is_valid_sample(sample):

    name = sample["function"]
    code = sample["code"]
    doc = sample["docstring"].strip()

    # ---- 1. Remove test functions ----
    if name.startswith("test_"):
        return False

    # ---- 2. Remove docstrings related to tests or GitHub issues ----
    BAD_DOC_PATTERNS = ["test", "pytest", "issue", "github", "exercise", "example:"]
    if any(bad in doc.lower() for bad in BAD_DOC_PATTERNS):
        return False

    # ---- 3. Remove functions too long ----
    if len(code.split("\n")) > 80:
        return False

    # ---- 4. Remove docstrings that are too short or low quality ----
    if len(doc.split()) < 4:
        return False

    # ---- 5. Remove docstrings containing only examples or weird text ----
    if doc.lower().startswith(">>>"):
        return False

    # ---- 6. No URLs please ----
    if "http" in doc.lower():
        return False

    return True


def main():
    with open(INPUT, "r") as f:
        data = json.load(f)

    clean = [s for s in data if is_valid_sample(s)]

    print(f"Original: {len(data)} samples")
    print(f"Filtered: {len(clean)} samples")

    with open(OUTPUT, "w") as f:
        json.dump(clean, f, indent=2)

    print(f"Saved to {OUTPUT}")

main()
