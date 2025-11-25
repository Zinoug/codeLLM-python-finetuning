import os
import subprocess
import ast
import json

# ---------------------------------------
# REPOS TO MINE
# ---------------------------------------

REPO_URLS = [
    "https://github.com/TheAlgorithms/Python",
    "https://github.com/more-itertools/more-itertools",
    "https://github.com/python-string-utils/python-string-utils",
    "https://github.com/jmoiron/humanize",
    "https://github.com/mahmoud/boltons"
]

REPO_DIR = "repos"
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "CODE_NL_PAIRS.json")


# ---------------------------------------
# STEP 1 — CLONE REPOSITORIES
# ---------------------------------------

def clone_repositories():
    os.makedirs(REPO_DIR, exist_ok=True)

    for url in REPO_URLS:
        name = url.split("/")[-1]
        dest = os.path.join(REPO_DIR, name)

        if not os.path.exists(dest):
            print(f"\nCloning {name}...")
            subprocess.run(["git", "clone", "--depth", "1", url, dest])
        else:
            print(f"{name} already exists — skipping.")


# ---------------------------------------
# STEP 2 — EXTRACT CODE + DOCSTRINGS
# ---------------------------------------

def extract_pairs_from_file(path):
    try:
        with open(path, "r", encoding="utf8") as f:
            source = f.read()
    except:
        return []

    try:
        tree = ast.parse(source)
    except Exception:
        return []

    pairs = []
    lines = source.split("\n")

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):

            doc = ast.get_docstring(node)
            if not doc:
                continue

            # simple validation: docstring should have meaning
            if len(doc.split()) < 3:
                continue

            start = node.lineno - 1
            end = node.end_lineno
            code = "\n".join(lines[start:end])

            pairs.append({
                "function": node.name,
                "code": code,
                "docstring": doc.strip()
            })

    return pairs


def extract_from_repositories():
    all_samples = []

    for repo_name in os.listdir(REPO_DIR):
        repo_path = os.path.join(REPO_DIR, repo_name)

        print(f"\nProcessing repo: {repo_name}")

        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    samples = extract_pairs_from_file(file_path)
                    all_samples.extend(samples)

    return all_samples


# ---------------------------------------
# MAIN
# ---------------------------------------

def main():
    print("==== DATASET 2 MINER ====\n")

    clone_repositories()

    print("\nExtracting function-docstring pairs...")
    samples = extract_from_repositories()
    print(f"\nTotal extracted: {len(samples)} samples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf8") as f:
        json.dump(samples, f, indent=2)

    print(f"\nSaved to: {OUTPUT_FILE}")
    print("\nDone ✓")


if __name__ == "__main__":
    main()
