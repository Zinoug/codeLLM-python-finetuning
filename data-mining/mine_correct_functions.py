"""
Script to mine correct functions with passing tests from GitHub repositories.
Extracts functions that already work correctly to train the model not to change working code.
"""

import os
import json
import subprocess
import ast
import re

REPOSITORIES = [
    "https://github.com/TheAlgorithms/Python",
]

INPUT_FILE = "data/REAL_TOP_LEVEL_BUGS.json"
OUTPUT_FILE = "data/REAL_TOP_LEVEL_BUGS.json"
CACHE_DIR = "mining_cache"
TARGET_NEW_SAMPLES = 200


def extract_functions(code):
    """Extract all top-level functions from code."""
    try:
        tree = ast.parse(code)
    except Exception:
        return []

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.col_offset == 0:
                start = node.lineno
                end = node.end_lineno
                lines = code.split("\n")
                func_code = "\n".join(lines[start - 1 : end])

                functions.append(
                    {
                        "name": node.name,
                        "code": func_code,
                        "start": start,
                        "end": end,
                    }
                )

    return functions


def generate_simple_tests(func_name, func_code):
    """Generate simple tests from doctest examples."""
    tests = []

    doctest_pattern = r">>>\s+(.+)\n\s*(.+)"
    matches = re.findall(doctest_pattern, func_code)

    for call, expected in matches:
        if func_name in call:
            test = f"assert {call.strip()} == {expected.strip()}"
            tests.append(test)

    return tests


def test_function(func_code, tests):
    """Test if code passes tests."""
    if not tests:
        return None

    test_script = func_code + "\n\n"
    for test in tests:
        test_script += test + "\n"

    try:
        result = subprocess.run(
            ["python3", "-c", test_script], capture_output=True, timeout=5, text=True
        )
        return result.returncode == 0
    except Exception:
        return None


def mine_correct_functions(repo_path, existing_functions, target_count):
    """Mine correct functions from repository main branch."""
    samples = []

    print(f"\n{'='*70}")
    print(f"🔍 Mining correct functions from main branch")
    print(f"{'='*70}")
    print(f"🎯 Target: {target_count} new samples")
    print(f"📝 Already have: {len(existing_functions)} existing samples")

    processed_files = 0

    # Walk through all Python files in the repo
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

        for filename in files:
            if not filename.endswith(".py"):
                continue

            if "test" in filename.lower():
                continue

            filepath = os.path.join(root, filename)
            relative_path = os.path.relpath(filepath, repo_path)

            processed_files += 1
            if processed_files % 100 == 0:
                print(
                    f"   ⏳ Processed {processed_files} files, found {len(samples)} samples..."
                )

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    code = f.read()
            except:
                continue

            functions = extract_functions(code)

            for func in functions:
                # Skip if already in dataset
                if func["name"] in existing_functions:
                    continue

                # Generate tests from doctests
                tests = generate_simple_tests(func["name"], func["code"])

                if not tests:
                    continue

                # Test if function passes its tests
                passes = test_function(func["code"], tests)

                if passes is True:
                    sample = {
                        "commit": "main",
                        "commit_msg": "Correct function - no bug",
                        "file": relative_path,
                        "function": func["name"],
                        "bug_type": "no_bug",
                        "pl": func["code"],
                        "fixed_code": func["code"],  # Same as buggy code
                        "tests": tests,
                    }

                    samples.append(sample)
                    existing_functions.add(func["name"])

                    print(f"✅ Found #{len(samples)}: {func['name']} ({relative_path})")

                    if len(samples) >= target_count:
                        print(
                            f"\n🎉 Target reached! Found {len(samples)} correct functions"
                        )
                        return samples

    print(
        f"\n📈 Mining complete: {len(samples)} correct functions from {processed_files} files"
    )
    return samples


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("🚀 Mining Correct Functions (No Bugs)")
    print("=" * 70)
    print(f"📦 Target: {TARGET_NEW_SAMPLES} new samples")
    print(f"💾 Output file: {OUTPUT_FILE}")
    print("=" * 70 + "\n")

    # Load existing dataset
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, "r") as f:
            existing_samples = json.load(f)
        print(f"📂 Loaded {len(existing_samples)} existing samples")
    else:
        existing_samples = []
        print(f"📂 No existing dataset found, starting fresh")

    # Get set of existing function names to avoid duplicates
    existing_functions = {s["function"] for s in existing_samples}

    # Mine from TheAlgorithms/Python repository
    repo_name = "Python"
    repo_path = os.path.join(CACHE_DIR, repo_name)

    if not os.path.exists(repo_path):
        print(f"❌ Repository cache not found: {repo_path}")
        print(f"💡 Run mine_top_level_bug_fixes.py first to clone the repository")
        return

    # Mine correct functions
    new_samples = mine_correct_functions(
        repo_path, existing_functions, TARGET_NEW_SAMPLES
    )

    # Combine with existing samples
    all_samples = existing_samples + new_samples

    # Save combined dataset
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_samples, f, indent=2)

    print("\n" + "=" * 70)
    print("🎉 MINING COMPLETE - RESULTS")
    print("=" * 70)
    print(f"📊 Previous samples: {len(existing_samples)}")
    print(f"➕ New correct samples: {len(new_samples)}")
    print(f"📦 Total samples: {len(all_samples)}")

    # Count bug types
    bug_types = {}
    for sample in all_samples:
        bug_type = sample.get("bug_type", "unknown")
        bug_types[bug_type] = bug_types.get(bug_type, 0) + 1

    print(f"\n🐛 Bug types distribution:")
    for bug_type, count in sorted(bug_types.items(), key=lambda x: -x[1]):
        print(f"   {bug_type:25s} : {count:4d} samples")

    print(f"\n💾 Output saved to: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
