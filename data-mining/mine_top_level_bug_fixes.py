"""
Script to mine real bug fixes from GitHub repositories.
Extracts top-level functions with their buggy and fixed versions.
"""

import os
import json
import subprocess
import ast
import re
from pydriller import Repository

REPOSITORIES = [
    "https://github.com/TheAlgorithms/Python",
    "https://github.com/keon/algorithms",
    "https://github.com/OmkarPathak/pygorithm",
    "https://github.com/geekcomputers/Python",
    "https://github.com/zhiwehu/Python-programming-exercises",
]

OUTPUT_FILE = "data/REAL_TOP_LEVEL_BUGS.json"
CACHE_DIR = "mining_cache"


def is_bug_fix(msg):
    """Check if commit message indicates a bug fix."""
    # keywords = ["fix", "bug", "error", "issue", "fault", "defect", "patch"]
    # if not any(keyword in msg.lower() for keyword in keywords):
    #     return False
    return True


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


def extract_unittest_tests(func_name, test_file_content):
    """Extract unittest assertions for a specific function."""
    tests = []

    patterns = [
        rf"self\.assertEqual\({func_name}\(([^)]+)\),\s*([^)]+)\)",
        rf"self\.assertEqual\(([^,]+),\s*{func_name}\(([^)]+)\)\)",
        rf"assert {func_name}\(([^)]+)\)\s*==\s*([^\n]+)",
        rf"assert ([^\n]+)\s*==\s*{func_name}\(([^)]+)\)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, test_file_content, re.MULTILINE)
        for match in matches:
            if len(match) == 2:
                args, expected = match
                args = args.strip()
                expected = expected.strip().rstrip(",").rstrip(")")
                test = f"assert {func_name}({args}) == {expected}"
                tests.append(test)

    return tests


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
    """Test if code passes/fails tests."""
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


def classify_bug(buggy_code, fixed_code):
    """Classify the type of bug between buggy and fixed versions."""
    if "range(" in buggy_code and "range(" in fixed_code:
        if buggy_code.count("range(") == fixed_code.count("range("):
            return "off_by_one"

    if ("= 0" in buggy_code and "= 1" in fixed_code) or (
        "= 1" in buggy_code and "= 0" in fixed_code
    ):
        return "wrong_initialization"

    if (" < " in buggy_code and " <= " in fixed_code) or (
        " <= " in buggy_code and " < " in fixed_code
    ):
        return "boundary_condition"
    if (" > " in buggy_code and " >= " in fixed_code) or (
        " >= " in buggy_code and " > " in fixed_code
    ):
        return "boundary_condition"

    if ("and" in buggy_code and "or" in fixed_code) or (
        "or" in buggy_code and "and" in fixed_code
    ):
        return "logic_error"

    if fixed_code.count("if") > buggy_code.count("if"):
        return "missing_edge_case"

    buggy_returns = re.findall(r"return\s+([^\n]+)", buggy_code)
    fixed_returns = re.findall(r"return\s+([^\n]+)", fixed_code)
    if buggy_returns and fixed_returns and buggy_returns != fixed_returns:
        return "wrong_return_value"

    return "other"


def mine_repo(repo_url):
    """Mine repository for bug fixes."""
    repo_name = repo_url.split("/")[-1]
    print(f"\n{'='*70}")
    print(f"🔍 Mining: {repo_name}")
    print(f"{'='*70}")

    repo_path = os.path.join(CACHE_DIR, repo_name)

    if not os.path.exists(repo_path):
        print(f"📥 Cloning repository {repo_name}...")
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth=5000", repo_url, repo_path],
                capture_output=True,
                timeout=600,
            )
            print(f"✅ Clone completed for {repo_name}")
        except subprocess.TimeoutExpired:
            print(f"⏱️  Clone timeout - skipping {repo_name}")
            return [], {}
    else:
        print(f"💾 Using cached repository: {repo_name}")

    samples = []
    bug_types = {}
    sample_count = 0
    commit_count = 0

    try:
        repo = Repository(repo_path, only_modifications_with_file_types=[".py"])

        print(f"📊 Analyzing commits in {repo_name}...")
        for commit in repo.traverse_commits():
            commit_count += 1
            if commit_count % 50 == 0:
                print(
                    f"   ⏳ Processed {commit_count} commits, found {sample_count} samples..."
                )
            try:
                if commit.hash.startswith("5f4da5d6"):
                    continue

                test_files_content = []
                for test_mod in commit.modified_files:
                    if (
                        test_mod.filename.endswith(".py")
                        and "test" in test_mod.filename.lower()
                    ):
                        if test_mod.source_code:
                            test_files_content.append(test_mod.source_code)

                for mod in commit.modified_files:
                    if not mod.filename.endswith(".py"):
                        continue

                    if "test" in mod.filename.lower():
                        continue

                    if not mod.source_code or not mod.source_code_before:
                        continue

                    buggy_funcs = extract_functions(mod.source_code_before)
                    fixed_funcs = extract_functions(mod.source_code)

                    for buggy_func in buggy_funcs:
                        for fixed_func in fixed_funcs:
                            if buggy_func["name"] == fixed_func["name"]:
                                if buggy_func["code"] == fixed_func["code"]:
                                    continue

                                if (
                                    abs(
                                        len(buggy_func["code"])
                                        - len(fixed_func["code"])
                                    )
                                    > 500
                                ):
                                    continue

                                tests = generate_simple_tests(
                                    buggy_func["name"], fixed_func["code"]
                                )

                                for test_content in test_files_content:
                                    unittest_tests = extract_unittest_tests(
                                        buggy_func["name"], test_content
                                    )
                                    tests.extend(unittest_tests)

                                if not tests:
                                    continue

                                buggy_passes = test_function(buggy_func["code"], tests)
                                fixed_passes = test_function(fixed_func["code"], tests)

                                if buggy_passes is False and fixed_passes is True:
                                    sample_count += 1

                                    bug_type = classify_bug(
                                        buggy_func["code"], fixed_func["code"]
                                    )
                                    bug_types[bug_type] = bug_types.get(bug_type, 0) + 1

                                    sample = {
                                        "commit": commit.hash[:8],
                                        "commit_msg": commit.msg[:100],
                                        "file": mod.new_path,
                                        "function": buggy_func["name"],
                                        "bug_type": bug_type,
                                        "pl": buggy_func["code"],
                                        "fixed_code": fixed_func["code"],
                                        "tests": tests,
                                    }

                                    samples.append(sample)

                                    print(
                                        f"✅ Found #{sample_count}: {bug_type} in {buggy_func['name']} ({mod.new_path})"
                                    )

            except Exception:
                continue

    except Exception as e:
        print(f"❌ Error mining {repo_name}: {e}")

    print(
        f"\n📈 {repo_name} Summary: {len(samples)} samples from {commit_count} commits"
    )
    return samples, bug_types


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("🚀 Mining Bug Fixes from GitHub Repositories")
    print("=" * 70)
    print(f"📦 Target repositories: {len(REPOSITORIES)}")
    print(f"💾 Output file: {OUTPUT_FILE}")
    print("=" * 70 + "\n")

    os.makedirs("data", exist_ok=True)

    all_samples = []
    all_bug_types = {}
    repo_stats = {}

    for idx, repo_url in enumerate(REPOSITORIES, 1):
        print(f"\n[{idx}/{len(REPOSITORIES)}] Processing repository...")
        repo_name = repo_url.split("/")[-1]

        samples, bug_types = mine_repo(repo_url)

        if samples:
            repo_stats[repo_name] = len(samples)
            all_samples.extend(samples)

        for bug_type, count in bug_types.items():
            all_bug_types[bug_type] = all_bug_types.get(bug_type, 0) + count

        with open(OUTPUT_FILE, "w") as f:
            json.dump(all_samples, f, indent=2)

        print(f"\n💾 Saved progress: {len(all_samples)} total samples")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_samples, f, indent=2)

    print("\n" + "=" * 70)
    print("🎉 MINING COMPLETE - RESULTS")
    print("=" * 70)
    print(f"📊 Total samples: {len(all_samples)}")
    print(f"\n🐛 Bug types found:")
    for bug_type, count in sorted(all_bug_types.items(), key=lambda x: -x[1]):
        print(f"   {bug_type:25s} : {count:4d} samples")
    print(f"\n📦 Samples per repository:")
    for repo, count in sorted(repo_stats.items(), key=lambda x: -x[1]):
        print(f"   {repo:30s} : {count:4d} samples")
    print(f"\n💾 Output saved to: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
