import os
import re
import json
import ast
from typing import Dict, Any, List, Tuple

from pydriller import Repository, ModificationType
from pydriller.domain.commit import Commit

import doctest
import io
import contextlib


#############################
# CONFIGURACIÓN DEL MINING #
#############################

REPOS = [
    "https://github.com/TheAlgorithms/Python",
    "https://github.com/more-itertools/more-itertools",
    "https://github.com/python-string-utils/python-string-utils",
    "https://github.com/jmoiron/humanize",
    "https://github.com/mahmoud/boltons",
]

OUTPUT_DIR = "bug_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUG_DATASET_PATH = os.path.join(OUTPUT_DIR, "bug_dataset.jsonl")
CODE_REPAIR_PATH = os.path.join(OUTPUT_DIR, "code_repair.jsonl")
BUG_DETECTION_PATH = os.path.join(OUTPUT_DIR, "bug_detection.jsonl")

# Cualquier cosa que contenga "fix" o "bug" (fixed, bugfix, etc.)
COMMIT_MSG_REGEX = re.compile(r"(fix|bug)", re.IGNORECASE)


#########################
# UTILIDADES DE PARSING #
#########################

def extract_functions(source_code: str) -> Dict[str, str]:
    """
    Extrae solo funciones top-level de un archivo Python.

    Devuelve:
    {
        func_name: "def func(...):\\n    ...",
        ...
    }
    """
    if source_code is None:
        return {}

    functions = {}
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return {}

    lines = source_code.splitlines()

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):  # solo top-level
            start = node.lineno
            end = getattr(node, "end_lineno", None)
            if end is None:
                # fallback: desde start hasta antes del siguiente nodo (heurística)
                end = start
                for other in tree.body:
                    if other is node:
                        continue
                    if hasattr(other, "lineno") and other.lineno > start:
                        end = other.lineno - 1
                        break

            func_code = "\n".join(lines[start - 1: end])
            functions[node.name] = func_code

    return functions


#############################
# EJECUCIÓN DE DOCTESTS    #
#############################

def run_doctests_on_function_src(func_src: str, func_name: str) -> Tuple[int, int]:
    """
    Ejecuta doctests en el docstring de la función definida en `func_src`.
    NO se usa para filtrar, solo como metadatos.
    Retorna: (num_tests, num_failed)
    """
    ns = {}
    try:
        exec(func_src, ns)  # define la función en ns
    except Exception:
        return 0, 0

    if func_name not in ns or not callable(ns[func_name]):
        return 0, 0

    func_obj = ns[func_name]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            failures, tests = doctest.run_docstring_examples(
                func_obj,
                ns,
                name=func_name,
                verbose=False,
                optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
            )
        except Exception:
            return 0, 0

    return tests, failures


##########################
# FILTRO DE COMMITS      #
##########################

def commit_interesting(commit: Commit) -> bool:
    """True si el mensaje del commit menciona algo tipo fix/bug."""
    return bool(COMMIT_MSG_REGEX.search(commit.msg or ""))


##########################
# PIPELINE PRINCIPAL     #
##########################

def mine_bugs() -> List[Dict[str, Any]]:
    """
    Recorre los repos y devuelve una lista de samples con:
    {
      "repo": str,
      "commit": str,
      "file_path": str,
      "function_name": str,
      "buggy_code": str,
      "fixed_code": str,
      "n_tests_old": int,
      "n_failed_old": int,
      "n_tests_new": int,
      "n_failed_new": int,
    }
    """
    all_samples = []

    for repo_url in REPOS:
        print(f"\n=== Minando repo: {repo_url} ===")
        try:
            for commit in Repository(repo_url).traverse_commits():
                if not commit_interesting(commit):
                    continue

                print(f"- Commit candidato {commit.hash[:8]}: {commit.msg[:60]!r}")

                for mod in commit.modified_files:
                    if mod.change_type != ModificationType.MODIFY:
                        continue
                    if not mod.filename.endswith(".py"):
                        continue

                    old_src = mod.source_code_before
                    new_src = mod.source_code

                    if old_src is None or new_src is None:
                        continue

                    old_funcs = extract_functions(old_src)
                    new_funcs = extract_functions(new_src)

                    # Intersección por nombre
                    common_names = set(old_funcs.keys()) & set(new_funcs.keys())

                    for fname in common_names:
                        old_code = old_funcs[fname]
                        new_code = new_funcs[fname]

                        # el cuerpo debe haber cambiado
                        if old_code.strip() == new_code.strip():
                            continue

                        # Ejecutamos doctests solo como metadatos (pueden ser 0)
                        tests_old, failed_old = run_doctests_on_function_src(old_code, fname)
                        tests_new, failed_new = run_doctests_on_function_src(new_code, fname)

                        sample = {
                            "repo": repo_url,
                            "commit": commit.hash,
                            "file_path": mod.new_path or mod.old_path,
                            "function_name": fname,
                            "buggy_code": old_code,
                            "fixed_code": new_code,
                            "n_tests_old": tests_old,
                            "n_failed_old": failed_old,
                            "n_tests_new": tests_new,
                            "n_failed_new": failed_new,
                        }
                        all_samples.append(sample)

                        print(
                            f"  ✔ Sample en {mod.filename}:{fname} "
                            f"(commit={commit.hash[:8]}, tests_old={tests_old}, tests_new={tests_new})"
                        )

        except Exception as e:
            # Aquí caen cosas tipo credenciales / repos problemáticos
            print(f"  ⚠ Error procesando {repo_url}: {e}")
            continue

    print(f"\nTotal de samples encontrados: {len(all_samples)}")
    return all_samples


###############################
# GUARDAR DATASETS EN JSONL   #
###############################

def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_task_datasets(bug_samples: List[Dict[str, Any]]) -> None:
    """
    A partir del dataset base (bug_samples) genera:

    - bug_dataset.jsonl       : buggy + fixed + metadatos
    - code_repair.jsonl       : formato para Code Repair
    - bug_detection.jsonl     : formato para Bug Detection
    """
    # 1) dataset base
    write_jsonl(BUG_DATASET_PATH, bug_samples)

    # 2) Code Repair
    code_repair_records = []
    for s in bug_samples:
        # Pair básico buggy -> fixed
        code_repair_records.append({
            "task": "code_repair",
            "repo": s["repo"],
            "commit": s["commit"],
            "file_path": s["file_path"],
            "function_name": s["function_name"],
            "input": f"fix a bug:\n{s['buggy_code']}",
            "output": s["fixed_code"],
        })

        # Opcional: también fixed->fixed (identity mapping)
        code_repair_records.append({
            "task": "code_repair",
            "repo": s["repo"],
            "commit": s["commit"],
            "file_path": s["file_path"],
            "function_name": s["function_name"],
            "input": f"fix a bug:\n{s['fixed_code']}",
            "output": s["fixed_code"],
        })

    write_jsonl(CODE_REPAIR_PATH, code_repair_records)

    # 3) Bug Detection
    bug_detection_records = []
    for s in bug_samples:
        buggy_input = {
            "task": "bug_detection",
            "repo": s["repo"],
            "commit": s["commit"],
            "file_path": s["file_path"],
            "function_name": s["function_name"],
            "input": "classify BUGGY OR CORRECT:\n" + s["buggy_code"],
            "output": "BUGGY",
        }
        fixed_input = {
            "task": "bug_detection",
            "repo": s["repo"],
            "commit": s["commit"],
            "file_path": s["file_path"],
            "function_name": s["function_name"],
            "input": "classify BUGGY OR CORRECT:\n" + s["fixed_code"],
            "output": "CORRECT",
        }
        bug_detection_records.append(buggy_input)
        bug_detection_records.append(fixed_input)

    write_jsonl(BUG_DETECTION_PATH, bug_detection_records)


def main():
    bug_samples = mine_bugs()
    build_task_datasets(bug_samples)
    print("\n✅ Datasets generados:")
    print(f"- Base bugs      : {BUG_DATASET_PATH}")
    print(f"- Code Repair    : {CODE_REPAIR_PATH}")
    print(f"- Bug Detection  : {BUG_DETECTION_PATH}")


if __name__ == "__main__":
    main()
