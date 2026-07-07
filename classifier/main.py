import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("heuristic_config.json")
SCAN_CHUNK_CHARS = 1800
SCAN_CHUNK_OVERLAP = 200
SCAN_MAX_FILE_BYTES = 5 * 1024 * 1024
SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "training_data",
}
TEXT_EXTENSIONS = {
    ".bat",
    ".c",
    ".cfg",
    ".conf",
    ".cpp",
    ".cs",
    ".css",
    ".csv",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".log",
    ".md",
    ".php",
    ".ps1",
    ".py",
    ".rb",
    ".rst",
    ".rs",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
DEFAULT_CONFIG = {
    "threshold": 0.35,
    "weights": {
        "repeated_lines": 1.7,
        "log_patterns": 3.2,
        "blob_like": 2.8,
        "long_lines": 0.2,
        "minified_shape": 2.0,
        "low_alpha": 1.2,
        "numeric_csv": 1.8,
        "random_ids": 1.0,
        "metadata_dump": 1.0,
        "prose": -2.1,
        "document_ext": -0.8,
        "code_shape": -1.0,
        "code_or_vocab_list": -1.4,
        "structured_labels": -0.7,
    },
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return DEFAULT_CONFIG


def parse_folder_command(command: str) -> Path | None:
    if not command.lower().startswith("h "):
        return None
    try:
        parts = shlex.split(command, posix=False)
    except ValueError:
        return None
    if len(parts) < 2 or parts[0].lower() != "h":
        return None
    return Path(" ".join(parts[1:]).strip().strip('"')).expanduser()


def looks_binary(sample: bytes) -> bool:
    if b"\x00" in sample:
        return True
    if not sample:
        return False
    printable = sum(byte in b"\t\r\n" or 32 <= byte <= 126 for byte in sample)
    return printable / len(sample) < 0.75


def read_text_file(path: Path) -> str | None:
    try:
        if path.stat().st_size > SCAN_MAX_FILE_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None

    if path.suffix.lower() not in TEXT_EXTENSIONS and looks_binary(data[:4096]):
        return None

    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def iter_text_files(folder: Path):
    for root, dirs, files in os.walk(folder):
        dirs[:] = [name for name in dirs if name not in SKIP_DIR_NAMES]
        for file_name in files:
            path = Path(root) / file_name
            if path.name.startswith(".") and path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            yield path


def chunk_text(text: str, chunk_size: int = SCAN_CHUNK_CHARS, overlap: int = SCAN_CHUNK_OVERLAP):
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    if not cleaned.strip():
        return
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            yield start, end, chunk
        if end == len(cleaned):
            break
        start = max(0, end - overlap)


def entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def repeated_line_ratio(lines: list[str]) -> float:
    useful_lines = [line.strip() for line in lines if len(line.strip()) >= 8]
    if not useful_lines:
        return 0.0
    return 1.0 - (len(set(useful_lines)) / len(useful_lines))


def extract_features(chunk: str, path: Path) -> dict:
    lower = f" {chunk.lower()} "
    lines = chunk.splitlines()
    alpha_ratio = sum(char.isalpha() for char in chunk) / max(1, len(chunk))
    digit_ratio = sum(char.isdigit() for char in chunk) / max(1, len(chunk))
    whitespace_ratio = sum(char.isspace() for char in chunk) / max(1, len(chunk))
    avg_line_len = sum(len(line) for line in lines) / max(1, len(lines))
    syntax_noise = sum(chunk.count(char) for char in "{}[];,")
    newline_ratio = chunk.count("\n") / max(1, len(chunk))
    words = re.findall(r"[a-zA-Z]{3,}", chunk)
    sentence_marks = sum(chunk.count(mark) for mark in ".!?")

    log_patterns = [
        " traceback ",
        " stack trace ",
        " exception ",
        " error ",
        " warning ",
        " debug ",
        " failed ",
        " status=500 ",
        " request_id=",
        " errno ",
        " panic ",
    ]
    code_terms = [
        "def ",
        "class ",
        "function ",
        "import ",
        "return ",
        "const ",
        "public ",
        "private ",
        "elif ",
        "if ",
        "for ",
        "try:",
        "except",
        "with ",
        "select ",
        "where ",
    ]
    metadata_terms = [
        "created_at",
        "updated_at",
        "uuid",
        "session_id",
        "trace_id",
        "metadata",
        "payload",
    ]
    structured_label_hits = len(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_ -]{2,30}:\s+\S+", chunk))
    random_id_hits = len(re.findall(r"\b[a-fA-F0-9]{16,}\b|\b[A-Za-z0-9]{24,}\b", chunk))
    csv_like_lines = sum(1 for line in lines if line.count(",") >= 8)

    return {
        "repeated_lines": repeated_line_ratio(lines),
        "log_patterns": min(1.0, sum(lower.count(pattern) for pattern in log_patterns) / 4),
        "blob_like": 1.0
        if len(chunk) > 500
        and (sum(char.isalnum() or char in "+/=" for char in chunk) / len(chunk)) > 0.9
        and whitespace_ratio < 0.08
        and entropy(chunk) > 4.2
        else 0.0,
        "long_lines": min(1.0, max(0.0, (avg_line_len - 180) / 300)),
        "minified_shape": 1.0 if len(chunk) > 800 and newline_ratio < 0.01 and syntax_noise > 35 else 0.0,
        "low_alpha": 1.0 if alpha_ratio < 0.18 and len(chunk) > 300 else 0.0,
        "numeric_csv": min(1.0, csv_like_lines / 10) if digit_ratio > 0.25 else 0.0,
        "random_ids": min(1.0, random_id_hits / 8),
        "metadata_dump": min(1.0, sum(lower.count(term) for term in metadata_terms) / 5),
        "prose": 1.0 if len(words) >= 30 and sentence_marks >= 2 and alpha_ratio > 0.45 else 0.0,
        "document_ext": 1.0 if path.suffix.lower() in {".md", ".txt", ".rst"} and alpha_ratio > 0.35 else 0.0,
        "code_shape": 1.0 if path.suffix.lower() in TEXT_EXTENSIONS and any(term in lower for term in code_terms) else 0.0,
        "code_or_vocab_list": 1.0
        if path.suffix.lower() in {".py", ".js", ".ts", ".json", ".yaml", ".yml", ".txt"}
        and alpha_ratio > 0.45
        and digit_ratio < 0.12
        and not any(term in lower for term in [" traceback ", " exception ", " status=500 ", " request_id="])
        else 0.0,
        "structured_labels": min(1.0, structured_label_hits / 8),
    }


def classify_heuristic(chunk: str, path: Path, config: dict | None = None) -> dict:
    config = config or load_config()
    features = extract_features(chunk, path)
    weights = config["weights"]
    score = sum(features[name] * weights.get(name, 0.0) for name in features)
    threshold = float(config.get("threshold", 0.35))
    is_dump = score >= threshold
    label = "NOT_USEFUL" if is_dump else "USEFUL"
    strongest = sorted(
        ((abs(features[name] * weights.get(name, 0.0)), name) for name in features),
        reverse=True,
    )[:3]
    reasons = [name for value, name in strongest if value > 0]
    return {
        "label": label,
        "is_dump": is_dump,
        "score": round(score, 4),
        "threshold": threshold,
        "reason": ", ".join(reasons) or "weak signals",
        "features": features,
    }


def scan_folder(folder: Path):
    folder = folder.resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}\n")
        return

    config = load_config()
    report_path = Path.cwd() / f"rag_heuristic_scan_{int(time.time())}.jsonl"
    total_chunks = 0
    not_useful_chunks = 0
    scanned_files = 0
    skipped_files = 0
    started = time.perf_counter()

    print("Mode: HEURISTIC_ONLY")
    print(f"Scanning: {folder}")
    print(f"Report: {report_path}\n")

    with report_path.open("w", encoding="utf-8") as report:
        for file_path in iter_text_files(folder):
            text = read_text_file(file_path)
            if text is None:
                skipped_files += 1
                continue

            file_had_chunks = False
            for chunk_index, (start, end, chunk) in enumerate(chunk_text(text), start=1):
                file_had_chunks = True
                total_chunks += 1
                chunk_started = time.perf_counter()
                result = classify_heuristic(chunk, file_path, config)
                elapsed = time.perf_counter() - chunk_started

                if result["is_dump"]:
                    not_useful_chunks += 1

                row = {
                    "file": str(file_path),
                    "chunk_index": chunk_index,
                    "char_start": start,
                    "char_end": end,
                    "answer": result["label"],
                    "is_not_useful": result["is_dump"],
                    "score": result["score"],
                    "threshold": result["threshold"],
                    "reason": result["reason"],
                    "features": result["features"],
                    "seconds": round(elapsed, 4),
                }
                report.write(json.dumps(row, ensure_ascii=False) + "\n")
                report.flush()
                print(f"[{total_chunks}] {result['label']} | {file_path} | chunk {chunk_index}")

            if file_had_chunks:
                scanned_files += 1

    total_elapsed = time.perf_counter() - started
    print(
        f"\nScan complete: {scanned_files} files, {skipped_files} skipped, "
        f"{total_chunks} chunks, {not_useful_chunks} NOT_USEFUL chunks."
    )
    print(f"Total time: {total_elapsed:.2f} seconds")
    print(f"Saved report: {report_path}\n")


def train_now():
    import train_heuristic

    train_heuristic.main()


def run_client():
    print("Local RAG preprocessing CLI")
    print("Heuristic-only.")
    print("")
    print("h folder_path        -> scan files, divide chunks, print USEFUL/NOT_USEFUL")
    print("train                -> tune heuristic config on 1500+ labeled chunks")
    print("quit                 -> close this terminal and return")
    print("")
    print("Example:")
    print("h B:\\ai_model_test\\verify_scan\n")

    while True:
        try:
            command = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if command.lower() in {"quit", "exit", "q"}:
            break
        if command.lower() == "train":
            train_now()
            continue

        folder = parse_folder_command(command)
        if folder is not None:
            try:
                scan_folder(folder)
            except Exception as exc:
                print(f"Error: {exc}\n")
        else:
            print('Unknown command. Use: h "folder_path", train, or quit.\n')

    print("CLI closed.")


def launch_cli_terminal():
    script = Path(__file__).resolve()
    python = Path(sys.executable).resolve()

    if os.name == "nt":
        process = subprocess.Popen(
            [str(python), str(script), "--client"],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        process.wait()
        return

    terminal_commands = [
        ["x-terminal-emulator", "-e", str(python), str(script), "--client"],
        ["gnome-terminal", "--wait", "--", str(python), str(script), "--client"],
        ["konsole", "--nofork", "-e", str(python), str(script), "--client"],
        ["xterm", "-e", str(python), str(script), "--client"],
    ]
    for command in terminal_commands:
        try:
            subprocess.run(command, check=True)
            return
        except FileNotFoundError:
            continue

    run_client()


def parse_args():
    parser = argparse.ArgumentParser(description="Heuristic-only RAG preprocessing scanner.")
    parser.add_argument("--client", action="store_true", help="Run CLI in this terminal.")
    parser.add_argument("--train", action="store_true", help="Tune heuristic config and exit.")
    parser.add_argument("--heuristic", "--hscan", type=Path, help="Scan a folder and exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.train:
        train_now()
    elif args.heuristic:
        scan_folder(args.heuristic)
    elif args.client:
        run_client()
    else:
        launch_cli_terminal()


if __name__ == "__main__":
    main()
