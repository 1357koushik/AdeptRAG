import os
import sys
import json
import asyncio
import time
import tiktoken
import subprocess
import threading
import contextlib
import queue as queue_module
import hashlib
import textwrap

from dotenv import load_dotenv
from classifier import classify_text
from model.extractor import extract_entities
from db.graph_store import GraphStore
from db.vector_store import VectorStore
from engine.query import QueryEngine

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window, FloatContainer, Float
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.key_binding import KeyBindings

from prompt_toolkit.completion import Completer, Completion

load_dotenv()

# ──────────────────────────────────────────────
#  Autocomplete for commands and model names
# ──────────────────────────────────────────────
class AragCompleter(Completer):
    def __init__(self, available_models):
        self.commands = ['/mount', '/model', '/serve', '/query', '/dedup', '/quit', '/exit']
        self.models = available_models

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        parts = text.split(' ')
        if len(parts) == 1:
            for cmd in self.commands:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))
        elif len(parts) == 2 and parts[0] == '/model':
            word = parts[1]
            for m in self.models:
                if m.startswith(word):
                    yield Completion(m, start_position=-len(word))

# ──────────────────────────────────────────────
#  Thread-safe output redirector (queue-based)
# ──────────────────────────────────────────────
class OutputRedirector:
    """Captures print() from background threads into a thread-safe queue."""
    def __init__(self, q):
        self._q = q

    def write(self, s):
        if s:
            self._q.put(s)

    def flush(self):
        pass

# Sentinel to signal clean exit from background thread
_EXIT = object()

# ──────────────────────────────────────────────
#  Main application
# ──────────────────────────────────────────────
def main():
    if os.name == 'nt':
        os.system("")  # Enable ANSI on Windows

    state = {
        "current_model": "gemini-3.1-flash-lite",
        "graph_store": GraphStore(),
        "vector_store": VectorStore(),
        "is_busy": False,
        "server_process": None,
    }

    available_models = [
        "gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20240620",
        "gemini-2.5-flash", "gemini-3.1-flash-lite", "gemini-2.0-flash-lite",
        "ollama/llama3", "ollama/mistral-nemo",
    ]

    welcome_text = (
        "==================================================\n"
        "            AdeptRAG Interactive CLI              \n"
        "==================================================\n"
        "Available commands:\n"
        "  /mount <folder> - Ingest documents from a folder\n"
        "  /model <name>   - Change the extraction LLM model\n"
        "  /query <text>   - Ask a question using the Dual-Search Query Engine\n"
        "  /dedup          - Run Native Deduplication on the graph\n"
        "  /serve          - Start the 3D Graph Web UI\n"
        "  /quit           - Exit the CLI\n\n"
    )

    # ── Output buffer (writable, not focused by user) ──
    output_buffer = Buffer(name="output", read_only=False)
    output_buffer.set_document(Document(welcome_text, cursor_position=len(welcome_text)), bypass_readonly=True)

    output_window = Window(
        content=BufferControl(
            buffer=output_buffer,
            focusable=False,
            preview_search=False,
        ),
        wrap_lines=False,  # We pre-wrap the text to avoid scroll bugs with very long lines
        right_margins=[ScrollbarMargin(display_arrows=True)],
        allow_scroll_beyond_bottom=True,
    )

    # ── Input area ──
    input_area = TextArea(
        height=1,
        prompt="arag> ",
        multiline=False,
        completer=AragCompleter(available_models),
        complete_while_typing=True,
    )

    # ── Layout ──
    root_container = FloatContainer(
        content=HSplit([
            output_window,
            Window(height=1, char='-', style='class:separator'),
            input_area,
        ]),
        floats=[
            Float(
                xcursor=True,
                ycursor=True,
                content=CompletionsMenu(max_height=16, scroll_offset=1),
            )
        ],
    )

    # ── Keybindings ──
    kb = KeyBindings()

    @kb.add("c-c")
    @kb.add("c-q")
    @kb.add("c-d")
    def exit_(event):
        if state["server_process"]:
            try:
                state["server_process"].terminate()
            except Exception:
                pass
        event.app.exit()

    # Scroll keybindings on the output window
    @kb.add("pageup")
    def page_up(event):
        output_window.vertical_scroll = max(0, output_window.vertical_scroll - 10)

    @kb.add("pagedown")
    def page_down(event):
        output_window.vertical_scroll += 10

    @kb.add("up")
    def scroll_up(event):
        b = event.app.current_buffer
        if b.complete_state:
            b.complete_previous()
        else:
            output_window.vertical_scroll = max(0, output_window.vertical_scroll - 1)

    @kb.add("down")
    def scroll_down(event):
        b = event.app.current_buffer
        if b.complete_state:
            b.complete_next()
        else:
            output_window.vertical_scroll += 1

    # ── Thread-safe queue ──
    output_queue = queue_module.Queue()

    # ── append_to_output: MAIN THREAD ONLY ──
    def append_to_output(text: str):
        """Append text and auto-scroll to bottom."""
        # Pre-wrap text to avoid prompt_toolkit scroll bugs with extremely long logical lines
        wrapped_lines = []
        for line in text.split('\n'):
            if len(line) > 120:
                wrapped_lines.extend(textwrap.wrap(line, width=120))
            else:
                wrapped_lines.append(line)
        wrapped_text = '\n'.join(wrapped_lines)
        
        current = output_buffer.text
        new_text = current + wrapped_text
        output_buffer.set_document(Document(new_text, cursor_position=len(new_text)), bypass_readonly=True)
        # Auto-scroll to the bottom: set a very large number, prompt_toolkit clamps it
        output_window.vertical_scroll = 10_000_000

    # ── drain_queue: MAIN THREAD ONLY ──
    def drain_queue():
        chunks = []
        should_exit = False
        while True:
            try:
                item = output_queue.get_nowait()
                if item is _EXIT:
                    should_exit = True
                else:
                    chunks.append(item)
            except queue_module.Empty:
                break
        if chunks:
            append_to_output("".join(chunks))
            app.invalidate()
        if should_exit:
            if state["server_process"]:
                try:
                    state["server_process"].terminate()
                except Exception:
                    pass
            app.exit()

    # ── poll_output: async task on main event loop ──
    async def poll_output():
        while True:
            try:
                drain_queue()
            except Exception as e:
                with open("arag_debug.log", "a") as df:
                    import traceback
                    df.write(f"POLL ERROR: {e}\n{traceback.format_exc()}\n")
            await asyncio.sleep(0.05)

    # ── execute_command: runs in a background thread ──
    def execute_command(user_input):
        redirector = OutputRedirector(output_queue)
        try:
            with contextlib.redirect_stdout(redirector), contextlib.redirect_stderr(redirector):
                command = user_input.lower().strip()

                if command in ['/quit', '/exit']:
                    if state["server_process"]:
                        try:
                            state["server_process"].terminate()
                        except Exception:
                            pass
                    output_queue.put(_EXIT)
                    return

                elif command.startswith('/mount '):
                    folder = user_input[7:].strip()
                    if folder:
                        print(f"\n[Mounting folder: {folder}...]")
                        if os.path.exists(folder) and os.path.isdir(folder):
                            enc = tiktoken.get_encoding("cl100k_base")
                            chunk_size = 1200
                            chunk_overlap = 100
                            allowed_exts = {'.txt', '.md', '.json', '.csv', '.py', '.js', '.html'}
                            for root_dir, _, files in os.walk(folder):
                                for file in files:
                                    if not any(file.endswith(ext) for ext in allowed_exts):
                                        continue
                                    filepath = os.path.join(root_dir, file)
                                    try:
                                        with open(filepath, 'r', encoding='utf-8') as f:
                                            content = f.read()
                                        tokens = enc.encode(content)
                                        if not tokens:
                                            continue
                                        for i in range(0, len(tokens), chunk_size - chunk_overlap):
                                            chunk_tokens = tokens[i:i + chunk_size]
                                            chunk_text = enc.decode(chunk_tokens)
                                            chunk_idx = i // (chunk_size - chunk_overlap) + 1
                                            chunk_id = hashlib.md5(
                                                f"{file}_{chunk_idx}_{chunk_text[:50]}".encode('utf-8')
                                            ).hexdigest()
                                            state["vector_store"].upsert_chunk(
                                                chunk_id=chunk_id,
                                                text=chunk_text,
                                                metadata={"file": file, "chunk_index": chunk_idx}
                                            )
                                            result_json = classify_text(chunk_text)
                                            result = json.loads(result_json)
                                            label = result.get('label', 'Uncertain')
                                            if label == 'Data Dump':
                                                print(f"- {file} (Chunk {chunk_idx}): Saved to VectorDB | Dump Data (Skipping LLM)")
                                            else:
                                                print(f"- {file} (Chunk {chunk_idx}): Saved to VectorDB | Useful -> Extracting with {state['current_model']}...")
                                                extracted = extract_entities(chunk_text, state['current_model'])
                                                entities_count = sum(1 for e in extracted if e['type'] == 'entity')
                                                relations_count = sum(1 for e in extracted if e['type'] == 'relation')
                                                print(f"  -> Extracted {entities_count} entities, {relations_count} relations.")
                                                for item in extracted:
                                                    if item['type'] == 'entity':
                                                        state["graph_store"].upsert_entity(
                                                            item['name'], item.get('entity_type', ''), item.get('description', ''))
                                                        print(f"     [Entity] {item['name']} ({item.get('entity_type', '')})")
                                                    elif item['type'] == 'relation':
                                                        state["graph_store"].upsert_relation(
                                                            item['source'], item['target'], item.get('keywords', ''), item.get('description', ''))
                                                        print(f"     [Relation] {item['source']} -> {item['target']} ({item.get('keywords', '')})")
                                    except Exception as e:
                                        print(f"  Skipping {file}: {e}")
                            state["graph_store"].save_to_disk()
                        else:
                            print(f"Error: Directory '{folder}' not found.")
                        print("\n[Mount complete]\n")
                    else:
                        print("Usage: /mount <folder_path>\n")

                elif command == '/mount':
                    print("Usage: /mount <folder_path>\n")

                elif command.startswith('/query '):
                    query_text = user_input[7:].strip()
                    if query_text:
                        engine = QueryEngine(state["vector_store"], state["graph_store"], state["current_model"])
                        print(f"\n[Querying with {state['current_model']}]")
                        try:
                            answer, debug_prompt = engine.query(query_text)
                        except Exception as qe:
                            # Write raw error to a debug file so we can see it outside TUI
                            with open("arag_debug.log", "a") as df:
                                import traceback
                                df.write(f"QUERY ERROR: {qe}\n{traceback.format_exc()}\n")
                            print(f"\n[Query failed: {qe}]\n")
                            answer, debug_prompt = "", ""
                        # Also log the answer length to debug file
                        with open("arag_debug.log", "a") as df:
                            df.write(f"ANSWER LEN={len(answer)} ANSWER_REPR={repr(answer[:100])}\n")
                        # Print debug prompt FIRST because it's huge
                        print("\n" + "-" * 50)
                        print("DEBUG - FULL PROMPT SENT TO LLM:")
                        print("-" * 50)
                        print(debug_prompt)
                        print("-" * 50 + "\n")

                        # Print answer LAST so it's at the bottom of the screen
                        print("\n" + "=" * 50)
                        print("ANSWER:")
                        print("=" * 50)
                        if answer:
                            print(answer)
                        else:
                            print("[No answer returned from LLM]")
                        print("=" * 50 + "\n")
                    else:
                        print("Usage: /query <your question>\n")

                elif command == '/query':
                    print("Usage: /query <your question>\n")

                elif command.startswith('/model'):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        state["current_model"] = parts[1].strip()
                        print(f"\n[Model updated to: {state['current_model']}]\n")
                    else:
                        print(f"\nCurrent Model: {state['current_model']}")
                        print("Best models for this job:")
                        for m in available_models:
                            print(f"  - {m}")
                        print("Usage: /model <model_name>\n")

                elif command == '/serve':
                    if state["server_process"] is None or state["server_process"].poll() is not None:
                        print("\n[Starting Web Server on http://localhost:8000]")
                        print("Note: The server will run in the background. It shuts down when you exit.\n")
                        try:
                            state["server_process"] = subprocess.Popen(
                                [sys.executable, "-m", "uvicorn", "web.server:app",
                                 "--host", "0.0.0.0", "--port", "8000"])
                        except Exception as e:
                            print(f"Failed to start server: {e}")
                    else:
                        print("\n[!] Web Server is already running on http://localhost:8000\n")
                elif command == '/dedup':
                    from engine.dedup import run_deduplication
                    print("\n[Running Native Deduplication (Fuzzy Match >= 95%)...]")
                    try:
                        merge_count = run_deduplication(state["graph_store"], threshold=95)
                        print(f"\n[Deduplication Complete] Merged {merge_count} duplicate entities.\n")
                    except Exception as e:
                        print(f"\n[Deduplication Failed]: {e}\n")

                else:
                    print(f"Unknown command. Try '/mount <folder>', '/query <text>', '/model <name>', '/dedup', '/serve', or '/quit'.\n")

        except Exception as e:
            output_queue.put(f"\n[CRITICAL ERROR]: {e}\n")
        finally:
            state["is_busy"] = False

    # ── accept_text: main thread, called on Enter ──
    def accept_text(buff):
        user_input = input_area.text.strip()
        if not user_input:
            return False

        if state["is_busy"]:
            append_to_output("\n[!] Please wait, a command is currently running.\n")
            app.invalidate()
            return False

        input_area.text = ""
        append_to_output(f"arag> {user_input}\n")
        app.invalidate()

        state["is_busy"] = True
        t = threading.Thread(target=execute_command, args=(user_input,), daemon=True)
        t.start()
        return False

    input_area.accept_handler = accept_text

    # ── Application ──
    app = Application(
        layout=Layout(root_container, focused_element=input_area),
        key_bindings=kb,
        full_screen=True,
        mouse_support=True,
    )

    def pre_run():
        app.create_background_task(poll_output())

    try:
        app.run(pre_run=pre_run)
    finally:
        if state["server_process"]:
            try:
                state["server_process"].terminate()
            except Exception:
                pass

if __name__ == "__main__":
    main()