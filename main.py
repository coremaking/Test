from __future__ import annotations
import math, operator, os, platform, random, re
import signal, socket, sqlite3, sys, textwrap, time
import ast, atexit, calendar, json, logging, hashlib, hmac, secrets
import datetime as dt
import threading, queue, subprocess, shlex, webbrowser
import urllib.request, urllib.parse, urllib.error
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    C = {
        "user": Fore.CYAN, "bot": Fore.GREEN, "sys": Fore.YELLOW,
        "err": Fore.RED, "warn": Fore.MAGENTA, "info": Fore.BLUE,
        "dim": Style.DIM, "bold": Style.BRIGHT, "reset": Style.RESET_ALL,
    }
except ImportError:
    class _Dummy:
        def __getattr__(self, _): return ""
    Fore = Style = _Dummy()
    C = {k: "" for k in ("user", "bot", "sys", "err", "warn", "info",
                          "dim", "bold", "reset")}
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.markdown import Markdown
    _RICH = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    _RICH = None

DEFAULT_CONFIG = {
    "basic-config": {
        "name": "User",
        "name_of_bot": "Robo",
        "similarity_threshold": "0.80",
        "history_limit": "20",
        "log_file": "chatbot.log",
        "db_file": "memory.db",
        "theme": "default",
        "auto_learn": "true",
        "sentiment_enabled": "true",
        "plugins_enabled": "true",
    },
    "personality": {
        "formality": "casual",
        "humor_level": "medium", 
        "verbosity": "normal",
    },
    "features": {
        "weather": "true",
        "wikipedia": "true",
        "translate": "false",
        "reminders": "true",
        "plugins": "true",
    },
}
CONFIG_FILE = Path("config.json")
LEGACY_CONFIG_FILE = Path("config.txt")

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open("r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            for section, values in DEFAULT_CONFIG.items():
                cfg.setdefault(section, {})
                for k, v in values.items():
                    cfg[section].setdefault(k, v)
            logging.info("Loaded config from %s", CONFIG_FILE)
            return cfg
        except (json.JSONDecodeError, OSError) as exc:
            logging.warning("Could not parse %s: %s. Using defaults.", CONFIG_FILE, exc)
    if LEGACY_CONFIG_FILE.exists():
        try:
            import configparser
            parser = configparser.RawConfigParser()
            parser.read(LEGACY_CONFIG_FILE)
            cfg = {"basic-config": dict(parser.items("basic-config"))}
            with CONFIG_FILE.open("w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
            logging.info("Migrated legacy config -> %s", CONFIG_FILE)
            return cfg
        except Exception as exc:
            logging.warning("Legacy migration failed: %s", exc)
    with CONFIG_FILE.open("w", encoding="utf-8") as fh:
        json.dump(DEFAULT_CONFIG, fh, indent=2)
    logging.info("Wrote default config to %s", CONFIG_FILE)
    return DEFAULT_CONFIG

def save_config(cfg: dict) -> None:
    with CONFIG_FILE.open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

def setup_logging(log_file: str = "chatbot.log") -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow, ast.BitXor: operator.xor,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_NAMES = {
    "pi": math.pi, "e": math.e, "tau": math.tau,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "log2": math.log2,
    "abs": abs, "round": round, "min": min, "max": max, "pow": pow,
    "floor": math.floor, "ceil": math.ceil, "factorial": math.factorial,
    "degrees": math.degrees, "radians": math.radians, "hypot": math.hypot,
}

def _safe_eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](
            _safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _ALLOWED_NAMES:
        return _ALLOWED_NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in _ALLOWED_NAMES:
            raise ValueError(f"Function '{node.func.id}' not allowed")
        return _ALLOWED_NAMES[node.func.id](
            *[_safe_eval_node(a) for a in node.args])
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")

def safe_math(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    return _safe_eval_node(tree)

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

@dataclass
class Turn:
    user: str
    bot: str
    ts: dt.datetime = field(default_factory=dt.datetime.now)

@dataclass
class Reminder:
    id: int
    message: str
    due: dt.datetime
    done: bool = False

class MemoryStore:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS response_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_voice TEXT NOT NULL,
        response   TEXT NOT NULL,
        hits       INTEGER DEFAULT 0,
        rating     INTEGER DEFAULT 0,
        tags       TEXT DEFAULT '',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_user_voice ON response_memory(user_voice);

    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_voice TEXT,
        bot_reply  TEXT,
        sentiment  TEXT DEFAULT 'neutral',
        ts         TEXT
    );

    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        body TEXT NOT NULL,
        tags TEXT DEFAULT '',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT NOT NULL,
        due     TEXT NOT NULL,
        done    INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS kv (
        key   TEXT PRIMARY KEY,
        value TEXT
    );
    """

    def __init__(self, db_path: str = "memory.db") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._cur = self._conn.cursor()
        self._cur.executescript(self.SCHEMA)
        self._conn.commit()

    def find(self, user_voice: str) -> Optional[sqlite3.Row]:
        self._cur.execute(
            "SELECT * FROM response_memory WHERE LOWER(user_voice)=LOWER(?)",
            (user_voice.strip(),))
        return self._cur.fetchone()

    def find_fuzzy(self, user_voice: str,
                   threshold: float = 0.8) -> Optional[sqlite3.Row]:
        self._cur.execute("SELECT * FROM response_memory")
        rows = self._cur.fetchall()
        best, best_score = None, 0.0
        for row in rows:
            score = similarity(user_voice, row["user_voice"])
            if score > best_score:
                best, best_score = row, score
        if best and best_score >= threshold:
            logging.debug("Fuzzy match %.2f: %r -> %r",
                          best_score, user_voice, best["user_voice"])
            return best
        return None

    def learn(self, user_voice: str, response: str,
              tags: str = "") -> int:
        existing = self.find(user_voice)
        if existing:
            self.update_response(existing["id"], response)
            return existing["id"]
        self._cur.execute(
            "INSERT INTO response_memory(user_voice, response, tags) VALUES(?,?,?)",
            (user_voice.strip(), response.strip(), tags))
        self._conn.commit()
        return self._cur.lastrowid

    def bump_hit(self, row_id: int) -> None:
        self._cur.execute(
            "UPDATE response_memory SET hits=hits+1, "
            "updated_at=CURRENT_TIMESTAMP WHERE id=?", (row_id,))
        self._conn.commit()

    def update_response(self, row_id: int, new_response: str) -> None:
        self._cur.execute(
            "UPDATE response_memory SET response=?, "
            "updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (new_response, row_id))
        self._conn.commit()

    def rate_response(self, row_id: int, delta: int) -> None:
        self._cur.execute(
            "UPDATE response_memory SET rating=rating+? WHERE id=?",
            (delta, row_id))
        self._conn.commit()

    def forget(self, user_voice: str) -> int:
        self._cur.execute(
            "DELETE FROM response_memory WHERE LOWER(user_voice)=LOWER(?)",
            (user_voice.strip(),))
        self._conn.commit()
        return self._cur.rowcount

    def forget_id(self, row_id: int) -> int:
        self._cur.execute("DELETE FROM response_memory WHERE id=?", (row_id,))
        self._conn.commit()
        return self._cur.rowcount

    def list_all(self, query: str = "") -> list[sqlite3.Row]:
        if query:
            like = f"%{query}%"
            self._cur.execute(
                "SELECT * FROM response_memory WHERE user_voice LIKE ? "
                "OR response LIKE ? OR tags LIKE ? "
                "ORDER BY hits DESC, user_voice", (like, like, like))
        else:
            self._cur.execute(
                "SELECT * FROM response_memory ORDER BY hits DESC, user_voice")
        return self._cur.fetchall()

    def export_json(self, path: str) -> None:
        rows = [{"user_voice": r["user_voice"], "response": r["response"],
                 "hits": r["hits"], "rating": r["rating"], "tags": r["tags"]}
                for r in self.list_all()]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, ensure_ascii=False)

    def import_json(self, path: str) -> int:
        with open(path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)
        count = 0
        for r in rows:
            try:
                self.learn(r["user_voice"], r["response"],
                           r.get("tags", ""))
                count += 1
            except KeyError:
                continue
        return count

    def log_turn(self, user_voice: str, bot_reply: str,
                 sentiment: str = "neutral") -> None:
        self._cur.execute(
            "INSERT INTO history(user_voice, bot_reply, sentiment, ts) "
            "VALUES(?,?,?,?)",
            (user_voice, bot_reply, sentiment, dt.datetime.now().isoformat()))
        self._conn.commit()

    def recent_turns(self, limit: int = 20) -> list[sqlite3.Row]:
        self._cur.execute(
            "SELECT user_voice, bot_reply, sentiment, ts FROM history "
            "ORDER BY id DESC LIMIT ?", (limit,))
        return list(reversed(self._cur.fetchall()))

    def search_history(self, term: str) -> list[sqlite3.Row]:
        like = f"%{term}%"
        self._cur.execute(
            "SELECT user_voice, bot_reply, sentiment, ts FROM history "
            "WHERE user_voice LIKE ? OR bot_reply LIKE ? ORDER BY id DESC",
            (like, like))
        return self._cur.fetchall()

    def clear_history(self) -> int:
        self._cur.execute("DELETE FROM history")
        self._conn.commit()
        return self._cur.rowcount

    def add_note(self, body: str, tags: str = "") -> int:
        self._cur.execute("INSERT INTO notes(body, tags) VALUES(?,?)",
                          (body, tags))
        self._conn.commit()
        return self._cur.lastrowid

    def list_notes(self) -> list[sqlite3.Row]:
        self._cur.execute("SELECT * FROM notes ORDER BY id DESC")
        return self._cur.fetchall()

    def delete_note(self, note_id: int) -> int:
        self._cur.execute("DELETE FROM notes WHERE id=?", (note_id,))
        self._conn.commit()
        return self._cur.rowcount

    def add_reminder(self, message: str, due: dt.datetime) -> int:
        self._cur.execute(
            "INSERT INTO reminders(message, due) VALUES(?,?)",
            (message, due.isoformat()))
        self._conn.commit()
        return self._cur.lastrowid

    def due_reminders(self) -> list[sqlite3.Row]:
        now = dt.datetime.now().isoformat()
        self._cur.execute(
            "SELECT * FROM reminders WHERE done=0 AND due<=? ORDER BY due",
            (now,))
        return self._cur.fetchall()

    def list_reminders(self, include_done: bool = False) -> list[sqlite3.Row]:
        if include_done:
            self._cur.execute("SELECT * FROM reminders ORDER BY due")
        else:
            self._cur.execute(
                "SELECT * FROM reminders WHERE done=0 ORDER BY due")
        return self._cur.fetchall()

    def complete_reminder(self, rid: int) -> int:
        self._cur.execute("UPDATE reminders SET done=1 WHERE id=?", (rid,))
        self._conn.commit()
        return self._cur.rowcount

    def delete_reminder(self, rid: int) -> int:
        self._cur.execute("DELETE FROM reminders WHERE id=?", (rid,))
        self._conn.commit()
        return self._cur.rowcount

    def kv_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        self._cur.execute("SELECT value FROM kv WHERE key=?", (key,))
        row = self._cur.fetchone()
        return row["value"] if row else default

    def kv_set(self, key: str, value: str) -> None:
        self._cur.execute(
            "INSERT INTO kv(key, value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value))
        self._conn.commit()

    def stats(self) -> dict:
        self._cur.execute("SELECT COUNT(*) c, SUM(hits) h, "
                          "AVG(rating) r FROM response_memory")
        row = self._cur.fetchone()
        self._cur.execute("SELECT COUNT(*) c FROM history")
        hist = self._cur.fetchone()["c"]
        self._cur.execute("SELECT COUNT(*) c FROM notes")
        notes = self._cur.fetchone()["c"]
        self._cur.execute("SELECT COUNT(*) c FROM reminders WHERE done=0")
        reminders = self._cur.fetchone()["c"]
        return {
            "responses": row["c"] or 0,
            "total_hits": row["h"] or 0,
            "avg_rating": round(row["r"] or 0.0, 2),
            "history": hist,
            "notes": notes,
            "reminders": reminders,
        }

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

class PluginManager:
    def __init__(self, plugin_dir: str = "plugins") -> None:
        self.plugin_dir = Path(plugin_dir)
        self.loaded: list[str] = []

    def load_all(self, registry: "CommandRegistry") -> None:
        if not self.plugin_dir.exists():
            return
        import importlib.util
        for py in sorted(self.plugin_dir.glob("*.py")):
            try:
                spec = importlib.util.spec_from_file_location(
                    f"plugin_{py.stem}", py)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, "register"):
                    mod.register(registry)
                    self.loaded.append(py.stem)
                    logging.info("Loaded plugin: %s", py.stem)
            except Exception as exc:
                logging.warning("Plugin %s failed: %s", py.name, exc)

class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, tuple[Callable, str, list[str]]] = {}

    def register(self, name: str, help_text: str,
                 aliases: Optional[list[str]] = None, category: str = "general"):
        def deco(fn: Callable):
            keys = [name] + (aliases or [])
            for k in keys:
                self._commands[k.lower()] = (fn, help_text, keys)
            fn._category = category  # type: ignore
            return fn
        return deco

    def get(self, name: str):
        return self._commands.get(name.lower())

    def all(self) -> dict:
        return self._commands

    def unique_help(self) -> list[tuple[str, str, str]]:
        seen: dict[str, tuple[str, str]] = {}
        for cmd, (fn, help_text, keys) in self._commands.items():
            if cmd == keys[0]:
                cat = getattr(fn, "_category", "general")
                seen[cmd] = (help_text, cat)
        return sorted([(c, h, cat) for c, (h, cat) in seen.items()])

TEACH_ME_PROMPTS = [
    "Teach me how to answer this!",
    "Please teach me how to respond to this, {name}!",
    "How do I respond to this, {name}? Train me!",
    "I have no clue how to respond to this! Teach me!",
    "What should I say to '{voice}'? Teach me!",
]

JOKES = [
    "Why did the programmer quit? Because they didn't get arrays.",
    "I'd tell you a UDP joke, but you might not get it.",
    "There are 10 kinds of people: those who understand binary and those who don't.",
    "Why do Java developers wear glasses? Because they don't C#.",
    "A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?'",
    "Why do Python programmers prefer snakes? Because they hiss and tell no lies.",
    "I would tell you a joke about recursion, but I'd have to tell you a joke about recursion first.",
]

FACTS = [
    "The first computer bug was an actual moth found in a Harvard Mark II in 1947.",
    "Python was named after Monty Python, not the snake.",
    "The term 'debugging' predates computers; it was used in aviation.",
    "The first 1 GB hard drive weighed about 500 pounds and cost $40,000.",
    "Email predates the World Wide Web by almost two decades.",
    "The original name for Windows was 'Interface Manager'.",
    "The first computer mouse was made of wood.",
]

QUOTES = [
    ("Simplicity is the ultimate sophistication.", "Leonardo da Vinci"),
    ("Talk is cheap. Show me the code.", "Linus Torvalds"),
    ("Programs must be written for people to read.", "Harold Abelson"),
    ("Premature optimization is the root of all evil.", "Donald Knuth"),
    ("The best way to predict the future is to invent it.", "Alan Kay"),
]

GREETINGS = ["hi", "hello", "hey", "yo", "hiya", "sup", "howdy",
             "good morning", "good evening", "good afternoon"]
FAREWELLS = ["bye", "goodbye", "see you later", "bubye", "cya", "exit", "quit"]

POSITIVE_WORDS = {"good", "great", "awesome", "love", "happy", "thanks",
                  "thank", "cool", "nice", "excellent", "amazing", "wonderful",
                  "fantastic", "perfect", "brilliant", "superb"}
NEGATIVE_WORDS = {"bad", "sad", "hate", "angry", "terrible", "awful",
                  "sorry", "upset", "annoyed", "frustrated", "depressed",
                  "miserable", "horrible", "disappointed", "furious"}

SENTIMENT_EMOJI = {"positive": "😊", "negative": "😟", "neutral": "😐"}

class Robot:
    def __init__(self, config: dict, memory: MemoryStore) -> None:
        basic = config.get("basic-config", {})
        self.config = config
        self.name = basic.get("name", "User")
        self.name_of_bot = basic.get("name_of_bot", "Robo")
        self.similarity_threshold = float(basic.get("similarity_threshold", 0.80))
        self.history_limit = int(basic.get("history_limit", 20))
        self.auto_learn = basic.get("auto_learn", "true").lower() == "true"
        self.sentiment_enabled = basic.get("sentiment_enabled", "true").lower() == "true"
        self.plugins_enabled = basic.get("plugins_enabled", "true").lower() == "true"
        self.personality = config.get("personality", {})
        self.features = config.get("features", {})
        self.memory = memory
        self.registry = CommandRegistry()
        self.running = True
        self._pending_teach: Optional[str] = None
        self._pending_edit: Optional[int] = None
        self._last_response: Optional[str] = None
        self._last_row_id: Optional[int] = None
        self._mood = "neutral"
        self._mood_counter = Counter()
        self._reminder_queue: queue.Queue[str] = queue.Queue()
        self._reminder_thread: Optional[threading.Thread] = None
        self._start_reminder_worker()
        self._register_commands()
        self._register_macro_commands()
        if self.plugins_enabled:
            self.plugins = PluginManager()
            self.plugins.load_all(self.registry)
        else:
            self.plugins = PluginManager()

    def cprint(self, text: str, kind: str = "bot",
               prefix: bool = True, use_rich: bool = False) -> None:
        if use_rich and HAS_RICH:
            _RICH.print(text)
            logging.info("%s", text)
            return
        color = C.get(kind, "")
        prefix_str = ""
        if prefix and kind == "bot":
            mood_emoji = SENTIMENT_EMOJI.get(self._mood, "")
            prefix_str = f"[{self.name_of_bot}{mood_emoji}] "
        print(f"{color}{prefix_str}{text}{C['reset']}")
        logging.info("%s%s", prefix_str, text)

    def prompt_user(self) -> str:
        return input(f"{C['user']}> {C['reset']}").strip()

    def confirm(self, prompt: str) -> bool:
        ans = input(f"{C['sys']}{prompt} [y/N]: {C['reset']}").strip().lower()
        return ans in ("y", "yes")

    def _table(self, title: str, rows: list[tuple], headers: list[str]) -> None:
        if HAS_RICH:
            t = Table(title=title)
            for h in headers:
                t.add_column(h)
            for r in rows:
                t.add_row(*[str(x) for x in r])
            _RICH.print(t)
        else:
            self.cprint(title, "info", prefix=False)
            self.cprint(" | ".join(headers), "bold", prefix=False)
            for r in rows:
                self.cprint(" | ".join(str(x) for x in r), prefix=False)

    def _start_reminder_worker(self) -> None:
        def worker():
            while self.running:
                try:
                    due = self.memory.due_reminders()
                    for r in due:
                        msg = f"⏰ REMINDER: {r['message']} (due {r['due'][:19]})"
                        self._reminder_queue.put(msg)
                        self.memory.complete_reminder(r["id"])
                except Exception:
                    logging.exception("Reminder worker error")
                time.sleep(5)
        self._reminder_thread = threading.Thread(
            target=worker, daemon=True, name="reminder-worker")
        self._reminder_thread.start()

    def _drain_reminders(self) -> None:
        while not self._reminder_queue.empty():
            try:
                msg = self._reminder_queue.get_nowait()
                self.cprint(f"\n{msg}", "warn", prefix=False)
            except queue.Empty:
                break

    def _register_commands(self) -> None:
        reg = self.registry.register
        @reg("help", "Show this help message", ["?"], "meta")
        def _help(arg: str):
            cat_filter = arg.strip().lower() if arg else ""
            grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for cmd, txt, cat in self.registry.unique_help():
                if cat_filter and cat != cat_filter:
                    continue
                grouped[cat].append((cmd, txt))
            lines = [f"{C['bold']}Available commands:{C['reset']}"]
            for cat in sorted(grouped):
                lines.append(f"\n{C['bold']}▸ {cat.upper()}{C['reset']}")
                for cmd, txt in grouped[cat]:
                    lines.append(f"  @{cmd:<18} {txt}")
            lines.append(f"\nUse @help <category> to filter.")
            self.cprint("\n".join(lines), "sys", prefix=False)

        @reg("exit", "Exit the chatbot", ["quit"], "meta")
        def _exit(_: str):
            self.cprint(random.choice(
                ["Bye!", "See you later!", "Take care!", "Goodbye, friend!"]))
            self.running = False

        @reg("clear", "Clear screen", [], "meta")
        def _clear(_: str):
            os.system("cls" if os.name == "nt" else "clear")

        @reg("version", "Show version", ["about"], "meta")
        def _version(_: str):
            self.cprint(f"{self.name_of_bot} v2.0 — feature-rich chatbot", "info")

        @reg("time", "Show current date & time", [], "info")
        def _time(_: str):
            self.cprint(dt.datetime.now().strftime("%A, %d %B %Y — %H:%M:%S"))

        @reg("calendar", "Show current month calendar", ["cal"], "info")
        def _calendar(_: str):
            now = dt.datetime.now()
            self.cprint("\n" + calendar.month(now.year, now.month), prefix=False)

        @reg("sysinfo", "Show computer info", ["computer info"], "info")
        def _sysinfo(_: str):
            info = (
                f"Host:   {socket.gethostname()}\n"
                f"OS:     {platform.platform()}\n"
                f"CPU:    {platform.processor() or 'unknown'}\n"
                f"Python: {platform.python_version()}\n"
                f"Arch:   {platform.machine()}\n"
                f"CWD:    {os.getcwd()}"
            )
            self.cprint(info, prefix=False)

        @reg("uptime", "Show process uptime", [], "info")
        def _uptime(_: str):
            started = self.memory.kv_get("process_started")
            if not started:
                self.cprint("Uptime unknown.")
                return
            delta = dt.datetime.now() - dt.datetime.fromisoformat(started)
            self.cprint(f"Uptime: {delta}")

        @reg("stats", "Show memory & usage statistics", [], "info")
        def _stats(_: str):
            s = self.memory.stats()
            rows = [(k.replace("_", " ").title(), str(v)) for k, v in s.items()]
            self._table("Statistics", rows, ["Metric", "Value"])

        @reg("teach", "Teach a new response: @teach <phrase>", [], "learning")
        def _teach(arg: str):
            if not arg:
                self.cprint("Usage: @teach <phrase>", "err")
                return
            self._pending_teach = arg
            self.cprint(f"What should I reply to '{arg}'?")

        @reg("forget", "Forget a learned phrase: @forget <phrase>", [], "learning")
        def _forget(arg: str):
            if not arg:
                self.cprint("Usage: @forget <phrase>", "err")
                return
            removed = self.memory.forget(arg)
            self.cprint(f"Forgot {removed} entr{'y' if removed == 1 else 'ies'}.")

        @reg("forgetid", "Forget by ID: @forgetid <id>", [], "learning")
        def _forgetid(arg: str):
            if not arg.isdigit():
                self.cprint("Usage: @forgetid <id>", "err")
                return
            removed = self.memory.forget_id(int(arg))
            self.cprint(f"Deleted {removed} entr{'y' if removed == 1 else 'ies'}.")

        @reg("edit", "Edit a learned response: @edit <id>", [], "learning")
        def _edit(arg: str):
            if not arg.isdigit():
                self.cprint("Usage: @edit <id>", "err")
                return
            self._pending_edit = int(arg)
            self.cprint("Enter the new response:")

        @reg("list", "List all learned responses (optional filter)",
             ["ls"], "learning")
        def _list(arg: str):
            rows = self.memory.list_all(arg)
            if not rows:
                self.cprint("Nothing learned yet.")
                return
            table_rows = [(r["id"], r["hits"], r["rating"],
                           r["user_voice"], r["response"]) for r in rows]
            self._table("Learned Responses", table_rows,
                        ["#", "Hits", "Rating", "Voice", "Response"])

        @reg("rate", "Rate last response: @rate up|down", [], "learning")
        def _rate(arg: str):
            if self._last_row_id is None:
                self.cprint("No response to rate yet.", "err")
                return
            if arg.lower() in ("up", "+", "good", "1"):
                self.memory.rate_response(self._last_row_id, 1)
                self.cprint("Thanks for the feedback! 👍")
            elif arg.lower() in ("down", "-", "bad", "-1"):
                self.memory.rate_response(self._last_row_id, -1)
                self.cprint("Sorry about that. 👎")
            else:
                self.cprint("Usage: @rate up|down", "err")

        @reg("history", "Show recent conversation history", ["hist"], "history")
        def _history(arg: str):
            limit = int(arg) if arg.isdigit() else self.history_limit
            turns = self.memory.recent_turns(limit)
            if not turns:
                self.cprint("No history yet.")
                return
            table_rows = [(t["ts"][:19], t["user_voice"], t["bot_reply"],
                           t["sentiment"]) for t in turns]
            self._table("History", table_rows,
                        ["Time", "You", "Bot", "Sentiment"])

        @reg("search", "Search history: @search <term>", [], "history")
        def _search(arg: str):
            if not arg:
                self.cprint("Usage: @search <term>", "err")
                return
            rows = self.memory.search_history(arg)
            if not rows:
                self.cprint("No matches.")
                return
            table_rows = [(t["ts"][:19], t["user_voice"], t["bot_reply"])
                          for t in rows]
            self._table(f"Search: {arg}", table_rows,
                        ["Time", "You", "Bot"])

        @reg("clearhistory", "Clear all history", [], "history")
        def _clearhistory(_: str):
            if not self.confirm("Delete ALL conversation history?"):
                self.cprint("Cancelled.")
                return
            n = self.memory.clear_history()
            self.cprint(f"Cleared {n} history entries.")

        @reg("note", "Add a note: @note <text>", [], "notes")
        def _note(arg: str):
            if not arg:
                self.cprint("Usage: @note <text>", "err")
                return
            nid = self.memory.add_note(arg)
            self.cprint(f"Note #{nid} saved.")

        @reg("notes", "List all notes", [], "notes")
        def _notes(_: str):
            rows = self.memory.list_notes()
            if not rows:
                self.cprint("No notes.")
                return
            table_rows = [(r["id"], r["created_at"][:19], r["body"])
                          for r in rows]
            self._table("Notes", table_rows, ["#", "Created", "Body"])

        @reg("delnote", "Delete a note: @delnote <id>", [], "notes")
        def _delnote(arg: str):
            if not arg.isdigit():
                self.cprint("Usage: @delnote <id>", "err")
                return
            removed = self.memory.delete_note(int(arg))
            self.cprint(f"Deleted {removed} note(s).")

        @reg("remind", "Set a reminder: @remind <minutes> <message>",
             [], "reminders")
        def _remind(arg: str):
            parts = arg.split(maxsplit=1)
            if len(parts) < 2 or not parts[0].isdigit():
                self.cprint("Usage: @remind <minutes> <message>", "err")
                return
            minutes = int(parts[0])
            msg = parts[1]
            due = dt.datetime.now() + dt.timedelta(minutes=minutes)
            rid = self.memory.add_reminder(msg, due)
            self.cprint(f"Reminder #{rid} set for {due.strftime('%H:%M')}.")

        @reg("reminders", "List pending reminders", [], "reminders")
        def _reminders(arg: str):
            include_done = arg.lower() == "all"
            rows = self.memory.list_reminders(include_done)
            if not rows:
                self.cprint("No reminders.")
                return
            table_rows = [(r["id"], r["due"][:19], r["message"],
                           "✓" if r["done"] else "•") for r in rows]
            self._table("Reminders", table_rows,
                        ["#", "Due", "Message", "Status"])

        @reg("donereminder", "Mark reminder done: @donereminder <id>",
             ["reminddone"], "reminders")
        def _donereminder(arg: str):
            if not arg.isdigit():
                self.cprint("Usage: @donereminder <id>", "err")
                return
            n = self.memory.complete_reminder(int(arg))
            self.cprint(f"Marked {n} reminder(s) done.")

        @reg("delreminder", "Delete a reminder: @delreminder <id>",
             [], "reminders")
        def _delreminder(arg: str):
            if not arg.isdigit():
                self.cprint("Usage: @delreminder <id>", "err")
                return
            n = self.memory.delete_reminder(int(arg))
            self.cprint(f"Deleted {n} reminder(s).")

        @reg("calc", "Evaluate math: @calc 2+3*4", ["math"], "utility")
        def _calc(arg: str):
            if not arg:
                self.cprint("Usage: @calc <expression>", "err")
                return
            try:
                result = safe_math(arg.replace("^", "**"))
                self.cprint(f"= {result}")
            except Exception as exc:
                self.cprint(f"Math error: {exc}", "err")

        @reg("joke", "Tell a joke", [], "fun")
        def _joke(_: str):
            self.cprint(random.choice(JOKES))

        @reg("fact", "Share a random fact", [], "fun")
        def _fact(_: str):
            self.cprint(random.choice(FACTS))

        @reg("quote", "Show an inspirational quote", [], "fun")
        def _quote(_: str):
            text, author = random.choice(QUOTES)
            self.cprint(f'"{text}"\n  — {author}')

        @reg("roll", "Roll dice: @roll [NdM]", [], "fun")
        def _roll(arg: str):
            if not arg:
                n, m = 1, 6
            else:
                m_ = re.fullmatch(r"(\d*)d(\d+)", arg.lower())
                if not m_:
                    self.cprint("Usage: @roll [NdM]  e.g. @roll 3d20", "err")
                    return
                n = int(m_.group(1) or 1)
                m = int(m_.group(2))
            if n > 100 or m > 10000:
                self.cprint("Too many dice / sides.", "err")
                return
            rolls = [random.randint(1, m) for _ in range(n)]
            self.cprint(f"{rolls}  (sum={sum(rolls)})")

        @reg("flip", "Flip a coin", [], "fun")
        def _flip(_: str):
            self.cprint(random.choice(["Heads!", "Tails!"]))

        @reg("8ball", "Ask the magic 8-ball: @8ball <question>",
             ["magic8ball"], "fun")
        def _8ball(_: str):
            answers = ["Yes", "No", "Maybe", "Ask again later",
                       "Definitely", "Absolutely not", "Signs point to yes",
                       "Cannot predict now", "Don't count on it"]
            self.cprint(random.choice(answers))

        @reg("password", "Generate a password: @password [length]",
             ["genpass"], "utility")
        def _password(arg: str):
            length = int(arg) if arg.isdigit() else 16
            length = max(4, min(length, 128))
            alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+"
            pwd = "".join(secrets.choice(alphabet) for _ in range(length))
            self.cprint(f"{pwd}")

        @reg("hash", "Hash text: @hash <algo> <text>  (md5/sha1/sha256)",
             [], "utility")
        def _hash(arg: str):
            parts = arg.split(maxsplit=1)
            if len(parts) < 2:
                self.cprint("Usage: @hash <algo> <text>", "err")
                return
            algo, text = parts[0].lower(), parts[1]
            if algo not in ("md5", "sha1", "sha256", "sha512"):
                self.cprint("Supported: md5, sha1, sha256, sha512", "err")
                return
            h = hashlib.new(algo, text.encode("utf-8")).hexdigest()
            self.cprint(f"{algo}({text!r}) = {h}")

        @reg("b64", "Base64 encode/decode: @b64 enc|dec <text>",
             [], "utility")
        def _b64(arg: str):
            import base64
            parts = arg.split(maxsplit=1)
            if len(parts) < 2 or parts[0] not in ("enc", "dec"):
                self.cprint("Usage: @b64 enc|dec <text>", "err")
                return
            try:
                if parts[0] == "enc":
                    out = base64.b64encode(parts[1].encode()).decode()
                else:
                    out = base64.b64decode(parts[1]).decode()
                self.cprint(out)
            except Exception as exc:
                self.cprint(f"Error: {exc}", "err")

        @reg("uuid", "Generate a UUID", [], "utility")
        def _uuid(_: str):
            import uuid
            self.cprint(str(uuid.uuid4()))

        @reg("wordcount", "Count words in text: @wordcount <text>",
             [], "utility")
        def _wordcount(arg: str):
            if not arg:
                self.cprint("Usage: @wordcount <text>", "err")
                return
            words = re.findall(r"\w+", arg)
            self.cprint(f"Words: {len(words)}  Characters: {len(arg)}")

        @reg("run", "Run a shell command (with confirmation): @run <cmd>",
             ["shell"], "system")
        def _run(arg: str):
            if not arg:
                self.cprint("Usage: @run <command>", "err")
                return
            if not self.confirm(f"Run `{arg}` ?"):
                self.cprint("Cancelled.")
                return
            try:
                out = subprocess.run(
                    arg, shell=True, capture_output=True, text=True,
                    timeout=30)
                if out.stdout:
                    self.cprint(out.stdout, prefix=False)
                if out.stderr:
                    self.cprint(out.stderr, "err", prefix=False)
                self.cprint(f"Exit code: {out.returncode}")
            except subprocess.TimeoutExpired:
                self.cprint("Command timed out.", "err")

        @reg("open", "Open URL in browser: @open <url>", ["browse"], "system")
        def _open(arg: str):
            if not arg:
                self.cprint("Usage: @open <url>", "err")
                return
            if not arg.startswith(("http://", "https://")):
                arg = "https://" + arg
            webbrowser.open(arg)
            self.cprint(f"Opened {arg}")

        @reg("pwd", "Print working directory", [], "system")
        def _pwd(_: str):
            self.cprint(os.getcwd())

        @reg("export", "Export memory to JSON: @export <file>",
             [], "io")
        def _export(arg: str):
            path = arg or "memory_export.json"
            self.memory.export_json(path)
            self.cprint(f"Exported to {path}")

        @reg("import", "Import memory from JSON: @import <file>",
             [], "io")
        def _import(arg: str):
            if not arg or not os.path.exists(arg):
                self.cprint("Usage: @import <existing-file.json>", "err")
                return
            n = self.memory.import_json(arg)
            self.cprint(f"Imported {n} entries.")

        @reg("config", "Show current config", [], "meta")
        def _config(arg: str):
            self.cprint(json.dumps(self.config, indent=2), prefix=False)

        @reg("setconfig", "Set a config value: @setconfig section.key value",
             [], "meta")
        def _setconfig(arg: str):
            parts = arg.split(maxsplit=2)
            if len(parts) < 3 or "." not in parts[0]:
                self.cprint("Usage: @setconfig <section>.<key> <value>", "err")
                return
            section, key = parts[0].split(".", 1)
            value = parts[1] if len(parts) == 2 else " ".join(parts[1:])
            self.config.setdefault(section, {})[key] = value
            save_config(self.config)
            self.cprint(f"Set {section}.{key} = {value}")

        @reg("mood", "Show current mood", [], "meta")
        def _mood(_: str):
            self.cprint(f"Current mood: {self._mood} "
                        f"{SENTIMENT_EMOJI[self._mood]}")

        @reg("macros", "List all macros", [], "meta")
        def _macros(_: str):
            macros = self._load_macros()
            if not macros:
                self.cprint("No macros defined.")
                return
            for name, val in macros.items():
                self.cprint(f"  @{name}  ->  {val}", prefix=False)

        @reg("macro", "Define a macro: @macro name @existing", [], "meta")
        def _macro(arg: str):
            parts = arg.split(maxsplit=1)
            if len(parts) < 2:
                self.cprint("Usage: @macro <name> <command text>", "err")
                return
            macros = self._load_macros()
            macros[parts[0].lower()] = parts[1]
            self.memory.kv_set("macros", json.dumps(macros))
            self.cprint(f"Macro '{parts[0]}' saved.")

        @reg("delmacro", "Delete a macro: @delmacro <name>", [], "meta")
        def _delmacro(arg: str):
            macros = self._load_macros()
            if arg.lower() in macros:
                del macros[arg.lower()]
                self.memory.kv_set("macros", json.dumps(macros))
                self.cprint(f"Macro '{arg}' deleted.")
            else:
                self.cprint("Macro not found.", "err")

        @reg("timeline", "Show turns per day", [], "info")
        def _timeline(_: str):
            turns = self.memory.recent_turns(1000)
            counter: Counter = Counter()
            for t in turns:
                counter[t["ts"][:10]] += 1
            if not counter:
                self.cprint("No data.")
                return
            rows = sorted(counter.items())
            self._table("Turns per Day", rows, ["Date", "Turns"])

        @reg("top", "Show most-used responses", [], "info")
        def _top(_: str):
            rows = self.memory.list_all()[:10]
            if not rows:
                self.cprint("No responses.")
                return
            table_rows = [(r["hits"], r["rating"], r["user_voice"])
                          for r in rows]
            self._table("Top Responses", table_rows,
                        ["Hits", "Rating", "Voice"])

        @reg("summary", "Summarize conversation", [], "info")
        def _summary(_: str):
            turns = self.memory.recent_turns(200)
            if not turns:
                self.cprint("No history.")
                return
            pos = sum(1 for t in turns if t["sentiment"] == "positive")
            neg = sum(1 for t in turns if t["sentiment"] == "negative")
            neu = len(turns) - pos - neg
            self.cprint(
                f"Summary of last {len(turns)} turns:\n"
                f"  Positive: {pos}\n"
                f"  Neutral:  {neu}\n"
                f"  Negative: {neg}")

    def _register_macro_commands(self) -> None:
        macros = self._load_macros()
        reg = self.registry.register
        for name, val in macros.items():
            if self.registry.get(name):
                continue
            def make_handler(command_text: str):
                def handler(arg: str):
                    text = command_text
                    if arg:
                        text = text.replace("{args}", arg)
                    if text.startswith("@"):
                        self._dispatch_command(text[1:])
                    else:
                        self._process(text)
                return handler
            fn = make_handler(val)
            self.registry._commands[name.lower()] = (
                fn, f"macro → {val}", [name])
            fn._category = "macros"  # type: ignore

    def _load_macros(self) -> dict[str, str]:
        raw = self.memory.kv_get("macros")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def run(self) -> None:
        self.memory.kv_set("process_started", dt.datetime.now().isoformat())
        self._banner()
        while self.running:
            self._drain_reminders()
            try:
                user_voice = self.prompt_user()
            except (EOFError, KeyboardInterrupt):
                self.cprint("\nInterrupted. Goodbye!", "sys", prefix=False)
                break
            if not user_voice:
                continue
            if self._pending_teach is not None:
                self.memory.learn(self._pending_teach, user_voice)
                self.cprint(f"Learned: {self._pending_teach!r} -> {user_voice!r}")
                self._pending_teach = None
                continue
            if self._pending_edit is not None:
                self.memory.update_response(self._pending_edit, user_voice)
                self.cprint(f"Updated response #{self._pending_edit}.")
                self._pending_edit = None
                continue
            if user_voice.startswith("@"):
                self._dispatch_command(user_voice[1:])
                continue
            self._process(user_voice)

    def _banner(self) -> None:
        art = textwrap.dedent(f"""\
        ╔══════════════════════════════════════════╗
        ║   {self.name_of_bot} — your personal chatbot       
        ║   Hello, {self.name}! Type @help for commands.
        ╚══════════════════════════════════════════╝""")
        self.cprint(art, "sys", prefix=False)

    def _dispatch_command(self, raw: str) -> None:
        parts = raw.split(maxsplit=1)
        cmd_name = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        entry = self.registry.get(cmd_name)
        if not entry:
            self.cprint(f"Unknown command: @{cmd_name}. Try @help.", "err")
            return
        fn, _, _ = entry
        try:
            fn(arg)
        except Exception as exc:
            logging.exception("Command failed")
            self.cprint(f"Command error: {exc}", "err")

    def _process(self, user_voice: str) -> None:
        sentiment = self._sentiment(user_voice) if self.sentiment_enabled else "neutral"
        if self.sentiment_enabled:
            self._mood_counter[sentiment] += 1
            self._mood = self._mood_counter.most_common(1)[0][0] if self._mood_counter else "neutral"
        row = self.memory.find(user_voice)
        if row:
            self.memory.bump_hit(row["id"])
            self._last_row_id = row["id"]
            self._last_response = row["response"]
            self._say(row["response"], user_voice, sentiment)
            return
        row = self.memory.find_fuzzy(user_voice, self.similarity_threshold)
        if row:
            self.memory.bump_hit(row["id"])
            self._last_row_id = row["id"]
            self._last_response = row["response"]
            self._say(row["response"], user_voice, sentiment)
            return
        if self._handle_builtin(user_voice, sentiment):
            return
        if self.auto_learn:
            self._learn_interactive(user_voice, sentiment)
        else:
            fallback = random.choice([
                "I don't know how to answer that yet.",
                "Teach me! Use @teach <phrase> then tell me the reply.",
                "Hmm, I'm not sure. Try @help to see my skills.",
            ])
            self._say(fallback, user_voice, sentiment)

    def _handle_builtin(self, user_voice: str, sentiment: str) -> bool:
        low = user_voice.lower().strip("!?.,;")
        stripped = user_voice.strip()
        if low in GREETINGS or any(low.startswith(g + " ") for g in GREETINGS):
            hour = dt.datetime.now().hour
            part = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
            self._say(random.choice([
                f"Good {part}, {self.name}!",
                f"Hey {self.name}! What can I do for you?",
                "Hello there!",
                "Hi! How can I help?",
            ]), user_voice, sentiment)
            return True

        if low in FAREWELLS:
            self._say(random.choice(
                ["Goodbye!", "See you!", "Take care!", "Farewell, friend!"]),
                user_voice, sentiment)
            self.running = False
            return True

        if low in ("how are you", "how're you", "how are you doing",
                   "how do you feel", "whats up", "what's up"):
            self._say(random.choice([
                "I'm a bundle of logic, but I'm running great!",
                "All circuits nominal. How about you?",
                f"Feeling {self._mood}! How are you?",
            ]), user_voice, sentiment)
            return True

        if low.startswith(("what is your name", "who are you",
                           "what's your name", "whats your name")):
            self._say(f"I'm {self.name_of_bot}, your personal assistant.")
            return True

        if low.startswith(("what time", "what's the time", "whats the time")):
            self._say(dt.datetime.now().strftime("%H:%M:%S"), user_voice, sentiment)
            return True

        if low.startswith(("what is the date", "what's the date",
                           "what day", "whats the date")):
            self._say(dt.datetime.now().strftime("%A, %d %B %Y"),
                      user_voice, sentiment)
            return True

        if low.startswith(("thank", "thanks", "thx", "ty")):
            self._say(random.choice(["You're welcome!", "Anytime!",
                                     "My pleasure.", "Glad to help!"]),
                      user_voice, sentiment)
            return True

        if low in ("help", "commands"):
            self._dispatch_command("help")
            return True

        if low in ("joke", "tell me a joke"):
            self._say(random.choice(JOKES), user_voice, sentiment)
            return True

        if low in ("fact", "tell me a fact", "random fact"):
            self._say(random.choice(FACTS), user_voice, sentiment)
            return True

        if low in ("quote", "inspire me", "inspirational quote"):
            text, author = random.choice(QUOTES)
            self._say(f'"{text}" — {author}', user_voice, sentiment)
            return True

        if re.fullmatch(r"[0-9\.\+\-\*/%\(\)\s\^e]+", stripped) and any(
                op in stripped for op in "+-*/%^"):
            try:
                result = safe_math(stripped.replace("^", "**"))
                self._say(f"= {result}", user_voice, sentiment)
                return True
            except Exception:
                pass

        if sentiment == "negative":
            self._say(random.choice([
                "I'm sorry to hear that. Want to talk about it?",
                "That sounds rough. I'm here if you need me.",
                "Sending you good vibes.",
            ]), user_voice, sentiment)
            return True

        if sentiment == "positive":
            self._say(random.choice([
                "Awesome! Glad to hear it!",
                "That's wonderful!",
                "Love the energy!",
            ]), user_voice, sentiment)
            return True

        return False

    def _learn_interactive(self, user_voice: str, sentiment: str = "neutral") -> None:
        prompt = random.choice(TEACH_ME_PROMPTS).format(
            name=self.name, voice=user_voice)
        self.cprint(prompt)
        try:
            learned = input(f"{C['user']}> {C['reset']}").strip()
        except (EOFError, KeyboardInterrupt):
            self.cprint("Okay, skipping.", "sys", prefix=False)
            return
        if not learned:
            self.cprint("Nothing learned. Moving on.", "sys", prefix=False)
            return
        self.memory.learn(user_voice, learned)
        self._say(learned, user_voice, sentiment)

    def _say(self, response: str, user_voice: str = "",
             sentiment: str = "neutral") -> None:
        self.cprint(response, "bot")
        if user_voice:
            self.memory.log_turn(user_voice, response, sentiment)

    @staticmethod
    def _sentiment(text: str) -> str:
        words = set(re.findall(r"\b\w+\b", text.lower()))
        pos = len(words & POSITIVE_WORDS)
        neg = len(words & NEGATIVE_WORDS)
        if any(e in text for e in ("😊", "😀", "😃", "❤️", "👍", ":)", ":-)")):
            pos += 1
        if any(e in text for e in ("😢", "😭", "😡", "👎", ":(", ":-(")):
            neg += 1
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"

def main() -> int:
    config = load_config()
    log_file = config.get("basic-config", {}).get("log_file", "chatbot.log")
    setup_logging(log_file)
    db_file = config.get("basic-config", {}).get("db_file", "memory.db")
    memory = MemoryStore(db_file)
    bot = Robot(config, memory)
    def _shutdown(*_):
        bot.running = False
        memory.close()
        logging.info("Shutdown complete")
    atexit.register(memory.close)
    try:
        signal.signal(signal.SIGINT, _shutdown)
    except (ValueError, OSError):
        pass
    try:
        bot.run()
    finally:
        memory.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
