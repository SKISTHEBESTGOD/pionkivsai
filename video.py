import time
import shutil
import sys
import json
import os
import threading
import queue
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Any, List, Tuple

# ====== (Opcjonalnie) kolory przez colorama ======
RESET = BOLD = DIM = ""
FG_CYAN = FG_GREEN = FG_YELLOW = FG_MAGENTA = FG_RED = FG_BLUE = FG_WHITE = ""
COLOR_ENABLED = True

try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init()
    RESET = Style.RESET_ALL
    BOLD = Style.BRIGHT
    DIM = Style.DIM
    FG_CYAN = Fore.CYAN
    FG_GREEN = Fore.GREEN
    FG_YELLOW = Fore.YELLOW
    FG_MAGENTA = Fore.MAGENTA
    FG_RED = Fore.RED
    FG_BLUE = Fore.BLUE
    FG_WHITE = Fore.WHITE
except Exception:
    COLOR_ENABLED = False

# ====== Konfiguracje trudności ======
DIFFICULTIES = {
    5: {"name": "NORMALNY",    "growth_pct": 5.0,  "minutes_per_flip": 60},
    7: {"name": "TRUDNY",      "growth_pct": 7.0,  "minutes_per_flip": 55},
    9: {"name": "EKSTREMALNY", "growth_pct": 11.0, "minutes_per_flip": 50},
    10: {"name": "ULTIMATE",   "growth_pct": 12.0, "minutes_per_flip": 45},
}

EXERCISES = {
    5: {"start_diamonds": 10, "inc_diamonds": 1, "start_classic": 20, "inc_classic": 2, "start_plank_s": 60, "inc_plank_s": 5, "start_dips": 0, "inc_dips": 0, "start_muscleup": 0, "inc_muscleup": 0},
    7: {"start_diamonds": 20, "inc_diamonds": 2, "start_classic": 40, "inc_classic": 4, "start_plank_s": 90, "inc_plank_s": 10, "start_dips": 0, "inc_dips": 0, "start_muscleup": 0, "inc_muscleup": 0},
    9: {"start_diamonds": 30, "inc_diamonds": 3, "start_classic": 60, "inc_classic": 6, "start_plank_s": 120, "inc_plank_s": 15, "start_dips": 10, "inc_dips": 2, "start_muscleup": 0, "inc_muscleup": 0},
    10: {"start_diamonds": 50, "inc_diamonds": 5, "start_classic": 100, "inc_classic": 10, "start_plank_s": 180, "inc_plank_s": 30, "start_dips": 20, "inc_dips": 3, "start_muscleup": 10, "inc_muscleup": 1},
}

SAVE_PATH = "flip_timer_save.json"
EXPORT_CSV_PATH = "flip_timer_history.csv"


# ====== UI helpers ======
def banner(title: str):
    w = shutil.get_terminal_size((100, 24)).columns
    line = "═" * max(10, min(100, w - 2))
    print(f"{FG_MAGENTA}{BOLD}╔{line}╗{RESET}")
    center = title.center(len(line))
    print(f"{FG_MAGENTA}{BOLD}║{RESET} {FG_CYAN}{BOLD}{center}{RESET} {FG_MAGENTA}{BOLD}║{RESET}")
    print(f"{FG_MAGENTA}{BOLD}╚{line}╝{RESET}")

def progress_bar(ratio: float, width: int = 30):
    width = max(10, min(60, width))
    ratio = max(0.0, min(1.0, ratio))
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(ratio * 100)
    return f"[{bar}] {pct:3d}%"

def ring(enabled: bool = True):
    if not enabled:
        return
    try:
        # cross-platform "beep"
        print("\a", end="")
    except Exception:
        pass

def fmt_time_mmss(seconds: int):
    m, s = divmod(max(0, int(seconds)), 60)
    return f"{m:02d}:{s:02d}"

def fmt_hhmmss(seconds: float):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def print_flip_header(flip, speed_mult, minutes_per_flip, growth_pct, quick_mode: bool, control_mode: str):
    print()
    tag = f"{FG_RED}{BOLD}QUICK MODE{RESET} " if quick_mode else ""
    print(f"{tag}{FG_YELLOW}{BOLD}Flip #{flip}{RESET}  "
          f"{FG_GREEN}tempo ×{speed_mult:.3f}{RESET}  "
          f"{FG_CYAN}długość: {minutes_per_flip} min{RESET}  "
          f"{FG_MAGENTA}wzrost: {growth_pct:.1f}%/flip{RESET}  "
          f"{DIM}sterowanie: {control_mode}{RESET}")

def show_exercise_block(title: str, d: int, c: int, p_s: int, dips: int, mu: int, note_boski: bool=False):
    print()
    print(f"{FG_BLUE}{BOLD}{title}{RESET}")
    rows = [
        ("Pompki diamentowe", f"{d}"),
        ("Pompki klasyczne",  f"{c}"),
        ("Plank",             f"{fmt_time_mmss(p_s)}"),
    ]
    if dips > 0:
        rows.append(("Dipy", f"{dips}"))
    if mu > 0:
        rows.append(("Muscle-up", f"{mu}"))
        if note_boski:
            rows.append(("Zamiennik (jeśli brak MU)", f"{mu*2} podciągnięć"))
    for k, v in rows:
        print(f"  • {k}: {FG_CYAN}{v}{RESET}")

def controls_help(control_mode: str):
    if control_mode == "single-key":
        print(f"{FG_WHITE}{DIM}Sterowanie (bez Enter):{RESET} "
              f"{FG_CYAN}[spacja]{RESET}=pauza/wznów, "
              f"{FG_CYAN}[n]{RESET}=następny flip, "
              f"{FG_CYAN}[r]{RESET}=restart flipa, "
              f"{FG_CYAN}[s]{RESET}=staty, "
              f"{FG_CYAN}[h]{RESET}=help, "
              f"{FG_CYAN}[q]{RESET}=zapisz i wyjdź")
    else:
        print(f"{FG_WHITE}{DIM}Sterowanie (komendy + Enter):{RESET} "
              f"{FG_CYAN}pause{RESET}, {FG_CYAN}next{RESET}, {FG_CYAN}restart{RESET}, "
              f"{FG_CYAN}stats{RESET}, {FG_CYAN}help{RESET}, {FG_CYAN}quit{RESET} "
              f"{DIM}(aliasy: p/n/r/s/h/q){RESET}")


# ====== Controls: single-key OR line commands (works in IDE) ======
class KeyReader:
    """
    Best-effort single-key reader (no Enter).
    Works in real terminals. In many IDE consoles it won't work -> we auto-fallback.
    """
    def __init__(self):
        self._is_windows = os.name == "nt"
        self._enabled = False
        self._orig_term_settings = None
        self._fd = None

    def __enter__(self):
        # If stdin isn't a TTY, single-key won't work.
        if not sys.stdin.isatty():
            self._enabled = False
            return self

        if self._is_windows:
            # msvcrt works only in real console
            self._enabled = True
            return self

        try:
            import termios
            import tty
            self._termios = termios
            self._tty = tty
            self._fd = sys.stdin.fileno()
            self._orig_term_settings = termios.tcgetattr(self._fd)
            tty.setraw(self._fd)  # raw mode: we will handle Ctrl+C ourselves
            self._enabled = True
        except Exception:
            self._enabled = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._enabled:
            return
        if self._is_windows:
            self._enabled = False
            return
        try:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._orig_term_settings)
        except Exception:
            pass
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_key(self) -> Optional[str]:
        if not self._enabled:
            return None
        if self._is_windows:
            try:
                import msvcrt
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    # Special keys (arrows etc.)
                    if ch in (b"\x00", b"\xe0"):
                        _ = msvcrt.getch()
                        return None
                    try:
                        return ch.decode("utf-8", errors="ignore")
                    except Exception:
                        return None
            except Exception:
                return None
            return None
        else:
            try:
                import select
                dr, _, _ = select.select([sys.stdin], [], [], 0)
                if dr:
                    ch = sys.stdin.read(1)
                    return ch
            except Exception:
                return None
            return None

class LineCommandReader:
    """
    Always works (also in IDE). Reads commands with Enter in background thread.
    """
    def __init__(self):
        self.q: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def start(self):
        def worker():
            while not self._stop.is_set():
                try:
                    line = input().strip()
                except EOFError:
                    break
                except Exception:
                    continue
                if line:
                    self.q.put(line)
        self._t = threading.Thread(target=worker, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()

    def poll(self) -> Optional[str]:
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

def normalize_command(raw: str) -> Optional[str]:
    if raw is None:
        return None
    s = raw.strip().lower()

    # Single-key mapping
    if s == " ":
        return "pause"
    if s in ("p", "pause"):
        return "pause"
    if s in ("n", "next"):
        return "next"
    if s in ("r", "restart"):
        return "restart"
    if s in ("q", "quit", "exit"):
        return "quit"
    if s in ("h", "help", "?"):
        return "help"
    if s in ("s", "stats"):
        return "stats"
    if s in ("eta",):
        return "toggle_eta"
    if s in ("beep", "sound"):
        return "toggle_beep"
    if s in ("export", "csv"):
        return "export"
    return None


# ====== Save / Load ======
@dataclass
class RunSettings:
    show_eta: bool = True
    beep: bool = True
    autosave: bool = True
    break_seconds: int = 0          # real-time break after each flip
    target_flips: int = 0           # 0 = no target
    ui_fps: int = 15

@dataclass
class FlipHistoryRow:
    flip: int
    speed_mult: float
    sec_len: float
    virtual_seconds: int
    real_active_seconds: float
    paused_seconds: float
    ended_epoch: float

    d: int
    c: int
    plank_s: int
    dips: int
    mu: int

@dataclass
class RunState:
    level: int
    flip: int
    quick_mode: bool
    minutes_per_flip: int
    growth_pct: float
    base_second: float
    min_sleep: float

    cur_diamonds: int
    cur_classic: int
    cur_plank_s: int
    cur_dips: int
    cur_mu: int

    inc_diamonds: int
    inc_classic: int
    inc_plank_s: int
    inc_dips: int
    inc_mu: int

    started_at_epoch: float
    total_active_real_seconds: float = 0.0
    total_paused_real_seconds: float = 0.0

    settings: RunSettings = field(default_factory=RunSettings)
    history: List[FlipHistoryRow] = field(default_factory=list)

def _state_to_json_dict(state: RunState) -> Dict[str, Any]:
    d = asdict(state)
    # dataclasses nested -> dict ok
    return d

def _state_from_json_dict(d: Dict[str, Any]) -> RunState:
    # rebuild nested dataclasses
    settings = RunSettings(**d.get("settings", {}))
    hist_raw = d.get("history", [])
    history = []
    for row in hist_raw:
        try:
            history.append(FlipHistoryRow(**row))
        except Exception:
            pass
    d2 = dict(d)
    d2["settings"] = settings
    d2["history"] = history
    return RunState(**d2)

def save_state(state: RunState):
    payload = _state_to_json_dict(state)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_state() -> Optional[RunState]:
    if not os.path.exists(SAVE_PATH):
        return None
    try:
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _state_from_json_dict(data)
    except Exception:
        return None

def delete_save():
    try:
        if os.path.exists(SAVE_PATH):
            os.remove(SAVE_PATH)
    except Exception:
        pass

def export_history_csv(state: RunState, path: str = EXPORT_CSV_PATH) -> bool:
    try:
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "flip","speed_mult","sec_len","virtual_seconds","real_active_seconds","paused_seconds","ended_epoch",
                "diamonds","classic","plank_s","dips","muscleup"
            ])
            for r in state.history:
                w.writerow([
                    r.flip, f"{r.speed_mult:.6f}", f"{r.sec_len:.6f}", r.virtual_seconds,
                    f"{r.real_active_seconds:.3f}", f"{r.paused_seconds:.3f}", int(r.ended_epoch),
                    r.d, r.c, r.plank_s, r.dips, r.mu
                ])
        return True
    except Exception:
        return False


# ====== Custom difficulty helper ======
def maybe_create_custom() -> Optional[Dict[str, Any]]:
    print(f"\n{FG_CYAN}{BOLD}CUSTOM?{RESET} Jeśli chcesz własny poziom, wpisz {FG_YELLOW}C{RESET}, inaczej Enter.")
    choice = input("Wybór: ").strip().lower()
    if choice != "c":
        return None

    def ask_float(prompt, default):
        raw = input(f"{prompt} (domyślnie {default}): ").strip()
        if not raw:
            return float(default)
        return float(raw.replace(",", "."))

    def ask_int(prompt, default):
        raw = input(f"{prompt} (domyślnie {default}): ").strip()
        if not raw:
            return int(default)
        return int(raw)

    name = input("Nazwa poziomu (np. 'KATRA MODE'): ").strip() or "CUSTOM"
    growth_pct = clamp(ask_float("Wzrost % na flip", 6.0), 0.0, 100.0)
    minutes = clamp(ask_int("Minuty na flip", 60), 1, 24*60)
    quick = input("Quick mode (flip=30s test)? (t/N): ").strip().lower() == "t"

    custom = {"name": name, "growth_pct": float(growth_pct), "minutes_per_flip": int(minutes), "quick_mode": bool(quick)}
    return custom


# ====== Core ======
def create_new_state(level: int, quick_mode: bool=False, override: Optional[Dict[str, Any]]=None) -> RunState:
    if override:
        growth_pct = float(override["growth_pct"])
        minutes_per_flip = int(override["minutes_per_flip"])
        quick_mode = bool(override.get("quick_mode", quick_mode))
    else:
        diff_cfg = DIFFICULTIES[level]
        growth_pct = diff_cfg["growth_pct"]
        minutes_per_flip = diff_cfg["minutes_per_flip"]

    ex_cfg = EXERCISES.get(level, EXERCISES[5])

    return RunState(
        level=level,
        flip=0,
        quick_mode=quick_mode,
        minutes_per_flip=minutes_per_flip,
        growth_pct=growth_pct,
        base_second=1.0,
        min_sleep=0.01,  # 10ms
        cur_diamonds=ex_cfg["start_diamonds"],
        cur_classic=ex_cfg["start_classic"],
        cur_plank_s=ex_cfg["start_plank_s"],
        cur_dips=ex_cfg["start_dips"],
        cur_mu=ex_cfg["start_muscleup"],
        inc_diamonds=ex_cfg["inc_diamonds"],
        inc_classic=ex_cfg["inc_classic"],
        inc_plank_s=ex_cfg["inc_plank_s"],
        inc_dips=ex_cfg["inc_dips"],
        inc_mu=ex_cfg["inc_muscleup"],
        started_at_epoch=time.time(),
        settings=RunSettings(
            show_eta=True,
            beep=True,
            autosave=True,
            break_seconds=0,
            target_flips=0,
            ui_fps=15
        ),
        history=[]
    )

def print_stats(state: RunState, current_flip: int, cur_t: int, total_seconds_flip: int, sec_len: float, paused_seconds: float):
    print()
    print(f"{FG_MAGENTA}{BOLD}STATYSTYKI{RESET}")
    print(f"  • Flipy: {FG_CYAN}{state.flip}{RESET}")
    if state.settings.target_flips > 0:
        print(f"  • Cel flipów: {FG_CYAN}{state.settings.target_flips}{RESET}  ({FG_GREEN}{state.flip}/{state.settings.target_flips}{RESET})")
    print(f"  • Czas aktywny (real): {FG_CYAN}{fmt_hhmmss(state.total_active_real_seconds)}{RESET}")
    print(f"  • Czas pauzy (real):   {FG_CYAN}{fmt_hhmmss(state.total_paused_real_seconds)}{RESET}")
    print(f"  • Aktualny flip #{current_flip}: {FG_CYAN}{fmt_time_mmss(cur_t)}{RESET}/{FG_CYAN}{fmt_time_mmss(total_seconds_flip)}{RESET}")
    print(f"  • Aktualny sec_len: {FG_CYAN}{sec_len:.4f}s{RESET}  (min_sleep={state.min_sleep}s)")
    print(f"  • Pauza w tym flipie: {FG_CYAN}{fmt_hhmmss(paused_seconds)}{RESET}")
    print(f"  • ETA włączone: {FG_CYAN}{'TAK' if state.settings.show_eta else 'NIE'}{RESET} | Dźwięk: {FG_CYAN}{'TAK' if state.settings.beep else 'NIE'}{RESET}")

def run_break(seconds: int, control_poll, control_mode: str):
    if seconds <= 0:
        return
    print()
    print(f"{FG_YELLOW}{BOLD}⏳ PRZERWA: {seconds}s{RESET}  {DIM}(next/n = pomiń){RESET}")
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        left = int(seconds - elapsed)
        if left <= 0:
            break
        cmd_raw = control_poll()
        cmd = normalize_command(cmd_raw) if cmd_raw else None
        if cmd in ("next",):
            print(f"{FG_MAGENTA}{BOLD}⏭ Pominięto przerwę.{RESET}")
            return
        sys.stdout.write(f"\r{FG_GREEN}{progress_bar(elapsed/seconds, 30)}{RESET}  {FG_CYAN}{left:3d}s{RESET}  {DIM}sterowanie: {control_mode}{RESET}   ")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\n")
    sys.stdout.flush()

def run_timer_from_state(state: RunState):
    # pick control mode
    control_mode = "single-key"
    line_reader = None

    with KeyReader() as kr:
        if not kr.enabled:
            control_mode = "line"
            line_reader = LineCommandReader()
            line_reader.start()

        diff_name = DIFFICULTIES.get(state.level, {"name": "CUSTOM"}).get("name", "CUSTOM")
        banner(f"FLIP TIMER+ • POZIOM {state.level} – {diff_name}")
        print(f"{FG_WHITE}{DIM}Konfiguracja:{RESET} {FG_GREEN}{state.growth_pct:.1f}% wzrost/flip{RESET}, "
              f"{FG_CYAN}{state.minutes_per_flip} min/flip{RESET}, "
              f"{FG_MAGENTA}min_sleep={state.min_sleep}s{RESET}, "
              f"{FG_YELLOW}break={state.settings.break_seconds}s{RESET}, "
              f"{FG_BLUE}ETA={'ON' if state.settings.show_eta else 'OFF'}{RESET}")
        controls_help(control_mode)
        print(f"{DIM}Dodatkowo: 'eta' przełącza ETA, 'beep' dźwięk, 'export' zapis CSV (w line-mode wpisz komendę).{RESET}")

        if state.flip == 0:
            show_exercise_block("ZESTAW STARTOWY — wykonaj TERAZ",
                                state.cur_diamonds, state.cur_classic, state.cur_plank_s,
                                state.cur_dips, state.cur_mu, note_boski=(state.level == 10))

        session_start = time.time()

        def poll_control():
            if control_mode == "single-key":
                return kr.get_key()
            else:
                return line_reader.poll() if line_reader else None

        try:
            while True:
                state.flip += 1
                flip = state.flip

                total_seconds_flip = 30 if state.quick_mode else state.minutes_per_flip * 60
                speed_mult = (1.0 + state.growth_pct / 100.0) ** (flip - 1)
                sec_len = max(state.min_sleep, state.base_second / speed_mult)

                # warn if clamped
                if sec_len <= state.min_sleep + 1e-12 and flip > 1:
                    print(f"{FG_RED}{BOLD}⚠ Uwaga:{RESET} sec_len doszło do min_sleep. Dalsze przyspieszanie będzie już „przycięte”.")

                print_flip_header(flip, speed_mult, state.minutes_per_flip, state.growth_pct, state.quick_mode, control_mode)

                paused = False
                pause_started = 0.0
                paused_acc = 0.0

                flip_real_start = time.perf_counter()
                flip_origin = flip_real_start  # moment t=0 w osi "planowanej"
                t = 0
                last_draw = -1

                term_w = shutil.get_terminal_size((100, 24)).columns
                bar_w = max(20, min(50, term_w - 65))
                fps = clamp(state.settings.ui_fps, 5, 60)

                while t <= total_seconds_flip:
                    raw = poll_control()
                    if raw:
                        # handle ctrl+c from raw mode
                        if raw == "\x03":
                            raise KeyboardInterrupt

                        cmd = normalize_command(raw)
                        if cmd == "pause":
                            paused = not paused
                            if paused:
                                pause_started = time.perf_counter()
                                sys.stdout.write(f"\n{FG_YELLOW}{BOLD}⏸ PAUZA{RESET}\n")
                                sys.stdout.flush()
                            else:
                                paused_dur = time.perf_counter() - pause_started
                                paused_acc += paused_dur
                                flip_origin += paused_dur  # shift schedule
                                sys.stdout.write(f"{FG_GREEN}{BOLD}▶ WZNOWIONO{RESET}\n")
                                sys.stdout.flush()

                        elif cmd == "next":
                            sys.stdout.write(f"\n{FG_MAGENTA}{BOLD}⏭ Pomijam do końca flipa...{RESET}\n")
                            sys.stdout.flush()
                            t = total_seconds_flip
                            break

                        elif cmd == "restart":
                            sys.stdout.write(f"\n{FG_RED}{BOLD}↺ Restart flipa...{RESET}\n")
                            sys.stdout.flush()
                            paused = False
                            paused_acc = 0.0
                            t = 0
                            last_draw = -1
                            flip_real_start = time.perf_counter()
                            flip_origin = flip_real_start
                            continue

                        elif cmd == "quit":
                            if state.settings.autosave:
                                save_state(state)
                            sys.stdout.write(f"\n{FG_YELLOW}{BOLD}Zapisano stan i zakończono.{RESET}\n")
                            sys.stdout.flush()
                            return

                        elif cmd == "help":
                            print()
                            controls_help(control_mode)

                        elif cmd == "stats":
                            print_stats(state, flip, t, total_seconds_flip, sec_len, paused_acc)

                        elif cmd == "toggle_eta":
                            state.settings.show_eta = not state.settings.show_eta
                            print(f"\n{FG_BLUE}{BOLD}ETA:{RESET} {'ON' if state.settings.show_eta else 'OFF'}")

                        elif cmd == "toggle_beep":
                            state.settings.beep = not state.settings.beep
                            print(f"\n{FG_BLUE}{BOLD}DŹWIĘK:{RESET} {'ON' if state.settings.beep else 'OFF'}")

                        elif cmd == "export":
                            ok = export_history_csv(state)
                            print(f"\n{FG_BLUE}{BOLD}EXPORT CSV:{RESET} {'OK -> ' + EXPORT_CSV_PATH if ok else 'NIEUDAŁO SIĘ'}")

                    if paused:
                        time.sleep(0.05)
                        continue

                    # draw throttling
                    now = time.perf_counter()
                    frame = int((now - flip_origin) * fps)
                    if frame != last_draw:
                        last_draw = frame
                        ratio = t / total_seconds_flip if total_seconds_flip else 1.0
                        bar = progress_bar(ratio, bar_w)

                        # clock
                        if state.quick_mode:
                            clock = f"{fmt_time_mmss(t)}/00:{total_seconds_flip:02d}"
                        else:
                            clock = f"{fmt_time_mmss(t)}/{state.minutes_per_flip:02d}:00"

                        eta = ""
                        if state.settings.show_eta:
                            remaining_virtual = (total_seconds_flip - t)
                            remaining_real = remaining_virtual * sec_len
                            eta = f"  {FG_YELLOW}ETA {fmt_hhmmss(remaining_real)}{RESET}"

                        sys.stdout.write(
                            f"\r{FG_GREEN}{bar}{RESET}  {FG_CYAN}{clock}{RESET}{eta}  {DIM}(pauza/next/restart/quit){RESET}   "
                        )
                        sys.stdout.flush()

                    if t == total_seconds_flip:
                        break

                    # pacing: next tick at (t+1)*sec_len from flip_origin
                    target = flip_origin + (t + 1) * sec_len
                    to_sleep = target - time.perf_counter()
                    if to_sleep > 0:
                        time.sleep(to_sleep)
                    t += 1

                sys.stdout.write("\n")
                ring(state.settings.beep)
                print(f"{FG_YELLOW}{BOLD}✔ KONIEC FLIPA #{flip}!{RESET}")

                flip_real_end = time.perf_counter()
                real_active = max(0.0, (flip_real_end - flip_real_start) - paused_acc)

                state.total_active_real_seconds += real_active
                state.total_paused_real_seconds += paused_acc

                # Progresja ćwiczeń (po flipie)
                state.cur_diamonds += state.inc_diamonds
                state.cur_classic  += state.inc_classic
                state.cur_plank_s  += state.inc_plank_s
                state.cur_dips     += state.inc_dips
                state.cur_mu       += state.inc_mu

                show_exercise_block("ĆWICZENIA PO FLIPIE",
                                    state.cur_diamonds, state.cur_classic, state.cur_plank_s,
                                    state.cur_dips, state.cur_mu, note_boski=(state.level == 10))

                # history
                state.history.append(FlipHistoryRow(
                    flip=flip,
                    speed_mult=speed_mult,
                    sec_len=sec_len,
                    virtual_seconds=total_seconds_flip,
                    real_active_seconds=real_active,
                    paused_seconds=paused_acc,
                    ended_epoch=time.time(),
                    d=state.cur_diamonds,
                    c=state.cur_classic,
                    plank_s=state.cur_plank_s,
                    dips=state.cur_dips,
                    mu=state.cur_mu
                ))
                # keep history from exploding
                if len(state.history) > 1000:
                    state.history = state.history[-1000:]

                # target flips -> stop
                if state.settings.target_flips > 0 and state.flip >= state.settings.target_flips:
                    print(f"{FG_GREEN}{BOLD}🏁 Osiągnięto cel flipów: {state.settings.target_flips}{RESET}")
                    if state.settings.autosave:
                        save_state(state)
                    export_history_csv(state)
                    return

                # autosave
                if state.settings.autosave:
                    save_state(state)

                # break
                run_break(state.settings.break_seconds, poll_control, control_mode)

        except KeyboardInterrupt:
            if state.settings.autosave:
                save_state(state)
            print(f"\n{FG_RED}{BOLD}Przerwano (CTRL+C). Stan zapisany.{RESET}")

        finally:
            if line_reader:
                line_reader.stop()
            elapsed = time.time() - session_start
            print(f"{FG_WHITE}{DIM}Statystyki sesji:{RESET} flipy={FG_CYAN}{state.flip}{RESET}, "
                  f"czas rzeczywisty={FG_CYAN}{fmt_hhmmss(elapsed)}{RESET}, "
                  f"czas aktywny(real)={FG_CYAN}{fmt_hhmmss(state.total_active_real_seconds)}{RESET}, "
                  f"pauza(real)={FG_CYAN}{fmt_hhmmss(state.total_paused_real_seconds)}{RESET}")
            # always try export (best-effort)
            export_history_csv(state)


def main():
    banner("FLIP TIMER+ • START")

    saved = load_state()
    if saved:
        print(f"{FG_YELLOW}{BOLD}Znaleziono zapisany stan!{RESET}")
        print(f"  • Poziom: {FG_CYAN}{saved.level}{RESET}")
        print(f"  • Flipy:  {FG_CYAN}{saved.flip}{RESET}")
        print(f"  • Tryb:   {FG_CYAN}{'QUICK' if saved.quick_mode else 'NORMAL'}{RESET}")
        print(f"  • Break:  {FG_CYAN}{saved.settings.break_seconds}s{RESET} | ETA: {FG_CYAN}{'ON' if saved.settings.show_eta else 'OFF'}{RESET}")
        ans = input("Wznowić? (T/n/usun): ").strip().lower()
        if ans in ("", "t", "tak", "y", "yes"):
            run_timer_from_state(saved)
            return
        if ans.startswith("u"):
            delete_save()
            print(f"{FG_GREEN}Zapis usunięty.{RESET}")

    custom = maybe_create_custom()

    print(f"\n{BOLD}{FG_CYAN}Wybierz poziom trudności:{RESET}")
    print(f"  {FG_GREEN}5{RESET} – normalny")
    print(f"  {FG_YELLOW}7{RESET} – trudny")
    print(f"  {FG_BLUE}9{RESET} – ekstremalny")
    print(f"  {FG_RED}10{RESET} – ULTIMATE")
    try:
        raw = input("\nPodaj poziom (5/7/9/10): ").strip()
        level = int(raw) if raw else 5
        if level not in DIFFICULTIES:
            raise ValueError
    except Exception:
        print(f"{FG_RED}Nieprawidłowy wybór. Ustawiam poziom 5 (normalny).{RESET}")
        level = 5

    quick_mode = False
    if not custom:
        quick_mode = input("Quick mode (flip=30s test UI)? (t/N): ").strip().lower() == "t"

    state = create_new_state(level=level, quick_mode=quick_mode, override=custom)

    # proste ustawienia bonusowe (bez rozbudowanego menu)
    try:
        br = input("Przerwa po flipie (sekundy, Enter=0): ").strip()
        if br:
            state.settings.break_seconds = int(clamp(int(br), 0, 3600))
    except Exception:
        pass
    try:
        tg = input("Cel flipów w tej sesji (0=brak, Enter=0): ").strip()
        if tg:
            state.settings.target_flips = int(clamp(int(tg), 0, 100000))
    except Exception:
        pass

    save_state(state)
    run_timer_from_state(state)


if __name__ == "__main__":
    main()
