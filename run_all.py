import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VENV_PYTHON = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
PYTHON = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

SCRIPTS = {
    "1": "interactive_replay.py",
    "2": "run_predictor.py",
    "3":  "run_evaluation.py",
}


def launch(script):
    script_path = os.path.join(BASE_DIR, script)
    creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    print(f"Starting {script}...")
    return subprocess.Popen(
        [PYTHON, script_path],
        cwd=BASE_DIR,
        creationflags=creationflags,
    )


def print_menu():
    print("\n" + "=" * 40)
    print("LAUNCHER CONTROLS")
    print("=" * 40)
    for key, script in SCRIPTS.items():
        print(f"  [{key}] (Re)start {script}")
    print("  [Q] Quit (stops all)")
    print("=" * 40)


def main():
    processes = {}
    for key, script in SCRIPTS.items():
        processes[key] = launch(script)

    print_menu()

    try:
        import msvcrt

        while True:
            if msvcrt.kbhit():
                try:
                    char = msvcrt.getch().decode("utf-8").lower()
                except UnicodeDecodeError:
                    continue

                if char == "q":
                    break

                if char in SCRIPTS:
                    proc = processes.get(char)
                    if proc and proc.poll() is None:
                        print(f"Restarting {SCRIPTS[char]}...")
                        proc.terminate()
                        proc.wait()
                    processes[char] = launch(SCRIPTS[char])
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping all scripts...")
        for proc in processes.values():
            if proc and proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()
