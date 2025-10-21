import os
import sys
import time
import subprocess
from typing import Optional


# ==============================
# CONFIG RÁPIDA (edítala tú)
# ==============================

# Temporadas a descargar (coma separadas)
SEASONS = "2024-25"          # Ej: "2022-23,2023-24,2024-25"
INCLUDE_PLAYOFFS = False      # True/False cuando se procesan RS+PO
PLAYOFFS_ONLY = False        # Ignora Regular Season y descarga solo Playoffs
REGULAR_ONLY = False         # Ignora Playoffs
PLAYOFFS_FILTER = "all"     # auto / all → ver script de dashboards
SLEEP = 0.8                  # segundos entre reintentos
MAX_RETRIES = 3              # reintentos por endpoint

# Activa/desactiva los jobs que quieres lanzar ahora.
# Añade aquí tus futuros scripts (mantén la misma firma de argumentos).
TASKS = [
    {
        "name": "Box Scores (todos los variantes)",
        "rel_path": "_01a_get_boxscores.py",
        "enabled": False,
        "extra_args": []  # puedes poner más flags si tu script los admite
    },

    {
        "name": "TeamGameLogs → per GAME (2 filas)",
        "rel_path": "_01a_get_teamgamelogs_by_game.py",
        "enabled": True,
        "extra_args": [],
    },

    {
        "name": "Team Dashboards",
        "rel_path": "_01a_get_team_dashboards.py",
        "enabled": False,
        "extra_args": [],
    },

    # ======= EJEMPLOS para cuando los tengas listos =======
    # {
    #     "name": "Play-by-Play",
    #     "rel_path": "_01c_get_pbp.py",
    #     "enabled": False,
    #     "extra_args": []
    # },
    # {
    #     "name": "Team Stats Season",
    #     "rel_path": "_01d_get_team_stats.py",
    #     "enabled": False,
    #     "extra_args": ["--mode", "regular"]
    # },
    # {
    #     "name": "Players Tracking",
    #     "rel_path": "_01e_get_player_tracking.py",
    #     "enabled": False,
    #     "extra_args": []
    # },
]

# ==============================
# NO TOCAR DE AQUÍ PARA ABAJO
# ==============================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _script_abs_path(rel_path: str) -> str:
    return os.path.join(THIS_DIR, rel_path)


def _format_command(cmd) -> str:
    return " ".join(cmd)


def _print_task(status: str, name: str, secs: float, note: Optional[str] = None):
    icons = {
        "ok": "✅",
        "skipped": "⏭️",
        "error": "❌",
        "warn": "⚠️",
        "running": "⏱️",
    }
    icon = icons.get(status, "•")
    base = f"{icon} {name}"
    base += f" ({secs:.1f}s)"
    if note:
        base += f" — {note}"
    print(base)


def _run_task(task):
    script_path = _script_abs_path(task["rel_path"])
    if not os.path.exists(script_path):
        return {
            "name": task["name"],
            "status": "error",
            "duration": 0.0,
            "note": f"script no encontrado: {script_path}",
            "command": "",
        }

    # Construye los argumentos estándar esperados por tus scripts
    cmd = [
        sys.executable,
        script_path,
        "--seasons", SEASONS,
        "--sleep", str(SLEEP),
        "--max-retries", str(MAX_RETRIES),
    ]

    if task.get("rel_path") == "_01a_get_team_dashboards.py":
        if PLAYOFFS_ONLY:
            cmd.append("--playoffs-only")
        elif REGULAR_ONLY:
            cmd.append("--regular-only")
        elif INCLUDE_PLAYOFFS:
            cmd.append("--include-playoffs")
        else:
            # Solo Regular Season por omisión
            pass

        cmd.extend(["--playoffs-filter", PLAYOFFS_FILTER])
    elif INCLUDE_PLAYOFFS:
        cmd.append("--include-playoffs")

    # Extra args específicos del task (si los tuviera)
    cmd.extend(task.get("extra_args", []))

    start = time.time()
    command_str = _format_command(cmd)
    try:
        # check=True para que levante excepción si devuelve código != 0
        subprocess.run(cmd, check=True)
        duration = time.time() - start
        return {
            "name": task["name"],
            "status": "ok",
            "duration": duration,
            "note": "",
            "command": command_str,
        }
    except subprocess.CalledProcessError as e:
        duration = time.time() - start
        return {
            "name": task["name"],
            "status": "error",
            "duration": duration,
            "note": f"exit code {e.returncode}",
            "command": command_str,
        }
    except Exception as e:
        duration = time.time() - start
        return {
            "name": task["name"],
            "status": "error",
            "duration": duration,
            "note": f"excepción: {e}",
            "command": command_str,
        }

def main():
    enabled_tasks = [task for task in TASKS if task.get("enabled", False)]

    if PLAYOFFS_ONLY and REGULAR_ONLY:
        print("❌ Configuración inválida: PLAYOFFS_ONLY y REGULAR_ONLY no pueden ser verdaderos a la vez")
        return

    print("Downloader TOTAL NBA")
    print(f"Temporadas: {SEASONS}")
    print(
        "Opciones: include_playoffs={include} · playoffs_only={po} · regular_only={ro} · filter={flt}".format(
            include=INCLUDE_PLAYOFFS,
            po=PLAYOFFS_ONLY,
            ro=REGULAR_ONLY,
            flt=PLAYOFFS_FILTER,
        )
    )
    print(f"sleep={SLEEP}s · retries={MAX_RETRIES} · tareas activas={len(enabled_tasks)}/{len(TASKS)}\n")

    results = []
    skipped = 0

    for task in TASKS:
        if task.get("enabled", False):
            result = _run_task(task)
            results.append(result)
            status = result.get("status", "error")
            note = result.get("note") or ""
            command = result.get("command")
            if status == "error" and command:
                note = f"{note} · {command}" if note else command
            _print_task(status, task["name"], result.get("duration", 0.0), note or None)
        else:
            skipped += 1
            _print_task("skipped", task["name"], 0.0, "deshabilitada")

    completed = sum(1 for result in results if result.get("status") == "ok")
    failed = sum(1 for result in results if result.get("status") == "error")
    warned = sum(1 for result in results if result.get("status") == "warn")
    total_duration = sum(result.get("duration", 0.0) for result in results)

    print("\nResumen:")
    print(f"  ✅ completadas: {completed}")
    if warned:
        print(f"  ⚠️ con avisos: {warned}")
    print(f"  ❌ fallidas: {failed}")
    print(f"  ⏭️ saltadas: {skipped}")
    print(f"  ⏱️ tiempo total: {total_duration:.1f}s")

    print("\n✔ Fin de tareas.")

if __name__ == "__main__":
    main()
