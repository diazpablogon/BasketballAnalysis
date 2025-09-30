import os
import sys
import time
import subprocess


# ==============================
# CONFIG RÁPIDA (edítala tú)
# ==============================

# Temporadas a descargar (coma separadas)
SEASONS = "2024-25"          # Ej: "2022-23,2023-24,2024-25"
INCLUDE_PLAYOFFS = True      # True/False
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
        "name": "Team Dashboards",
        "rel_path": "_01a_get_team_dashboards.py",
        "enabled": True,
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


def _print_task_header(task_name: str, cmd):
    command_str = _format_command(cmd)
    header = f"▶ {task_name}"
    line_length = max(len(header), len(command_str) + 10, 70)
    border = "─" * line_length
    print("\n" + border)
    print(header)
    print(f"   comando: {command_str}")
    print(border)
    return line_length


def _print_task_footer(status: str, duration: float, message: str = "", *, line_length: int = 70):
    icon = "✅" if status == "ok" else "⚠️"
    label = message.strip() if message else ("OK" if status == "ok" else "Error")
    print(f"   {icon} {label} (t={duration:.1f}s)")
    print("─" * line_length)


def _run_task(task):
    script_path = _script_abs_path(task["rel_path"])
    if not os.path.exists(script_path):
        print(f"❌ No encontrado: {script_path}  (task: {task['name']})")
        return {"name": task["name"], "status": "missing", "duration": 0.0}

    # Construye los argumentos estándar esperados por tus scripts
    cmd = [
        sys.executable,
        script_path,
        "--seasons", SEASONS,
        "--sleep", str(SLEEP),
        "--max-retries", str(MAX_RETRIES),
    ]
    if INCLUDE_PLAYOFFS:
        cmd.append("--include-playoffs")

    # Extra args específicos del task (si los tuviera)
    cmd.extend(task.get("extra_args", []))

    line_length = _print_task_header(task["name"], cmd)
    start = time.time()
    try:
        # check=True para que levante excepción si devuelve código != 0
        subprocess.run(cmd, check=True)
        duration = time.time() - start
        _print_task_footer("ok", duration, task["name"], line_length=line_length)
        return {"name": task["name"], "status": "ok", "duration": duration}
    except subprocess.CalledProcessError as e:
        duration = time.time() - start
        _print_task_footer(
            "error",
            duration,
            f"Error en {task['name']} (exit code {e.returncode})",
            line_length=line_length,
        )
        return {"name": task["name"], "status": "error", "duration": duration}
    except Exception as e:
        duration = time.time() - start
        _print_task_footer(
            "error",
            duration,
            f"Excepción en {task['name']}: {e}",
            line_length=line_length,
        )
        return {"name": task["name"], "status": "error", "duration": duration}

def main():
    enabled_tasks = [task for task in TASKS if task.get("enabled", False)]

    print("╔════════════════════════════════════════════════════╗")
    print("║               Downloader TOTAL NBA                 ║")
    print("╠════════════════════════════════════════════════════╣")
    print(f"║ Temporadas: {SEASONS:<35}║")
    print(f"║ Playoffs: {str(INCLUDE_PLAYOFFS):<36}║")
    print(f"║ sleep={SLEEP}s | retries={MAX_RETRIES:<22}║")
    print(f"║ Tareas activas: {len(enabled_tasks)}/{len(TASKS):<29}║")
    print("╚════════════════════════════════════════════════════╝")

    results = []
    for task in TASKS:
        if task.get("enabled", False):
            result = _run_task(task)
            if result is not None:
                results.append(result)
        else:
            print(f"⏭  Saltado: {task['name']}")

    if results:
        print("\nResumen de tareas:")
        for result in results:
            icon = "✅" if result["status"] == "ok" else "⚠️"
            duration = result.get("duration", 0.0)
            print(f"  {icon} {result['name']} (t={duration:.1f}s)")

    print("\n✔ Fin de tareas.")

if __name__ == "__main__":
    main()
