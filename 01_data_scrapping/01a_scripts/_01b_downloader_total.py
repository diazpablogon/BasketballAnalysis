import os
import sys
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
        "enabled": True,
        "extra_args": []  # puedes poner más flags si tu script los admite
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

def _run_task(task):
    script_path = _script_abs_path(task["rel_path"])
    if not os.path.exists(script_path):
        print(f"❌ No encontrado: {script_path}  (task: {task['name']})")
        return

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

    print(f"\n▶ Ejecutando: {task['name']}")
    print("   ", " ".join(cmd))

    try:
        # check=True para que levante excepción si devuelve código != 0
        subprocess.run(cmd, check=True)
        print(f"✅ OK: {task['name']}")
    except subprocess.CalledProcessError as e:
        print(f"⚠ Error ejecutando {task['name']} (exit code {e.returncode})")
    except Exception as e:
        print(f"⚠ Excepción en {task['name']}: {e}")

def main():
    print("=== Downloader TOTAL ===")
    print(f"Temporadas: {SEASONS} | Playoffs: {INCLUDE_PLAYOFFS} | sleep={SLEEP}s | retries={MAX_RETRIES}")

    for task in TASKS:
        if task.get("enabled", False):
            _run_task(task)
        else:
            print(f"⏭  Saltado: {task['name']}")

    print("\n✔ Fin de tareas.")

if __name__ == "__main__":
    main()
