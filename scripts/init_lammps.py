# utils/init_lammps.py

import os
import json
import subprocess
from pathlib import Path


def load_cluster_modules_into_python(
    modules=("lammps/md-vis-20250722",),
):
    """
    Load cluster modules in a bash subshell and copy the resulting
    environment variables into the current Python/Jupyter process.
    """

    module_commands = "\n".join(f"module load {module}" for module in modules)

    script = f"""
source /etc/profile
{module_commands}

python3 - <<'PY'
import os, json
print(json.dumps(dict(os.environ)))
PY
"""

    result = subprocess.run(
        script,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        check=True,
    )

    env = json.loads(result.stdout.strip().splitlines()[-1])
    os.environ.update(env)

    return env


def init_lammps(
    modules=("lammps/md-vis-20250722",),
):
    """
    Initialize the Python environment for running LAMMPS from Jupyter.

    Returns
    -------
    LAMMPS_EXE : pathlib.Path
        Path to the LAMMPS executable provided by the loaded module.
    """

    load_cluster_modules_into_python(modules=modules)

    result = subprocess.run(
        "which lmp",
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        check=True,
    )

    LAMMPS_EXE = Path(result.stdout.strip())

    if not LAMMPS_EXE.exists():
        raise FileNotFoundError(f"LAMMPS executable not found: {LAMMPS_EXE}")

    return LAMMPS_EXE