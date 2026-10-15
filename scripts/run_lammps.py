import subprocess
import threading


def run_lammps_background(cmd, workdir):
    """
    Run a LAMMPS command in the background and stream stdout/stderr.

    Parameters
    ----------
    cmd : list[str]
        Command as a list, for example:
        ["mpirun", "-np", "8", "lmp", "-in", "in.melt"]
    workdir : str or pathlib.Path
        Directory where the command should be executed.

    Returns
    -------
    process : subprocess.Popen
        Running subprocess object.
    stdout_thread : threading.Thread
        Thread streaming stdout.
    stderr_thread : threading.Thread
        Thread streaming stderr.
    """

    process = subprocess.Popen(
        cmd,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    def read_output(stream, prefix=""):
        for line in iter(stream.readline, ""):
            if line:
                print(prefix + line, end="")
        stream.close()

    stdout_thread = threading.Thread(
        target=read_output,
        args=(process.stdout, "")
    )

    stderr_thread = threading.Thread(
        target=read_output,
        args=(process.stderr, "STDERR: ")
    )

    stdout_thread.daemon = True
    stderr_thread.daemon = True

    stdout_thread.start()
    stderr_thread.start()

    return process, stdout_thread, stderr_thread