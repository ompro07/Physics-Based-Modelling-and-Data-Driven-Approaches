def detect_lammps_atom_style(data_file):
    """
    Detect LAMMPS atom_style from the Atoms section header.

    Supports:
        Atoms # atomic
        Atoms # charge
        Atoms # molecular
        Atoms # full

    Returns MDAnalysis atom_style string.
    """

    styles = {
        "atomic": "id type x y z",
        "charge": "id type charge x y z",
        "molecular": "id resid type x y z",
        "full": "id resid type charge x y z",
    }

    with open(data_file, "r") as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("Atoms"):
                if "#" in stripped:
                    style = stripped.split("#", 1)[1].strip().lower()
                    if style in styles:
                        return styles[style]
                break

    # Fallback: inspect first atom line after Atoms section
    in_atoms = False

    with open(data_file, "r") as f:
        for line in f:
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith("Atoms"):
                in_atoms = True
                continue

            if in_atoms:
                if stripped.startswith("#"):
                    continue

                if stripped[0].isalpha():
                    break

                parts = stripped.split()

                if len(parts) == 5:
                    return "id type x y z"

                if len(parts) == 6:
                    # Your hybrid files use molecular:
                    # id mol type x y z
                    return "id resid type x y z"

                if len(parts) == 7:
                    return "id resid type charge x y z"

                break

    raise ValueError(f"Could not detect LAMMPS atom_style from: {data_file}")


def show_py3dmol(
    data_file,
    traj_file=None,
    width=800,
    height=600,
    animate=True,
    interval=100,
    frame_stride=1,
):
    """
    Visualize a LAMMPS data file or a LAMMPS dump trajectory with py3Dmol.

    Parameters
    ----------
    data_file : str or pathlib.Path
        Required LAMMPS data file. Can also be a directory containing one .data file.

    traj_file : str or pathlib.Path or None
        Optional LAMMPS dump trajectory, e.g. traj.lammpstrj.
        If provided, all frames are converted to a multi-frame XYZ and animated.

    width, height : int
        Viewer size.

    animate : bool
        If True, animate trajectory frames.

    interval : int
        Animation interval in ms.

    frame_stride : int
        Write every Nth frame from the trajectory.
        Use frame_stride=10 for long trajectories.
    """

    import re
    from pathlib import Path

    import MDAnalysis as mda
    import py3Dmol

    def guess_element_from_mass(mass):
        periodic_masses = {
            "H": 1.008,
            "C": 12.011,
            "N": 14.007,
            "O": 15.999,
            "F": 18.998,
            "Na": 22.990,
            "Mg": 24.305,
            "Al": 26.982,
            "Si": 28.085,
            "P": 30.974,
            "S": 32.060,
            "Cl": 35.450,
            "K": 39.098,
            "Ca": 40.078,
            "Fe": 55.845,
            "Cu": 63.546,
            "Zn": 65.380,
            "Br": 79.904,
            "I": 126.904,
        }

        return min(
            periodic_masses,
            key=lambda element: abs(periodic_masses[element] - mass),
        )

    def read_type_map_from_lammps_data(data_file):
        type_map = {}
        in_masses = False

        with open(data_file, "r") as f:
            for line in f:
                stripped = line.strip()

                if not stripped:
                    continue

                if stripped.startswith("Masses"):
                    in_masses = True
                    continue

                if in_masses:
                    # Stop at next LAMMPS section
                    if re.match(r"^[A-Za-z]", stripped):
                        break

                    line_no_comment = stripped.split("#", 1)[0].strip()
                    parts = line_no_comment.split()

                    if len(parts) >= 2:
                        atom_type = int(parts[0])
                        mass = float(parts[1])
                        type_map[atom_type] = guess_element_from_mass(mass)

        if not type_map:
            raise ValueError("Could not read atom types from the Masses section.")

        return type_map

    def resolve_data_file(path_like):
        path = Path(path_like)

        # If user passes folder/outname, resolve to folder/folder.data
        if path.is_dir():
            candidate = path / f"{path.name}.data"

            if candidate.exists():
                return candidate

            data_files = sorted(path.glob("*.data"))

            if len(data_files) == 1:
                return data_files[0]

            if not data_files:
                raise FileNotFoundError(f"No .data file found in directory: {path}")

            raise ValueError(
                f"Multiple .data files found in {path}. Pass data_file explicitly."
            )

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        return path

    def resolve_traj_file(path_like):
        if path_like is None:
            return None

        path = Path(path_like)

        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")

        return path

    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    data_path = resolve_data_file(data_file)
    traj_path = resolve_traj_file(traj_file)

    atom_style = detect_lammps_atom_style(data_path)
    type_map = read_type_map_from_lammps_data(data_path)

    # ------------------------------------------------------------------
    # Create MDAnalysis Universe
    # ------------------------------------------------------------------

    if traj_path is None:
        u = mda.Universe(
            str(data_path),
            format="DATA",
            atom_style=atom_style,
        )

    else:
        # Read data file as topology
        u = mda.Universe(
            str(data_path),
            topology_format="DATA",
            atom_style=atom_style,
        )

        # Load LAMMPS dump trajectory as coordinates
        u.load_new(
            str(traj_path),
            format="LAMMPSDUMP",
        )

    # ------------------------------------------------------------------
    # Assign names/elements from Masses section
    # ------------------------------------------------------------------

    types = u.atoms.types.astype(int)
    names = [type_map.get(t, "X") for t in types]

    try:
        u.add_TopologyAttr("names", names)
    except Exception:
        try:
            u.atoms.names = names
        except Exception:
            pass

    try:
        u.add_TopologyAttr("elements", names)
    except Exception:
        try:
            u.atoms.elements = names
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Write XYZ and visualize
    # ------------------------------------------------------------------

    if traj_path is None:
        outfile = str(data_path.with_suffix(".xyz"))

        with mda.Writer(outfile, n_atoms=len(u.atoms)) as w:
            w.write(u.atoms)

        with open(outfile, "r") as f:
            xyz_data = f.read()

        view = py3Dmol.view(width=width, height=height)
        view.addModel(xyz_data, "xyz")
        view.setStyle({"stick": {}})
        view.zoomTo()
        # view.show()

        print(f"Data file : {data_path}")
        print("Displayed : data-file coordinates")
        print(f"XYZ written to: {outfile}")

        return view

    else:
        outfile = str(data_path.with_name(f"{data_path.stem}_traj.xyz"))

        n_written = 0

        with mda.Writer(outfile, n_atoms=len(u.atoms)) as w:
            for iframe, ts in enumerate(u.trajectory):
                if iframe % frame_stride != 0:
                    continue

                w.write(u.atoms)
                n_written += 1

        with open(outfile, "r") as f:
            xyz_data = f.read()

        view = py3Dmol.view(width=width, height=height)
        view.addModelsAsFrames(xyz_data, "xyz")
        view.setStyle({"stick": {}})
        view.zoomTo()

        if animate:
            view.animate(
                {
                    "loop": "forward",
                    "reps": 0,
                    "interval": interval,
                }
            )

        #  view.show()

        print(f"Data file : {data_path}")
        print(f"Trajectory: {traj_path}")
        print("Displayed : full trajectory")
        print(f"XYZ written to: {outfile}")
        print(f"Frames in trajectory : {len(u.trajectory)}")
        print(f"Frames displayed     : {n_written}")
        print(f"Frame stride         : {frame_stride}")

        return view


from pathlib import Path
import subprocess
import sys


def gen_cnt(
    N,
    M,
    repeat,
    functionalization,
    perc,
    outname,
    HOME_DIR,
    vacuum=25.0,
    seed=7,
    model=None,
    airebo_file=None,
    python_exe=None,
    timeout=60,
):
    home = Path(HOME_DIR).resolve()
    workdir = home / outname
    workdir.mkdir(parents=True, exist_ok=True)

    generator = home / "scripts" / "cnt_generator.py"
    data_file = workdir / f"{outname}.data"
    input_file = workdir / "in.relaxation"

    if python_exe is None:
        python_exe = sys.executable

    if not generator.exists():
        raise FileNotFoundError(f"Cannot find generator script: {generator}")

    if functionalization is None:
        functionalization = "none"

    functionalization = str(functionalization)

    if functionalization.lower() == "none":
        functionalization = "none"
        perc = 0.0

    if model is None:
        if functionalization in {"none", "H"}:
            model = "airebo"
        else:
            model = "hybrid"

    if model == "airebo" and functionalization not in {"none", "H"}:
        raise ValueError("Use model='hybrid' for OH, NH2, or COOH.")

    if airebo_file is None:
        airebo_file = "/mnt/sonne/WORK/programs/lammps-22Jul2025/potentials/CH.airebo"

    cmd = [
        python_exe,
        str(generator),
        "--n", str(N),
        "--m", str(M),
        "--repeat", str(repeat),
        "--functional", functionalization,
        "--percent", str(perc),
        "--seed", str(seed),
        "--model", model,
        "--vacuum", str(vacuum),
        "--output", str(data_file),
        "--input", str(input_file),
        "--airebo-file", str(airebo_file),
    ]

    print("Generating CNT:")
    print(" ".join(cmd), flush=True)

    result = subprocess.run(
        cmd,
        cwd=str(home),
        text=True,
        capture_output=True,
        timeout=timeout,
    )

    print("STDOUT:")
    print(result.stdout)

    if result.stderr.strip():
        print("STDERR:")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"cnt_generator.py failed with exit code {result.returncode}")

    if not data_file.exists():
        raise RuntimeError(f"Data file was not created: {data_file}")

    if not input_file.exists():
        raise RuntimeError(f"Input file was not created: {input_file}")

    print(f"Generated data : {data_file}")
    print(f"Generated input: {input_file}")

    return data_file


from pathlib import Path
import os


def write_vmd_beads(data_file=None, traj_file=None, desktop_dir=None):
    """
    Write a Bash launcher and a Tcl script to the Desktop so that VMD can be
    opened from the desktop environment.

    Parameters
    ----------
    data_file : str or Path
        LAMMPS data file.
    traj_file : str or Path, optional
        LAMMPS trajectory file.
    desktop_dir : str or Path, optional
        Target directory for the launcher scripts. Defaults to ~/Desktop.
    """

    if data_file is None:
        raise ValueError("Please provide a LAMMPS data file.")

    if desktop_dir is None:
        desktop_dir = Path.home() / "Desktop"
    else:
        desktop_dir = Path(desktop_dir)

    desktop_dir.mkdir(parents=True, exist_ok=True)

    bash_filename = desktop_dir / "launch_vmd_beads.sh"
    tcl_filename = desktop_dir / "vmd_beads.tcl"
    desktop_launcher = desktop_dir / "VMD_Beads.desktop"

    data_file = Path(data_file).resolve()

    if traj_file is not None:
        traj_file = Path(traj_file).resolve()

    # Bash launcher
    if traj_file is None:
        bash_script = f"""#!/bin/bash
vmd -e "{tcl_filename}" -args "{data_file}"
"""
    else:
        bash_script = f"""#!/bin/bash
vmd -e "{tcl_filename}" -args "{data_file}" "{traj_file}"
"""

    # Tcl script for VMD
    tcl_script = r"""package require topotools
package require pbctools

set num_input [llength $argv]

if {$num_input == 0} {
    exit
} elseif {$num_input == 1} {
    topo readlammpsdata [lindex $argv 0]
} elseif {$num_input == 2} {
    topo readlammpsdata [lindex $argv 0]
    # animate delete beg 0 end -1 skip 0 0
    mol addfile [lindex $argv 1] type lammpstrj first 0 last -1 step 1 waitfor -1
} else {
    exit
}

topo guessatom element mass

mol modcolor 0 0 ResID
mol modstyle 0 0 CPK 0.500000 0.200000 12.000000 12.000000

pbc join residue -bondlist -all
pbc box

color Display Background white
display projection Orthographic

display resize 800 800
display resetview

animate goto 0

"""

    # Desktop launcher
    desktop_entry = f"""[Desktop Entry]
Type=Application
Name=VMD Beads
Comment=Open bead-spring polymer system in VMD
Exec=/bin/xfce4-terminal --hold --command="/bin/bash -lc '{bash_filename}'"
Path={desktop_dir}
Terminal=false
StartupNotify=false
Categories=Science;
"""

    
    bash_filename.write_text(bash_script)
    tcl_filename.write_text(tcl_script)
    desktop_launcher.write_text(desktop_entry)

    os.chmod(bash_filename, 0o755)
    os.chmod(desktop_launcher, 0o755)

    print(f"Wrote desktop launcher: VMD Beads")
    # print(f"Wrote VMD launcher: {bash_filename}")
    # print(f"Wrote VMD Tcl script: {tcl_filename}")
    # print(f"Wrote desktop launcher: {desktop_launcher}")


def write_vmd_polymer(data_file=None, traj_file=None, desktop_dir=None):
    """
    Write a Bash launcher and a Tcl script to the Desktop so that VMD can be
    opened from the desktop environment.

    Parameters
    ----------
    data_file : str or Path
        LAMMPS data file.
    traj_file : str or Path, optional
        LAMMPS trajectory file.
    desktop_dir : str or Path, optional
        Target directory for the launcher scripts. Defaults to ~/Desktop.
    """

    if data_file is None:
        raise ValueError("Please provide a LAMMPS data file.")

    if desktop_dir is None:
        desktop_dir = Path.home() / "Desktop"
    else:
        desktop_dir = Path(desktop_dir)

    desktop_dir.mkdir(parents=True, exist_ok=True)

    bash_filename = desktop_dir / "launch_vmd_polymer.sh"
    tcl_filename = desktop_dir / "vmd_polymer.tcl"
    desktop_launcher = desktop_dir / "VMD_Polymer.desktop"

    data_file = Path(data_file).resolve()

    if traj_file is not None:
        traj_file = Path(traj_file).resolve()

    # Bash launcher
    if traj_file is None:
        bash_script = f"""#!/bin/bash
vmd -e "{tcl_filename}" -args "{data_file}"
"""
    else:
        bash_script = f"""#!/bin/bash
vmd -e "{tcl_filename}" -args "{data_file}" "{traj_file}"
"""

    # Tcl script for VMD
    tcl_script = r"""package require topotools
package require pbctools

set num_input [llength $argv]

if {$num_input == 0} {
    exit
} elseif {$num_input == 1} {
    topo readlammpsdata [lindex $argv 0]
} elseif {$num_input == 2} {
    topo readlammpsdata [lindex $argv 0]
    animate delete beg 0 end -1 skip 0 0
    mol addfile [lindex $argv 1] type lammpstrj first 0 last -1 step 1 waitfor -1
} else {
    exit
}

topo guessatom element mass

mol addrep 0
mol addrep 0

mol modcolor 0 0 Element
mol modcolor 1 0 Element
mol modcolor 2 0 Element

mol modstyle 0 0 DynamicBonds 1.700000 0.070000 12.000000
mol modstyle 1 0 DynamicBonds 1.200000 0.050000 12.000000
mol modstyle 2 0 VDW 0.0500000 12.000000

mol modselect 0 0 element C N O
mol modselect 1 0 all
mol modselect 2 0 all

pbc wrap -all

color Display Background white
color Element C gray
display projection Orthographic

axes location Off
display resize 800 800
display resetview

"""

    # Desktop launcher
    desktop_entry = f"""[Desktop Entry]
Type=Application
Name=VMD Polymer
Comment=Open atomistic polymer system in VMD
Exec=/bin/xfce4-terminal --hold --command="/bin/bash -lc '{bash_filename}'"
Path={desktop_dir}
Terminal=false
StartupNotify=false
Categories=Science;
"""

    
    bash_filename.write_text(bash_script)
    tcl_filename.write_text(tcl_script)
    desktop_launcher.write_text(desktop_entry)

    os.chmod(bash_filename, 0o755)
    os.chmod(desktop_launcher, 0o755)

    print(f"Wrote desktop launcher: VMD Polymer")
    # print(f"Wrote VMD launcher: {bash_filename}")
    # print(f"Wrote VMD Tcl script: {tcl_filename}")
    # print(f"Wrote desktop launcher: {desktop_launcher}")


def show_smiles(
    smiles,
    width=800,
    height=600,
    add_hydrogens=True,
    optimize=True,
    style="stick",
):
    """
    Generate and visualize a small molecule from a SMILES string.

    Parameters
    ----------
    smiles : str
        Molecular SMILES string.

    width, height : int
        Viewer dimensions in pixels.

    add_hydrogens : bool
        Add explicit hydrogen atoms before generating coordinates.

    optimize : bool
        Optimize the generated 3D geometry using a molecular mechanics
        force field.

    style : {"stick", "sphere", "line", "ball_and_stick"}
        Molecular display style.

    Returns
    -------
    py3Dmol.view
        Interactive py3Dmol viewer.
    """
    import py3Dmol
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")

    # Add explicit hydrogens for a complete 3D structure
    if add_hydrogens:
        mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    params = AllChem.ETKDGv3()
    params.randomSeed = 42

    status = AllChem.EmbedMolecule(mol, params)

    if status != 0:
        raise RuntimeError(
            f"RDKit could not generate a 3D structure for {smiles!r}."
        )

    # Optimize geometry
    if optimize:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol)
            else:
                AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            # Retain the embedded geometry if optimization fails
            pass

    mol_block = Chem.MolToMolBlock(mol)

    view = py3Dmol.view(width=width, height=height)
    view.addModel(mol_block, "mol")

    if style == "stick":
        view.setStyle({"stick": {}})

    elif style == "sphere":
        view.setStyle({"sphere": {"scale": 0.3}})

    elif style == "line":
        view.setStyle({"line": {}})

    elif style == "ball_and_stick":
        view.setStyle(
            {
                "stick": {"radius": 0.15},
                "sphere": {"scale": 0.25},
            }
        )

    else:
        raise ValueError(
            "style must be 'stick', 'sphere', 'line', or 'ball_and_stick'."
        )

    view.zoomTo()
    return view

    