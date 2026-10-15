#!/usr/bin/env python3
"""
cnt_generator.py

Generate periodic LAMMPS data files for carbon nanotubes.

Supported models
----------------
1. AIREBO
   - For pristine CNT and H-functionalized CNT only.
   - Elements: C/H only.
   - Uses atom_style atomic.
   - No explicit bonds/angles.

2. AIREBO-hybrid
   - AIREBO for CNT C/H backbone.
   - Harmonic bonds/angles for OH, NH2, COOH functional groups.
   - Generic LJ for O/N-containing nonbonded interactions.
   - Uses atom_style molecular.
   - Nonreactive functional groups.

Important
---------
AIREBO supports only C/H. Therefore OH, NH2, and COOH cannot be treated by
AIREBO directly. The hybrid model treats the CNT backbone with AIREBO and
functional groups with fixed bonded harmonic terms.

Examples
--------
Pristine CNT with AIREBO:
    python cnt_generator.py --n 10 --m 4 --repeat 2 --functional none \\
        --model airebo --vacuum 25 --output cnt.data

H-functionalized CNT with AIREBO:
    python cnt_generator.py --n 10 --m 4 --repeat 2 --functional H --percent 5 \\
        --model airebo --vacuum 25 --output cnt_h.data

OH-functionalized CNT with AIREBO-hybrid:
    python cnt_generator.py --n 10 --m 4 --repeat 2 --functional OH --percent 10 \\
        --model hybrid --vacuum 25 --output cnt_oh.data

NH2-functionalized CNT with AIREBO-hybrid:
    python cnt_generator.py --n 10 --m 4 --repeat 2 --functional NH2 --percent 10 \\
        --model hybrid --vacuum 25 --output cnt_nh2.data

COOH-functionalized CNT with AIREBO-hybrid:
    python cnt_generator.py --n 10 --m 4 --repeat 2 --functional COOH --percent 10 \\
        --model hybrid --vacuum 25 --output cnt_cooh.data
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math
import random
import shutil

# ----------------------------------------------------------------------
# Basic 2D vector utilities
# ----------------------------------------------------------------------

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def norm(v):
    return math.sqrt(dot(v, v))


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def mul(s, v):
    return (s * v[0], s * v[1])


def gcd3_related(n, m):
    """
    d_R = gcd(2m+n, 2n+m), used for CNT translational vector.
    """
    return math.gcd(2 * m + n, 2 * n + m)


# ----------------------------------------------------------------------
# Atom representation
# ----------------------------------------------------------------------

def make_atom(element, x, y, z, theta, role):
    """
    Internal atom tuple:
        (element, x, y, z, theta, role)

    role is used to distinguish CNT C/H from functional-group atoms.
    """
    return (element, x, y, z, theta, role)


def atom_element(atom):
    return atom[0]


def atom_xyz(atom):
    return atom[1], atom[2], atom[3]


def atom_theta(atom):
    return atom[4]


def atom_role(atom):
    return atom[5]


def element_mass(element):
    element = element.upper()
    masses = {
        "C": 12.011,
        "H": 1.008,
        "O": 15.999,
        "N": 14.007,
    }
    if element not in masses:
        raise ValueError(f"No mass defined for element {element}")
    return masses[element]


# ----------------------------------------------------------------------
# CNT generation
# ----------------------------------------------------------------------

def generate_cnt(n, m, repeat=1, acc=1.42):
    """
    Generate CNT carbon atoms.

    Returns:
        atoms, radius, length

    atoms are:
        (element, x, y, z, theta, role)
    """
    if n <= 0 or m < 0 or m > n:
        raise ValueError("Require n > 0 and 0 <= m <= n. Use n >= m convention.")

    if repeat <= 0:
        raise ValueError("repeat must be positive.")

    # Graphene primitive vectors, C-C bond length = acc.
    # Basis: A=(0,0), B=(0,acc)
    a1 = (math.sqrt(3.0) * acc, 0.0)
    a2 = (math.sqrt(3.0) * acc / 2.0, 1.5 * acc)
    basis = [(0.0, 0.0), (0.0, acc)]

    # Chiral vector C_h and translational vector T.
    Ch = add(mul(n, a1), mul(m, a2))
    dR = gcd3_related(n, m)
    t1 = (2 * m + n) // dR
    t2 = -(2 * n + m) // dR
    T = add(mul(t1, a1), mul(t2, a2))

    Ch_len = norm(Ch)
    T_len = norm(T)

    radius = Ch_len / (2.0 * math.pi)
    length = T_len * repeat

    span = int(abs(n) + abs(m) + repeat * (abs(t1) + abs(t2)) + 8)

    atoms = []
    seen = set()

    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            cell = add(mul(i, a1), mul(j, a2))

            for b in basis:
                r2 = add(cell, b)

                s = dot(r2, Ch) / (Ch_len * Ch_len)
                t = dot(r2, T) / (T_len * T_len)

                if -1e-8 <= t < repeat - 1e-8:
                    theta = 2.0 * math.pi * (s % 1.0)
                    x = radius * math.cos(theta)
                    y = radius * math.sin(theta)
                    z = t * T_len

                    key = (round(x, 5), round(y, 5), round(z, 5))

                    if key not in seen:
                        seen.add(key)
                        atoms.append(
                            make_atom("C", x, y, z, theta, role="C_CNT")
                        )

    atoms.sort(key=lambda a: (a[3], math.atan2(a[2], a[1])))

    return atoms, radius, length


# ----------------------------------------------------------------------
# Functionalization with explicit topology
# ----------------------------------------------------------------------

def functionalize_with_topology(atoms, percent=5.0, group="OH", seed=None):
    """
    Add functional groups and return:
        atoms_out, bonds, angles

    bonds:
        (bond_type, atom_i, atom_j)

    angles:
        (angle_type, atom_i, atom_j, atom_k)

    Atom IDs are 1-based.
    """

    if seed is not None:
        random.seed(seed)

    group = group.upper()

    if group not in {"NONE", "H", "OH", "NH2", "COOH"}:
        raise ValueError("Functional group must be one of: none, H, OH, NH2, COOH")

    atoms_out = list(atoms)
    bonds = []
    angles = []

    if group == "NONE" or percent <= 0.0:
        return atoms_out, bonds, angles

    carbons = [i for i, atom in enumerate(atoms) if atom_role(atom) == "C_CNT"]

    k = max(0, round(len(carbons) * percent / 100.0))
    chosen = random.sample(carbons, min(k, len(carbons)))

    def add_new_atom(element, x, y, z, theta, role):
        atoms_out.append(make_atom(element, x, y, z, theta, role))
        return len(atoms_out)  # 1-based LAMMPS atom ID

    for idx in chosen:
        element, x, y, z, theta, role = atoms[idx]
        cnt_c_id = idx + 1

        nx = math.cos(theta)
        ny = math.sin(theta)
        nz = 0.0

        tx = -math.sin(theta)
        ty = math.cos(theta)
        tz = 0.0

        def xyz(radial, tangential=0.0, axial=0.0):
            return (
                x + radial * nx + tangential * tx,
                y + radial * ny + tangential * ty,
                z + axial,
            )

        if group == "H":
            hx, hy, hz = xyz(1.09)
            h_id = add_new_atom("H", hx, hy, hz, theta, role="H_CNT")

            # Only used in hybrid model. In pure AIREBO these bonds are ignored
            # because no Bonds section is written.
            bonds.append((1, cnt_c_id, h_id))

        elif group == "OH":
            ox, oy, oz = xyz(1.43)
            o_id = add_new_atom("O", ox, oy, oz, theta, role="O_FUNC")

            # Bent OH geometry.
            # O-H ≈ sqrt(0.35^2 + 0.89^2) = 0.956 Å
            hx, hy, hz = xyz(1.43 + 0.35, 0.89, 0.0)
            h_id = add_new_atom("H", hx, hy, hz, theta, role="H_FUNC")

            # Bond types:
            # 2 = C_CNT - O_FUNC
            # 3 = O_FUNC - H_FUNC
            bonds.append((2, cnt_c_id, o_id))
            bonds.append((3, o_id, h_id))

            # Angle type:
            # 1 = C-O-H
            angles.append((1, cnt_c_id, o_id, h_id))

        elif group == "NH2":
            nx1, ny1, nz1 = xyz(1.47)
            n_id = add_new_atom("N", nx1, ny1, nz1, theta, role="N_FUNC")

            h1x, h1y, h1z = xyz(1.47 + 0.35, 0.94, 0.20)
            h2x, h2y, h2z = xyz(1.47 + 0.35, -0.94, -0.20)

            h1_id = add_new_atom("H", h1x, h1y, h1z, theta, role="H_FUNC")
            h2_id = add_new_atom("H", h2x, h2y, h2z, theta, role="H_FUNC")

            # Bond types:
            # 4 = C_CNT - N_FUNC
            # 5 = N_FUNC - H_FUNC
            bonds.append((4, cnt_c_id, n_id))
            bonds.append((5, n_id, h1_id))
            bonds.append((5, n_id, h2_id))

            # Angle types:
            # 2 = C-N-H
            # 3 = H-N-H
            angles.append((2, cnt_c_id, n_id, h1_id))
            angles.append((2, cnt_c_id, n_id, h2_id))
            angles.append((3, h1_id, n_id, h2_id))

        elif group == "COOH":
            cx, cy, cz = xyz(1.54)
            c_id = add_new_atom("C", cx, cy, cz, theta, role="C_FUNC")

            o1x, o1y, o1z = xyz(1.54 + 1.23, 0.0, 0.0)
            o_carbonyl_id = add_new_atom("O", o1x, o1y, o1z, theta, role="O_FUNC")

            o2x, o2y, o2z = xyz(1.54 + 1.32, 0.35, 0.15)
            o_hydroxyl_id = add_new_atom("O", o2x, o2y, o2z, theta, role="O_FUNC")

            hox, hoy, hoz = xyz(1.54 + 1.32 + 0.35, 1.20, 0.20)
            h_id = add_new_atom("H", hox, hoy, hoz, theta, role="H_FUNC")

            # Bond types:
            # 6 = C_CNT - C_FUNC
            # 7 = C_FUNC = O_FUNC, carbonyl-like
            # 8 = C_FUNC - O_FUNC
            # 3 = O_FUNC - H_FUNC
            bonds.append((6, cnt_c_id, c_id))
            bonds.append((7, c_id, o_carbonyl_id))
            bonds.append((8, c_id, o_hydroxyl_id))
            bonds.append((3, o_hydroxyl_id, h_id))

            # Angle types:
            # 4 = C_CNT-C_FUNC-O_FUNC
            # 5 = O-C-O
            # 6 = C-O-H
            angles.append((4, cnt_c_id, c_id, o_carbonyl_id))
            angles.append((4, cnt_c_id, c_id, o_hydroxyl_id))
            angles.append((5, o_carbonyl_id, c_id, o_hydroxyl_id))
            angles.append((6, c_id, o_hydroxyl_id, h_id))

    return atoms_out, bonds, angles


# ----------------------------------------------------------------------
# Type maps
# ----------------------------------------------------------------------

def airebo_atomic_type_map(atoms):
    """
    For pure AIREBO data:
        C -> 1
        H -> 2, if present
    """
    present = {atom_element(a).upper() for a in atoms}

    unsupported = present - {"C", "H"}

    if unsupported:
        raise ValueError(
            "Pure AIREBO supports only C/H. "
            f"Found unsupported elements: {sorted(unsupported)}. "
            "Use --model hybrid for OH, NH2, or COOH."
        )

    elements = ["C"]
    if "H" in present:
        elements.append("H")

    type_map = {element: i + 1 for i, element in enumerate(elements)}

    return elements, type_map


def hybrid_role_type_map(atoms):
    """
    Role-based types for the hybrid model.

    This prevents hydroxyl/amine hydrogens from being treated as AIREBO hydrogens.
    AIREBO should only see:
        C_CNT -> C
        H_CNT -> H

    Functional-group atoms are mapped as NULL in the AIREBO pair_coeff line
    and get LJ plus harmonic bonded terms.
    """
    preferred_roles = [
        "C_CNT",
        "H_CNT",
        "C_FUNC",
        "H_FUNC",
        "O_FUNC",
        "N_FUNC",
    ]

    present_roles = {atom_role(a) for a in atoms}
    roles = [r for r in preferred_roles if r in present_roles]

    type_map = {role: i + 1 for i, role in enumerate(roles)}

    return roles, type_map


def role_element(role):
    if role.startswith("C"):
        return "C"
    if role.startswith("H"):
        return "H"
    if role.startswith("O"):
        return "O"
    if role.startswith("N"):
        return "N"
    raise ValueError(f"Cannot infer element from role {role}")


# ----------------------------------------------------------------------
# LAMMPS writers
# ----------------------------------------------------------------------

def write_print_file(workdir):
    from pathlib import Path

    workdir = Path(workdir)
    print_file = workdir / "in.print"

    print_text = r"""# LAMMPS print/output variables

variable        step        equal step
variable        temp        equal temp
variable        etotal      equal etotal
variable        ke          equal ke
variable        pe          equal pe
variable        press       equal press
variable        vol         equal vol
variable        density     equal density
variable        lx          equal lx
variable        ly          equal ly
variable        lz          equal lz
variable        pxx         equal -pxx*0.1
variable        pyy         equal -pyy*0.1
variable        pzz         equal -pzz*0.1

fix             p all print ${print_steps} "${step} ${temp} ${etotal} ${ke} ${pe} ${press} ${vol} ${density} ${lx} ${ly} ${lz} ${pxx} ${pyy} ${pzz}" file ${print_file} screen no title "# step temp etotal ke pe press vol density lx ly lz pxx pyy pzz"
"""

    print_file.write_text(print_text)
    
def write_lammps_airebo_data(atoms, path, radius, length, vacuum=25.0, comment=""):
    """
    Write pure AIREBO LAMMPS data file.

    LAMMPS:
        units metal
        atom_style atomic
        pair_style airebo 3.0
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    elements, type_map = airebo_atomic_type_map(atoms)

    xy_half = radius + vacuum

    xlo, xhi = -xy_half, xy_half
    ylo, yhi = -xy_half, xy_half
    zlo, zhi = 0.0, length

    with path.open("w", encoding="utf-8") as f:
        f.write(f"LAMMPS AIREBO data: {comment}\n\n")
        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(elements)} atom types\n\n")

        f.write(f"{xlo:16.8f} {xhi:16.8f} xlo xhi\n")
        f.write(f"{ylo:16.8f} {yhi:16.8f} ylo yhi\n")
        f.write(f"{zlo:16.8f} {zhi:16.8f} zlo zhi\n\n")

        f.write("Masses\n\n")
        for element in elements:
            f.write(
                f"{type_map[element]:4d} "
                f"{element_mass(element):12.6f} # {element}\n"
            )

        f.write("\nAtoms # atomic\n\n")
        for atom_id, atom in enumerate(atoms, start=1):
            element, x, y, z, theta, role = atom
            atom_type = type_map[element.upper()]
            z_wrapped = z % length

            f.write(
                f"{atom_id:8d} "
                f"{atom_type:4d} "
                f"{x:16.8f} {y:16.8f} {z_wrapped:16.8f}\n"
            )


def write_lammps_hybrid_data(
    atoms,
    bonds,
    angles,
    path,
    radius,
    length,
    vacuum=25.0,
    comment="",
):
    """
    Write AIREBO-hybrid LAMMPS data file.

    LAMMPS:
        units metal
        atom_style molecular
        pair_style hybrid airebo 3.0 lj/cut 10.0
        bond_style harmonic
        angle_style harmonic

    Bonds and angles are only for functional groups.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    roles, type_map = hybrid_role_type_map(atoms)

    # Bond coefficients, units metal:
    # harmonic: E = K (r-r0)^2
    # K in eV/Angstrom^2, r0 in Angstrom.
    bond_coeffs = {
        1: (22.0, 1.09, "C_CNT-H_CNT"),
        2: (28.0, 1.43, "C_CNT-O_FUNC"),
        3: (30.0, 0.96, "O_FUNC-H_FUNC"),
        4: (25.0, 1.47, "C_CNT-N_FUNC"),
        5: (30.0, 1.01, "N_FUNC-H_FUNC"),
        6: (24.0, 1.54, "C_CNT-C_FUNC"),
        7: (45.0, 1.23, "C_FUNC=O_FUNC"),
        8: (28.0, 1.32, "C_FUNC-O_FUNC"),
    }

    # Angle coefficients, units metal:
    # harmonic angle: E = K (theta-theta0)^2
    # K in eV/radian^2, theta0 in degrees.
    angle_coeffs = {
        1: (2.0, 104.5, "C-O-H"),
        2: (2.0, 109.5, "C-N-H"),
        3: (2.0, 107.0, "H-N-H"),
        4: (2.0, 120.0, "C-C-O"),
        5: (3.0, 120.0, "O-C-O"),
        6: (2.0, 104.5, "C-O-H carboxyl"),
    }

    used_bond_types = sorted({b[0] for b in bonds})
    used_angle_types = sorted({a[0] for a in angles})
    
    max_bond_type = max(used_bond_types) if used_bond_types else 0
    max_angle_type = max(used_angle_types) if used_angle_types else 0

    xy_half = radius + vacuum

    xlo, xhi = -xy_half, xy_half
    ylo, yhi = -xy_half, xy_half
    zlo, zhi = 0.0, length

    with path.open("w", encoding="utf-8") as f:
        f.write(f"LAMMPS AIREBO-hybrid data: {comment}\n\n")

        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(bonds)} bonds\n")
        f.write(f"{len(angles)} angles\n\n")

        f.write(f"{len(roles)} atom types\n")
        f.write(f"{max_bond_type} bond types\n")
        f.write(f"{max_angle_type} angle types\n\n")

        f.write(f"{xlo:16.8f} {xhi:16.8f} xlo xhi\n")
        f.write(f"{ylo:16.8f} {yhi:16.8f} ylo yhi\n")
        f.write(f"{zlo:16.8f} {zhi:16.8f} zlo zhi\n\n")

        f.write("Masses\n\n")
        for role in roles:
            element = role_element(role)
            f.write(
                f"{type_map[role]:4d} "
                f"{element_mass(element):12.6f} # {role}\n"
            )

        if max_bond_type > 0:
            f.write("\nBond Coeffs\n\n")
            for bt in range(1, max_bond_type + 1):
                k, r0, label = bond_coeffs[bt]
                f.write(f"{bt:4d} {k:12.6f} {r0:12.6f} # {label}\n")

        if max_angle_type > 0:
            f.write("\nAngle Coeffs\n\n")
            for at in range(1, max_angle_type + 1):
                k, theta0, label = angle_coeffs[at]
                f.write(f"{at:4d} {k:12.6f} {theta0:12.6f} # {label}\n")

        f.write("\nAtoms # molecular\n\n")
        mol_id = 1
        for atom_id, atom in enumerate(atoms, start=1):
            element, x, y, z, theta, role = atom
            atom_type = type_map[role]
            z_wrapped = z % length

            f.write(
                f"{atom_id:8d} "
                f"{mol_id:6d} "
                f"{atom_type:4d} "
                f"{x:16.8f} {y:16.8f} {z_wrapped:16.8f}\n"
            )

        if bonds:
            f.write("\nBonds\n\n")
            for bond_id, (bond_type, i, j) in enumerate(bonds, start=1):
                f.write(f"{bond_id:8d} {bond_type:4d} {i:8d} {j:8d}\n")

        if angles:
            f.write("\nAngles\n\n")
            for angle_id, (angle_type, i, j, k) in enumerate(angles, start=1):
                f.write(
                    f"{angle_id:8d} {angle_type:4d} "
                    f"{i:8d} {j:8d} {k:8d}\n"
                )


# ----------------------------------------------------------------------
# LAMMPS input writers
# ----------------------------------------------------------------------

def write_airebo_input(input_path, data_filename, airebo_file, elements):
    from pathlib import Path

    input_path = Path(input_path)
    workdir = input_path.parent
    workdir.mkdir(parents=True, exist_ok=True)

    input_path = workdir / "in.relaxation"

    settings_file = workdir / "in.settings"
    forcefield_file = workdir / "in.forcefield"

    deform_file = workdir / "in.deform"

    settings_text = """# LAMMPS settings for pure AIREBO CNT

units metal
atom_style atomic

boundary p p p

timestep 0.0005
"""

    element_map = " ".join(elements)
    forcefield_text = f"""# LAMMPS force field for pure AIREBO CNT

pair_style airebo 3.0
pair_coeff * * {airebo_file} {element_map}
"""

    main_text = f"""# LAMMPS input: pure AIREBO CNT
# This input is valid for pristine CNTs and H-functionalized CNTs only.
# The CNT is described with the AIREBO potential.
# No explicit bonds or angles are used in the data file.

# General LAMMPS settings:
# units, atom_style, boundary conditions, and timestep
include in.settings

# Read the generated CNT structure
read_data {data_filename}

# Read force-field definitions:
# AIREBO pair style and CH.airebo potential file
include in.forcefield

# Print thermodynamic output every 1000 steps
thermo 1000
thermo_style custom step temp pe etotal press

# Relax the periodic box only along the CNT axis, z.
# This is used only during minimization.
fix axial all box/relax z 0.0 vmax 0.001

# Energy minimization with box relaxation.
# cg is required here because fix box/relax cannot be used with min_style fire.
min_style cg
minimize 1.0e-8 1.0e-10 1000 10000

# Stop relaxing the box after minimization
unfix axial

# Start the MD step counter from zero after minimization
reset_timestep 0

# Define output frequency and output file for custom printed quantities
variable print_steps equal 100
variable print_file string relaxation.dat
include in.print

# Write trajectory frames every 100 steps
dump traj all custom 100 relaxation.lammpstrj id type x y z
dump_modify traj sort id

# Assign initial velocities at 300 K
velocity all create 300.0 12345 mom yes rot yes dist gaussian

# Equilibrate the relaxed CNT at 300 K with fixed box dimensions
fix nvt all nvt temp 300.0 300.0 0.05

# Run the NVT relaxation/equilibration
run 10000

# Write the final relaxed structure
write_data relaxation.data
"""

    deform_text = f"""# LAMMPS input: pure AIREBO CNT
# This input is valid for pristine CNTs and H-functionalized CNTs only.
# The CNT is described with the AIREBO potential.
# No explicit bonds or angles are used in the data file.

# General LAMMPS settings:
# units, atom_style, boundary conditions, and timestep
include in.settings

# Read the relaxed CNT structure
read_data relaxation.data

# Read force-field definitions:
# AIREBO pair style and CH.airebo potential file
include in.forcefield

# Print thermodynamic output every 5000 steps
thermo 5000
thermo_style custom step temp pe etotal press

# Define output frequency and output file for custom printed quantities
variable print_steps equal 100
variable print_file string deform.dat
include in.print

# Write trajectory frames every 1000 steps
dump traj all custom 1000 deform.lammpstrj id type x y z
dump_modify traj sort id

# Apply deformation with constant velocity
fix deform all deform 1 z vel 5.0e-1 units box remap x

# Integrate the relaxed CNT at 300 K
fix nvt all nvt temp 300.0 300.0 0.05

# Run the NVT deformation
run 100000

# Write the final deformed structure
write_data deform.data
"""


    settings_file.write_text(settings_text)
    forcefield_file.write_text(forcefield_text)
    input_path.write_text(main_text)
    deform_file.write_text(deform_text)
    write_print_file(workdir)


def lj_params_for_role(role):
    """
    Generic LJ parameters in metal units:
        epsilon in eV
        sigma in Angstrom

    These are placeholders for stabilization, not a validated force field.
    """
    element = role_element(role)

    if element == "C":
        return 0.0046, 3.40
    if element == "H":
        return 0.0013, 2.50
    if element == "O":
        return 0.0067, 3.00
    if element == "N":
        return 0.0064, 3.25

    raise ValueError(f"No LJ parameters for role {role}")


def write_hybrid_input(input_path, data_filename, airebo_file, atoms):
    import math
    from pathlib import Path

    input_path = Path(input_path)
    workdir = input_path.parent
    workdir.mkdir(parents=True, exist_ok=True)

    input_path = workdir / "in.relaxation"

    settings_file = workdir / "in.settings"
    forcefield_file = workdir / "in.forcefield"

    deform_file = workdir / "in.deform"

    roles, type_map = hybrid_role_type_map(atoms)

    # AIREBO element mapping.
    # Only C_CNT and H_CNT participate in AIREBO.
    airebo_map = []
    for role in roles:
        if role == "C_CNT":
            airebo_map.append("C")
        elif role == "H_CNT":
            airebo_map.append("H")
        else:
            airebo_map.append("NULL")

    lj_lines = []

    for i, role_i in enumerate(roles, start=1):
        for j, role_j in enumerate(roles[i - 1:], start=i):
            both_airebo = role_i in {"C_CNT", "H_CNT"} and role_j in {"C_CNT", "H_CNT"}

            if both_airebo:
                continue

            eps_i, sig_i = lj_params_for_role(role_i)
            eps_j, sig_j = lj_params_for_role(role_j)

            eps = math.sqrt(eps_i * eps_j)
            sig = 0.5 * (sig_i + sig_j)

            lj_lines.append(
                f"pair_coeff {i:4d} {j:4d} lj/cut {eps:.8f} {sig:.6f} "
                f"# {role_i}-{role_j}"
            )

    lj_block = "\n".join(lj_lines)

    settings_text = """# LAMMPS settings for AIREBO-hybrid CNT

units metal
atom_style molecular

bond_style harmonic
angle_style harmonic

special_bonds lj 0.0 0.0 0.5

boundary p p p

timestep 0.0005
"""

    forcefield_text = f"""# LAMMPS force field for AIREBO-hybrid CNT

pair_style hybrid airebo 3.0 lj/cut 10.0
pair_coeff * * airebo {airebo_file} {" ".join(airebo_map)}

{lj_block}
"""

    main_text = f"""# LAMMPS input: AIREBO-hybrid CNT
# AIREBO handles the CNT C/H backbone.
# Harmonic bonds and angles handle the functional groups.
# LJ interactions handle non-AIREBO nonbonded interactions.

# General LAMMPS settings:
# units, atom_style, boundary conditions, timestep, bond/angle styles
include in.settings

# Read the generated CNT structure
read_data {data_filename}

# Read force-field definitions:
# AIREBO, LJ pair coefficients, bond coefficients, angle coefficients
include in.forcefield

# Print thermodynamic output every 1000 steps
thermo 1000
thermo_style custom step temp pe ebond eangle evdwl etotal press

# Relax the periodic box only along the CNT axis, z.
# This is used only during minimization.
fix axial all box/relax z 0.0 vmax 0.001

# Energy minimization with box relaxation.
# cg is required here because fix box/relax cannot be used with min_style fire.
min_style cg
minimize 1.0e-8 1.0e-10 1000 10000

# Stop relaxing the box after minimization
unfix axial

# Start the MD step counter from zero after minimization
reset_timestep 0

# Define output frequency and output file for custom printed quantities
variable print_steps equal 100
variable print_file string relaxation.dat
include in.print

# Write trajectory frames every 100 steps
dump traj all custom 100 relaxation.lammpstrj id mol type x y z
dump_modify traj sort id

# Assign initial velocities at 300 K
velocity all create 300.0 12345 mom yes rot yes dist gaussian

# Equilibrate the relaxed CNT at 300 K with fixed box dimensions
fix nvt all nvt temp 300.0 300.0 0.05

# Run the NVT relaxation/equilibration
run 10000

# Write the final relaxed structure
write_data relaxation.data
"""

    
    deform_text = f"""# LAMMPS input: AIREBO-hybrid CNT
# AIREBO handles the CNT C/H backbone.
# Harmonic bonds and angles handle the functional groups.
# LJ interactions handle non-AIREBO nonbonded interactions.

# General LAMMPS settings:
# units, atom_style, boundary conditions, timestep, bond/angle styles
include in.settings

# Read the relaxation CNT structure
read_data relaxation.data

# Read force-field definitions:
# AIREBO, LJ pair coefficients, bond coefficients, angle coefficients
include in.forcefield

# Print thermodynamic output every 5000 steps
thermo 5000
thermo_style custom step temp pe etotal press

# Define output frequency and output file for custom printed quantities
variable print_steps equal 100
variable print_file string deform.dat
include in.print

# Write trajectory frames every 1000 steps
dump traj all custom 1000 deform.lammpstrj id type x y z
dump_modify traj sort id

# Apply deformation with constant velocity
fix deform all deform 1 z vel 5.0e-1 units box remap x

# Integrate the relaxed CNT at 300 K
fix nvt all nvt temp 300.0 300.0 0.05

# Run the NVT deformation
run 100000

# Write the final deformed structure
write_data deform.data
"""
    settings_file.write_text(settings_text)
    forcefield_file.write_text(forcefield_text)
    input_path.write_text(main_text)
    deform_file.write_text(deform_text)
    write_print_file(workdir)
    


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Generate periodic LAMMPS CNT data for AIREBO or AIREBO-hybrid."
    )

    p.add_argument("--n", type=int, required=True, help="CNT chiral index n")
    p.add_argument("--m", type=int, required=True, help="CNT chiral index m, use 0 <= m <= n")
    p.add_argument("--repeat", type=int, default=1, help="Number of unit-cell repeats along tube axis")
    p.add_argument("--acc", type=float, default=1.42, help="C-C bond length in Angstrom")

    p.add_argument(
        "--functional",
        choices=["none", "H", "OH", "NH2", "COOH"],
        default="none",
        help="Functional group"
    )

    p.add_argument(
        "--percent",
        type=float,
        default=5.0,
        help="Percent of CNT carbons to functionalize"
    )

    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for functionalization"
    )

    p.add_argument(
        "--model",
        choices=["auto", "airebo", "hybrid"],
        default="auto",
        help="auto = choose airebo for none/H and hybrid for OH/NH2/COOH; "
             "airebo = C/H only; hybrid = AIREBO + harmonic functional groups"
    )

    p.add_argument(
        "--vacuum",
        type=float,
        default=25.0,
        help="Vacuum padding in x/y for LAMMPS box, Angstrom"
    )

    p.add_argument(
        "--output",
        default="cnt.data",
        help="Output LAMMPS data file"
    )

    p.add_argument(
        "--input",
        default=None,
        help="Optional output LAMMPS input file. Default: in.<output stem>"
    )

    p.add_argument(
        "--airebo-file",
        default="CH.airebo",
        help="Path to CH.airebo as used inside the generated LAMMPS input file"
    )

    args = p.parse_args()

    functional = args.functional.strip().upper()
    
    if functional == "NONE":
        functional = "none"
    
    # Automatic model selection
    if args.model == "auto":
        if functional in {"none", "H"}:
            model = "airebo"
        else:
            model = "hybrid"
    else:
        model = args.model
    
    # If user explicitly asks for AIREBO with O/N groups, switch instead of crashing
    if model == "airebo" and functional not in {"none", "H"}:
        print(
            f"Requested --model airebo with --functional {args.functional}. "
            "Switching to --model hybrid because pure AIREBO supports only C/H."
        )
        model = "hybrid"

    atoms, radius, length = generate_cnt(args.n, args.m, args.repeat, args.acc)

    bonds = []
    angles = []

    if functional != "none":
        atoms, bonds, angles = functionalize_with_topology(
            atoms,
            percent=args.percent,
            group=functional,
            seed=args.seed,
        )

    output_path = Path(args.output)
    
    # Directory where the new output file will be written
    output_directory = output_path.parent.resolve()
    
    for directory in Path(".").glob("cnt_N*"):
        if not directory.is_dir():
            continue
    
        # Do not rename the directory selected for the new output
        if directory.resolve() == output_directory:
            continue
    
        destination = directory.with_name(f"prev_{directory.name}")
    
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
    
        directory.rename(destination)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(args.input) if args.input else output_path.with_name(f"in.{output_path.stem}")
    
    # Copy the AIREBO potential from the scripts directory into the output directory
    script_dir = Path(__file__).resolve().parent
    airebo_source = script_dir / "CH.airebo"
    airebo_target = output_path.parent / "CH.airebo"
    
    if not airebo_source.is_file():
        raise FileNotFoundError(
            f"AIREBO potential file not found: {airebo_source}"
        )
    
    # Avoid SameFileError if the output directory is the scripts directory
    if airebo_source.resolve() != airebo_target.resolve():
        shutil.copy2(airebo_source, airebo_target)
    
    # Only the local filename should be written into in.forcefield
    airebo_filename = airebo_target.name

    comment = (
        f"CNT({args.n},{args.m}) repeat={args.repeat}; "
        f"radius={radius:.4f} A; length={length:.4f} A; "
        f"functional={args.functional}; percent={args.percent}; model={args.model}"
    )
   
    
    
    if model == "airebo":

        elements, _ = airebo_atomic_type_map(atoms)
        
        write_lammps_airebo_data(
            atoms=atoms,
            path=output_path,
            radius=radius,
            length=length,
            vacuum=args.vacuum,
            comment=comment,
        )

        write_airebo_input(
            input_path=input_path,
            data_filename=output_path.name,
            airebo_file=airebo_filename,
            elements=elements,
        )

    elif model == "hybrid":
        write_lammps_hybrid_data(
            atoms=atoms,
            bonds=bonds,
            angles=angles,
            path=output_path,
            radius=radius,
            length=length,
            vacuum=args.vacuum,
            comment=comment,
        )

        write_hybrid_input(
            input_path=input_path,
            data_filename=output_path.name,
            airebo_file=airebo_filename,
            atoms=atoms,
        )

    print(f"Wrote data file : {output_path}")
    print(f"Wrote input file: {input_path}")
    print(comment)
    print(f"Atoms : {len(atoms)}")
    print(f"Bonds : {len(bonds)}")
    print(f"Angles: {len(angles)}")


if __name__ == "__main__":
    main()