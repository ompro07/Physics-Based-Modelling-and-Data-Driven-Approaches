from pathlib import Path
import MDAnalysis as mda
import freesasa


MASS_TO_ELEMENT = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.904,
}

ELEMENT_TO_RADIUS = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
}


def guess_element_from_mass(mass):
    return min(
        MASS_TO_ELEMENT,
        key=lambda element: abs(MASS_TO_ELEMENT[element] - mass),
    )


def get_molecular_surface_area(
    data_file,
    probe_radius=1.4,
    atom_style="id type x y z",
    n_points=960,
):
    """
    Calculate SASA from a LAMMPS data file.

    Parameters
    ----------
    data_file : str
        Path to LAMMPS data file, e.g. "benzene/data.benzene".

    probe_radius : float
        Probe radius in Angstrom. Standard water probe is 1.4 Å.

    atom_style : str
        LAMMPS atom style used in the Atoms section.

    n_points : int
        Number of Shrake-Rupley points per atom.

    Returns
    -------
    area : float
        SASA in Å².
    """

    data_file = Path(data_file)

    if not data_file.exists():
        raise FileNotFoundError(f"Could not find data file: {data_file}")

    u = mda.Universe(
        str(data_file),
        format="DATA",
        atom_style=atom_style,
    )

    coords = u.atoms.positions.reshape(-1).tolist()

    radii = []
    elements = []

    for mass in u.atoms.masses:
        element = guess_element_from_mass(mass)
        elements.append(element)
        radii.append(ELEMENT_TO_RADIUS[element])

    params = freesasa.Parameters({
        "algorithm": freesasa.ShrakeRupley,
        "probe-radius": probe_radius,
        "n-points": n_points,
    })

    result = freesasa.calcCoord(coords, radii, params)

    return result.totalArea()