"""Synthetic EQdyna-output fixture for the data-prep guard tests.

Builds a tiny on-disk case directory that looks enough like a real EQdyna
output + case folder (frt.txt<chunk>, src_evol<chunk>, user_defined_params.py,
optionally fractal_stress.txt) for the shared `create_train_data()` logic in
scripts/utils/prepare.eqdyna.4gns.py and
scripts/utils/prepare.fractal.stress.eqdyna.4gns.py to run on,
deterministically and in milliseconds. Never touches the (not checked out,
249GB) real scenario datasets.

Geometry: 4 fault nodes, 2 strike x 2 dip positions, dx=dy=dz=1000 m.

    station 0: (x=   0, z=    0)   -- surface boundary (x=fxmin, z=fzmax)
    station 1: (x=   0, z=-1000)   -- x=fxmin boundary, z=fzmin (surface)
    station 2: (x=1000, z=    0)   -- x=fxmax boundary, z=fzmax
    station 3: (x=1000, z=-1000)   -- x=fxmax boundary, z=fzmin (surface)

Chosen so every station sits exactly on a fault/surface boundary (within
compute_node_type's tol=dx/100), giving a non-trivial (non-all-zero)
node_type when fault_boundary_node_type_mask=True.
"""
import re
import types
from pathlib import Path

import numpy as np

NROW = 4  # number of fault nodes / stations
DX = DY = DZ = 1000.0
FXMIN, FXMAX = 0.0, 1000.0
FZMIN, FZMAX = -1000.0, 0.0
DT = 1.0
TERM = 15.0
XSOURCE = ZSOURCE = -1.0e6  # far outside the domain -> never within the 3km
                            # "forced to 40 MPa" radius in the fractal-stress
                            # script's create_train_data()

STATIONS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, -1000.0],
    [1000.0, 0.0, 0.0],
    [1000.0, 0.0, -1000.0],
])

# With dt=1.0: timestep_pre = round(15/dt) - 1 = 14, nskip = int32(1.2/dt) = 1,
# timestep_after = 14 - 1 = 13. These are derived once here, from the fixture
# constants above, and reused by every test module (not re-derived from the
# production formula under test).
TIMESTEP_PRE = round(15.0 / DT) - 1  # 14
NSKIP = int(1.2 / DT)                # 1 (np.int32 truncation == plain int here)
TIMESTEP_AFTER = TIMESTEP_PRE - NSKIP  # 13


def velocity_value(timestep_id, station_idx):
    """Deterministic per-(timestep, station) slip-rate baked into src_evol0.
    Always >= 1.0, so no legitimately-computed velocity frame can be
    all-zero, and the (timestep, station) pair can be recovered exactly from
    the value (to one part in 100) to check the nskip offset mapping."""
    return 1.0 + timestep_id * 0.1 + station_idx * 0.01


def write_user_defined_params(case_dir: Path):
    text = (
        "class _Par:\n"
        "    pass\n\n"
        "par = _Par()\n"
        f"par.dx = {DX}\n"
        f"par.dy = {DY}\n"
        f"par.dz = {DZ}\n"
        f"par.dt = {DT}\n"
        f"par.term = {TERM}\n"
        f"par.fxmin = {FXMIN}\n"
        f"par.fxmax = {FXMAX}\n"
        f"par.fzmin = {FZMIN}\n"
        f"par.fzmax = {FZMAX}\n"
        f"par.xsource = {XSOURCE}\n"
        f"par.zsource = {ZSOURCE}\n"
    )
    (case_dir / "user_defined_params.py").write_text(text)


def write_station_grid(case_dir: Path):
    # frt.txt0: columns are (x, y, z) in meters; genMapsForEQDYNA's 'src'
    # mapType reads col 0 (x) and col 2 (z).
    np.savetxt(case_dir / "frt.txt0", STATIONS, fmt="%.6f")


def write_src_evol(case_dir: Path, max_timestep_id: int = 20):
    # One float64 per (timestep, station), laid out as t*NROW + station,
    # matching genMapsForEQDYNA.genMap()'s read offset
    # (timeStepId*numOfSt*nValue + i*nValue, nValue=1 for mapType='src').
    n = (max_timestep_id + 1) * NROW
    vals = np.zeros(n, dtype=np.float64)
    for k in range(n):
        t, i = divmod(k, NROW)
        vals[k] = velocity_value(t, i)
    vals.tofile(case_dir / "src_evol0")


def write_fractal_stress(case_dir: Path):
    # ndx = round((fxmax-fxmin)/dx)+1 = 2, ndz = round((fzmax-fzmin)/dz)+1 = 2
    # -> 4 rows needed, indexed by idz*ndx+idx. Column index 2 (0-based) is
    # read as the shear stress in Pa; create_train_data() does
    # np.loadtxt(..., skiprows=7).
    header = "\n".join(f"# header line {i}" for i in range(7)) + "\n"
    rows = [(0.0, 0.0, 40e6), (0.0, 0.0, 41e6), (0.0, 0.0, 42e6), (0.0, 0.0, 43e6)]
    body = "\n".join(f"{r[0]:.1f} {r[1]:.1f} {r[2]:.6e}" for r in rows)
    (case_dir / "fractal_stress.txt").write_text(header + body + "\n")


def build_case(tmp_path: Path, with_fractal_stress: bool = False) -> str:
    """Builds caseroot/scenario0/ under tmp_path and returns the 2-component
    relative path create_train_data() expects as `caseName` -- it does
    `os.chdir(caseName)` on the way in and `os.chdir('../..')` on the way
    out, i.e. it expects a 2-level-deep relative path."""
    case_dir = tmp_path / "caseroot" / "scenario0"
    case_dir.mkdir(parents=True)
    write_user_defined_params(case_dir)
    write_station_grid(case_dir)
    write_src_evol(case_dir)
    if with_fractal_stress:
        write_fractal_stress(case_dir)
    return "caseroot/scenario0"


_STRIP_IMPORT_RE = re.compile(
    r"^(import netCDF4 as nc|from scipy\.interpolate import griddata)\s*$",
    re.MULTILINE,
)


def load_prepare_module(script_path: Path, module_name: str):
    """Execs the function/class definitions of a scripts/utils/prepare*.py script,
    stopping before its trailing `if case == '...':` dataset-export blocks
    (which do heavy I/O against external, not-checked-out datasets and are
    out of scope for a CPU/CI-fast guard test).

    Also strips two dead-for-this-code-path imports so the test doesn't
    require installing netCDF4 (imported at module scope in both scripts
    but never referenced anywhere -- a no-op import) or scipy (only used by
    genMapsForEQDYNA.plotMap(), never called by create_train_data()); CI's
    pinned dependency list (.github/workflows/tests.yml) has neither, and
    this guard must not add one. This execs the real file's own source
    text -- it is not a reimplementation of any logic under test.
    """
    text = script_path.read_text()
    text = _STRIP_IMPORT_RE.sub(
        "# (import stripped for test-only import; see load_prepare_module)", text)
    marker = re.search(r"^if case\s*==", text, flags=re.MULTILINE)
    assert marker, f"expected a top-level 'if case ==' dispatch block in {script_path}"
    preamble = text[: marker.start()]
    module = types.ModuleType(module_name)
    module.__file__ = str(script_path)
    exec(compile(preamble, str(script_path), "exec"), module.__dict__)
    return module


def extract_split_block(script_path: Path, start_marker: str, end_marker: str) -> str:
    """Returns the literal source text between (and including) start_marker
    and (excluding) end_marker. Used to run just the seed/shuffle/slice
    train-valid-test split logic from an `if case == '...':` block whose
    surrounding loop depends on real, not-checked-out scenario datasets."""
    text = script_path.read_text()
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def run_split_block(script_path: Path, start_marker: str, end_marker: str, case_value: str):
    """Runs extract_split_block()'s source text and returns the namespace it
    produced (so callers can read train_set_idx/valid_set_idx/test_set_idx)."""
    block = extract_split_block(script_path, start_marker, end_marker)
    namespace = {"random": __import__("random"), "case": case_value}
    code_obj = compile(block, str(script_path), "exec")
    eval(code_obj, namespace)
    return namespace
