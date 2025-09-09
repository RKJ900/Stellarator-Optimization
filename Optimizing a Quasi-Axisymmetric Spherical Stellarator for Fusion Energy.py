# -*- coding: utf-8 -*-
"""
================================================================================
Optimizing a Quasi-Axisymmetric Spherical Stellarator for Fusion Energy
================================================================================
This script provides a comprehensive workflow for the design and optimization
of a compact stellarator configuration for commercial fusion energy. It
integrates DESC for 3D MHD equilibrium and SIMSOPT for coil design, stability,
and transport optimization.

Key Optimization Goals:
1.  **MHD Equilibrium**: Achieve a force-balance solution for the plasma.
2.  **Quasi-Symmetry (QA)**: Minimize neoclassical transport for ignition.
3.  **Coil Realizability**: Ensure coils are buildable (length, curvature, spacing).
4.  **Stability & Control**: Target a favorable rotational transform (iota)
    profile to avoid major instabilities and include a magnetic well for
    interchange stability.
5.  **High-Beta**: Design for a high plasma pressure-to-magnetic pressure ratio,
    essential for economic fusion power.
"""

import numpy as np
import os
from simsopt.geo import (
    SurfaceRZFourier, CurveXYZFourier, curves_to_vtk, create_equally_spaced_curves,
    CurveLength, CurveCurveDistance, MinimumDistance
)
from simsopt.field import BiotSavart, Current, Coil
from simsopt.mhd import Vmec, Quasisymmetry
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util.mpi import MpiPartition
from desc.objectives import (
    ForceBalance, RadialForceBalance, HelicalForceBalance,
    ObjectiveFunction, CurrentDensity, RotationalTransform, MagneticWell
)
from desc.grid import QuadratureGrid
from desc.equilibrium import Equilibrium

# ==============================================================================
# 1. Configuration and Initial Parameters
# ==============================================================================
# MPI setup for parallel processing
mpi = MpiPartition()
# Here, we define the basic parameters for our compact, ignition-prone stellarator.
# A low aspect ratio is key for a "spherical" or "compact" stellarator.
# NFP = Number of Field Periods
NFP = 2
# Aspect Ratio ~ R/a (Major Radius / Minor Radius)
ASPECT_RATIO_TARGET = 3.5
# Target average beta (proxy for fusion power output)
BETA_TARGET = 0.05  # 5% is a robust target for ignition scenarios

# Initial guess for the plasma boundary.
# A simple torus is a good starting point. The optimizer will find the complex shape.
R_MAJOR = 1.7  # meters
A_MINOR = R_MAJOR / ASPECT_RATIO_TARGET

# Discretization for equilibrium calculations (higher is more accurate but slower)
L_GRID = 12  # Poloidal grid resolution
M_GRID = 12  # Toroidal grid resolution
N_GRID = 0   # Radial grid resolution (for surface-only calculations initially)

# ==============================================================================
# 2. DESC Equilibrium Setup
# ==============================================================================
# We start with an initial guess for the plasma equilibrium.
# This equilibrium will be optimized simultaneously with the coils.
eq = Equilibrium.from_shape(
    R_major=R_MAJOR,
    a_minor=A_MINOR,
    NFP=NFP,
    pressure=np.array([[0, 2 * BETA_TARGET], [2, -2 * BETA_TARGET]]), # Pressure profile p(psi) = 2*beta*(1-psi)
    iota=np.array([[0, 0.5]]), # Initial guess for rotational transform
    spectral_indexing='ansi',
    L=L_GRID, M=M_GRID, N=N_GRID
)
eq.solve(verbose=2) # Get an initial solution

# ==============================================================================
# 3. SIMSOPT Coil and Magnetic Field Setup
# ==============================================================================
# Define the coils that will generate the magnetic field to confine the plasma.
# We start with simple circular coils and let the optimizer find the complex shape.
N_COILS = 4 # Number of unique coils per half-period
COIL_RADIUS = 1.3 * A_MINOR + 0.5 # Initial guess for coil radius
FOURIER_ORDER = 8 # Degrees of freedom for the coil shape

# Create a set of equally spaced coils around a torus.
base_curves = create_equally_spaced_curves(
    N_COILS,
    NFP,
    stellsym=True,
    R0=R_MAJOR,
    R1=COIL_RADIUS,
    order=FOURIER_ORDER
)
base_currents = [Current(1.0) * 1e5 for _ in range(N_COILS)]
base_currents[0].fix() # Fix one current to set the magnetic field scale

# Create the Biot-Savart solver from the coils
coils = [Coil(cur, curr) for cur, curr in zip(base_curves, base_currents)]
bs = BiotSavart(coils)

# ==============================================================================
# 4. Defining the Multi-Objective Optimization Function
# ==============================================================================
# This is the core of the design process. We define all the physics and
# engineering goals that our final design must satisfy.

# --- Objective 1: MHD Equilibrium (using DESC) ---
# This ensures the plasma is in a stable force-balance state.
# We use a high-resolution grid for the objective to ensure accuracy.
grid = QuadratureGrid(L=eq.L*2, M=eq.M*2, N=eq.N*2, NFP=eq.NFP)
eq_obj = ForceBalance(eq=eq, grid=grid, weight=1e2)

# --- Objective 2: Quasi-Symmetry (Neoclassical Confinement) ---
# This is the most critical objective for a high-performance stellarator.
# We aim to make the magnetic field strength symmetric in a helical direction,
# which dramatically improves plasma confinement, similar to a tokamak.
qs_obj = Quasisymmetry(
    bs,
    eq.get_surface_at(1.0), # Optimize on the plasma boundary
    helicity_m=1, helicity_n=0 # (m=1, n=0) for quasi-axisymmetry
)

# --- Objective 3: Rotational Transform (Stability Proxy) ---
# We want to keep the iota profile away from rational numbers where instabilities grow.
# We target a flat iota profile, slightly reversed shear for stability.
iota_target = -0.45
iota_obj = RotationalTransform(eq=eq, target=iota_target, weight=1e2)

# --- Objective 4: Magnetic Well (Stability Proxy) ---
# A magnetic well (vacuum field strength increasing outwards) helps suppress
# interchange instabilities. We target a 1% well.
well_obj = MagneticWell(eq=eq, target=-0.01, weight=1e2)

# --- Objective 5: Coil Engineering Constraints ---
# Coils must be buildable. We penalize excessive length and curvature.
# We also enforce a minimum distance between coils to allow for structure.
COIL_LENGTH_TARGET = 7.0
MAX_CURVATURE = 5.0
MIN_COIL_DISTANCE = 0.20

J_coil_length = [CurveLength(c, COIL_LENGTH_TARGET) for c in base_curves]
J_coil_curvature = [CurveCurveDistance(c, MAX_CURVATURE) for c in base_curves]
J_coil_distance = MinimumDistance(coils, MIN_COIL_DISTANCE)

# --- Objective 6: Plasma Aspect Ratio ---
# We explicitly target a compact configuration.
J_aspect = AspectRatio(eq, ASPECT_RATIO_TARGET)

# --- Combine all objectives into a single problem ---
# The weights determine the relative importance of each goal.
# Tuning these weights is a key part of the design process.
problem = LeastSquaresProblem.from_tuples(
    [
        (eq_obj, 0, 1),
        (qs_obj, 0, 1e4), # High weight for good confinement
        (iota_obj, 0, 1),
        (well_obj, 0, 1e2),
        (J_aspect, 0, 1),
        (J_coil_length, 0, 1e-2),
        (J_coil_curvature, 0, 1e-3),
        (J_coil_distance, 0, 1e1)
    ]
)

# ==============================================================================
# 5. Defining Degrees of Freedom and Solving
# ==============================================================================
# We define which parameters the optimizer is allowed to change.
# Here, we are optimizing both the plasma shape and the coil shapes.

# Degrees of Freedom (DOFs):
# - Plasma boundary shape (R_mn, Z_mn Fourier coefficients)
# - Coil shapes (X_mn, Y_mn, Z_mn Fourier coefficients)
# - Coil currents (except the one we fixed)
# - Pressure profile parameters

# Free up the plasma boundary shape for optimization
eq.fix_boundary = False
# Free up the pressure profile to allow beta to be optimized
eq.pressure.unfix_all()
# Free up all the coil shape parameters
for curve in base_curves:
    curve.unfix_all()
# Free up all currents except the first one
for current in base_currents[1:]:
    current.unfix_all()

# Link the Biot-Savart field in the DESC equilibrium to the SIMSOPT coils
eq.B_ext = bs.B

# Perform the optimization
# This is a computationally intensive step that requires an MPI environment.
# The 'max_nfev' parameter limits the number of iterations. A real run
# would use a much larger number or a more advanced convergence criterion.
least_squares_mpi_solve(
    problem,
    mpi=mpi,
    grad=True,
    max_nfev=100 # Increase for a production run
)

# ==============================================================================
# 6. Post-Processing and Analysis
# ==============================================================================
if mpi.proc0_world:
    print("==================================================")
    print("          Optimization Complete                 ")
    print("==================================================")
    # Print final values of the objectives
    problem.objective()

    # Save results
    output_dir = "stellarator_design_results/"
    os.makedirs(output_dir, exist_ok=True)
    eq.save(os.path.join(output_dir, "optimized_equilibrium.h5"))
    curves_to_vtk(base_curves, os.path.join(output_dir, "optimized_coils"))

    # Further analysis would go here:
    # - Run stability analysis codes (e.g., BOOZ_XFORM, GATO, CAS3D).
    # - Perform neoclassical transport calculations (e.g., SFINCS).
    # - Perform gyrokinetic analysis for turbulence (e.g., GENE, GKV).
    # - Generate engineering drawings from coil data.

    print(f"\n✅ Results saved in '{output_dir}'")
    print("Final equilibrium and coil shapes can be visualized using Paraview.")