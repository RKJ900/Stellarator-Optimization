# =========================================================================
# 1. SETUP AND IMPORTS
# =========================================================================
# Ensure you have the latest versions of desc and simsopt installed:
# pip install --upgrade desc-opt simsopt

import numpy as np
import os
from desc.grid import QuadratureGrid
from desc.equilibrium import Equilibrium
from desc.objectives import (
    ObjectiveFunction,

    ForceBalance,
    RadialForceBalance,
    HelicalForceBalance,
    FixBoundaryR,
    FixBoundaryZ,
    FixPressure,
    FixIota,
    FixPsi,
)
from simsopt.geo import (
    SurfaceRZFourier,
    CurveRZFourier,
    curves_to_vtk,
    create_equally_spaced_curves,
)
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_serial_solve as least_squares_simsopt_solve
from simsopt.field import BiotSavart, Current, Coil
from simsopt.objectives import SquaredFlux

# Suppress verbose DESC output for cleaner execution
os.environ["DESC_QUIET"] = "true"


# =========================================================================
# 2. STAGE 1: PLASMA BOUNDARY (MHD EQUILIBRIUM) OPTIMIZATION
# =========================================================================
# In this stage, we optimize the shape of the plasma boundary to achieve
# desired physics properties like good confinement (quasi-symmetry) and
# stability, while respecting constraints on aspect ratio and volume.

print("--- STAGE 1: OPTIMIZING PLASMA BOUNDARY ---")

# -------------------------------------------------------------------------
# 2a. Initial Equilibrium Configuration
# -------------------------------------------------------------------------
# We start with an initial guess for the equilibrium. This could be a
# known configuration like W7-X or a simple tokamak.
# Here, we initialize a basic configuration.

# Define computational grid resolution
# L: radial, M: poloidal, N: toroidal
L, M, N = 24, 24, 24
grid = QuadratureGrid(L=L, M=M, N=N)

# Initialize a base equilibrium. We'll let the boundary shape be free.
equil = Equilibrium(
    M=M,
    N=N,
    pressure=np.array([[0, 1e5], [2, -2e5], [4, 1e5]]), # Peaked pressure profile
    iota=np.array([[0, 0.87], [2, 0.23]]), # Target rotational transform
    Psi=1.0,  # Total toroidal flux of 1 Weber
)
equil.solve(verbose=0) # Initial solve

# -------------------------------------------------------------------------
# 2b. Define Physics Objectives for the Plasma
# -------------------------------------------------------------------------
# This is the core of the physics design. We define what makes a "good" plasma.

# For a commercial reactor, excellent alpha particle confinement is non-negotiable.
# We target this by optimizing for quasi-helical symmetry (QH).
# The objective is to make the magnetic field strength |B| constant on flux surfaces
# in Boozer coordinates. We use a least-squares objective to minimize the
# non-symmetric Fourier modes of |B|.

from desc.objectives import QuasisymmetryBoozer

# The 'grid' defines where we evaluate the objective. A denser grid is more accurate.
qs_objective = QuasisymmetryBoozer(
    helicity=(1, -equil.NFP), # (M, N) for QH symmetry
    grid=grid,
    normalize=True,
)

# Other important physics and engineering objectives:
from desc.objectives import AspectRatio, PlasmaVolume

# Define the complete objective function for DESC
# We combine the force balance equations (the equilibrium itself) with our physics targets.
# The weights determine the relative importance of each objective.
# ForceBalance must be included as it defines the equilibrium state.
objective_stage1 = ObjectiveFunction(
    (
        ForceBalance(weight=1e2),      # Crucial: ensures MHD equilibrium
        qs_objective,                  # Main physics goal: quasi-symmetry
        AspectRatio(target=7.5, weight=1e-1), # Constraint on machine size
        PlasmaVolume(target=300, weight=1e-3), # Ensure sufficient fusion volume
    )
)

# -------------------------------------------------------------------------
# 2c. Define Free Variables and Run Optimization
# -------------------------------------------------------------------------
# We tell DESC what parameters it is allowed to change. Here, we let the
# boundary shape vary freely to find the optimal configuration.

# We also free the pressure and iota profiles to let the optimizer
# find the most stable configuration.
from desc.objectives import FixBoundaryR, FixBoundaryZ, FixPressure, FixIota

constraints = (
    FixBoundaryR(modes=np.array([0, 0, 0])), # Fix major radius
    FixBoundaryZ(modes=np.array([0, 0, 0])), # Fix vertical position
    FixPressure(),
    FixIota(),
    FixPsi()
)

# Set the boundary coefficients (R_mn, Z_mn) as free variables
equil.unlock() # Unlock all parameters
# We keep toroidal field period (NFP) and major radius fixed
equil.set_free("R_lmn", l=1) # Free boundary shape R
equil.set_free("Z_lmn", l=1) # Free boundary shape Z

print("Initial objective value:", objective_stage1.compute(equil.params_dict))

# Run the optimization
# 'lbfgs' is a good gradient-based optimizer for this type of problem.
equil.solve(
    objective=objective_stage1,
    constraints=constraints,
    optimizer="lbfgs",
    maxiter=100, # A full run would use more iterations
    verbose=2,
    ftol=1e-4, # Function tolerance
)

print("Final objective value:", objective_stage1.compute(equil.params_dict))
print("Optimization of plasma boundary complete.")

# We save the optimized boundary shape for the next stage
optimized_surface = equil.get_surface_at(rho=1)


# =========================================================================
# 3. STAGE 2: ELECTROMAGNETIC COIL DESIGN
# =========================================================================
# Now that we have a desirable plasma boundary, we design a set of realistic
# coils that can generate this magnetic field configuration.

print("\n--- STAGE 2: DESIGNING COILS ---")

# -------------------------------------------------------------------------
# 3a. Initialize Coils and Target Magnetic Field
# -------------------------------------------------------------------------
# We define a set of initial coil shapes. A good starting point is to place
# them on a toroidal surface surrounding the plasma.
n_coils = 4
n_fourier = 12 # Fourier modes to describe each coil's shape
toroidal_distance = 1.5 # Distance from plasma to the coil winding surface

# Create an initial set of equally spaced modular coils
base_curves = create_equally_spaced_curves(
    n_coils,
    equil.NFP,
    stellsym=True,
    R0=equil.R_lmn[0, 0, 0],
    R1=toroidal_distance,
    order=n_fourier
)
base_currents = [Current(1.0e5) for i in range(n_coils)]
base_currents[0].fix_all() # Fix one current to set the magnetic field scale

coils = [Coil(curve, current) for curve, current in zip(base_curves, base_currents)]
bs = BiotSavart(coils)

# -------------------------------------------------------------------------
# 3b. Define Coil Engineering Objectives
# -------------------------------------------------------------------------
# We need to ensure the coils are buildable and produce the correct field.
# The primary objective is to minimize the magnetic field normal to the
# plasma surface, effectively recreating the target boundary shape.
# B_normal = B_coils ⋅ n_surface ≈ 0

# Define the target plasma surface from the DESC optimization
s = SurfaceRZFourier.from_desc_surface(optimized_surface, nphi=64, ntheta=64)
bs.set_points(s.gamma().reshape((-1, 3))) # Tell Biot-Savart where to calculate B

# 1. Magnetic field matching objective
flux_objective = SquaredFlux(s, bs)

# 2. Coil engineering and physics objectives
from simsopt.geo import (
    CurveLength,
    CurveCurvature,
    MinimumDistance,
    MeanSquaredCurvature,
)

# Total length of all coils (penalize long coils)
J_coil_length = CurveLength(base_curves)

# Penalize high curvature to prevent sharp bends
J_coil_curvature = CurveCurvature(base_curves, order=2, threshold=5)

# Ensure coils don't collide with each other
J_coil_distance = MinimumDistance(base_curves, min_dist=0.2)

# Combine all objectives into a least-squares problem for SIMSOPT
problem = LeastSquaresProblem.from_tuples(
    [
        (flux_objective.J, 0, 1.0),            # Target B_normal = 0, weight 1.0
        (J_coil_length.J, 0, 1e-5),            # Penalize total length
        (J_coil_curvature.J, 0, 1e-4),         # Penalize high curvature
        (J_coil_distance.J, 0, 1e1),           # Enforce minimum distance
    ]
)

# -------------------------------------------------------------------------
# 3c. Run Coil Optimization
# -------------------------------------------------------------------------
# This step adjusts the shape (Fourier coefficients) of each coil to
# minimize the combined objective function.

least_squares_simsopt_solve(
    problem,
    max_nfev=50, # A full run would use more iterations
    ftol=1e-3
)

print("Coil optimization complete.")
print(f"Final B_normal (flux) objective: {flux_objective.J():.4e}")
print(f"Final total coil length: {J_coil_length.J():.2f} m")

# =========================================================================
# 4. VISUALIZATION AND ANALYSIS
# =========================================================================
# Finally, visualize the result to verify the design.

print("\n--- VISUALIZING FINAL CONFIGURATION ---")

# Save the coils in VTK format for viewing in Paraview or other software
curves_to_vtk(coils, "optimized_coils")

# Use DESC's plotting functions to analyze the final equilibrium
# produced by the optimized coils. (This would involve a final DESC solve
# using the coil field as a boundary condition).
fig, ax = equil.plot_boozer_surface(b_resolution=60)
fig.suptitle("Boozer Spectrum of Optimized Equilibrium")
fig.show()

fig, ax = equil.plot_boundaries()
ax.set_title("Optimized Plasma and Coil Cross-Sections")
fig.show()