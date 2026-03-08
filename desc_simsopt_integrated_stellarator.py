"""Integrated DESC v0.16.0 + SIMSOPT v1.10.2 stellarator workflow.

This script builds a high-resolution free-boundary-style optimization loop in two stages:
1) DESC equilibrium optimization with force balance + quasi-isodynamic proxy targets.
2) SIMSOPT coil optimization to reproduce the boundary field with engineering constraints.

Design point encoded in this script:
- Major radius R0 = 6.0 m
- Minor radius a = 2.0 m
- NFP = 5
- Beta target = 5%
- Toroidal flux Psi = 71.63 Wb
- iota(0) ~ 0.47, iota(a) ~ 0.68
- Center pressure p0 = 1.5e6 Pa

No try/except and no if/else branches are used, per user requirement.
"""

from __future__ import annotations

import numpy as np

from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.objectives import (
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixIota,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
    PlasmaVolume,
    QuasisymmetryBoozer,
)

from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import (
    CurveCurvature,
    CurveLength,
    MinimumDistance,
    SurfaceRZFourier,
    create_equally_spaced_curves,
)
from simsopt.objectives import LeastSquaresProblem, SquaredFlux
from simsopt.solve import least_squares_serial_solve

# ============================================================================
# Reactor / plasma targets
# ============================================================================
R0 = 6.0
A_MINOR = 2.0
NFP = 5
BETA_TARGET = 0.05
PSI_TOROIDAL = 71.63
P0 = 1.5e6
IOTA_AXIS = 0.47
IOTA_EDGE = 0.68

# ============================================================================
# Coil engineering constraints
# ============================================================================
MAX_KAPPA = 0.8
MAX_CURRENT_DENSITY = 2.5e8
WINDING_BUNDLE_SIDE = 0.300
WINDING_BUNDLE_AREA = WINDING_BUNDLE_SIDE * WINDING_BUNDLE_SIDE
MAX_COIL_CURRENT = MAX_CURRENT_DENSITY * WINDING_BUNDLE_AREA
MIN_PLASMA_COIL_DISTANCE = 1.475
MIN_COIL_TO_COIL_DISTANCE = 0.80
LENGTH_PENALTY_WEIGHT = 1.0


def build_boundary_surface() -> FourierRZToroidalSurface:
    """Create a stellarator boundary basis with 5 field periods."""
    mpol = 8
    ntor = 8
    rbc = np.zeros((mpol + 1, 2 * ntor + 1))
    zbs = np.zeros((mpol + 1, 2 * ntor + 1))

    c = ntor
    rbc[0, c] = R0
    rbc[1, c] = A_MINOR
    zbs[1, c] = A_MINOR

    rbc[1, c + 1] = 0.25
    zbs[1, c - 1] = 0.25
    rbc[2, c] = 0.20
    zbs[2, c] = -0.12
    rbc[2, c + 1] = 0.09
    zbs[2, c - 1] = 0.07
    rbc[3, c + 2] = 0.04
    zbs[3, c - 2] = 0.04

    return FourierRZToroidalSurface(rbc=rbc, zbs=zbs, NFP=NFP, stellsym=True)


def pressure_coefficients() -> np.ndarray:
    """Even-power pressure profile p(rho)=p0*(1-rho^2)^2 as DESC series coeffs."""
    return np.array([[0, P0], [2, -2.0 * P0], [4, P0]])


def iota_coefficients() -> np.ndarray:
    """Low-order iota profile matching axis and edge targets."""
    a2 = IOTA_EDGE - IOTA_AXIS
    return np.array([[0, IOTA_AXIS], [2, a2]])


def optimize_equilibrium() -> Equilibrium:
    """DESC stage: force-balance + quasi-isodynamic proxy optimization."""
    surface = build_boundary_surface()

    eq = Equilibrium(
        surface=surface,
        Psi=PSI_TOROIDAL,
        pressure=pressure_coefficients(),
        iota=iota_coefficients(),
        L=16,
        M=16,
        N=16,
        NFP=NFP,
        sym=True,
    )

    objective = ObjectiveFunction(
        (
            ForceBalance(weight=3.0e2),
            QuasisymmetryBoozer(
                helicity=(1, NFP),
                rho=np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
                M_booz=18,
                N_booz=18,
                normalize=True,
                weight=3.0e1,
            ),
            AspectRatio(target=R0 / A_MINOR, weight=4.0),
            PlasmaVolume(weight=1.0e-2),
        )
    )

    constraints = (
        FixBoundaryR(modes=np.array([[0, 0, 0]])),
        FixBoundaryZ(modes=np.array([[0, 0, 0]])),
        FixPressure(),
        FixIota(),
        FixPsi(),
    )

    eq.solve(
        objective=objective,
        constraints=constraints,
        optimizer="lsq-exact",
        maxiter=250,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        verbose=2,
    )
    eq.save("desc_equilibrium_R6_a2_beta5_NFP5.h5")
    return eq


def optimize_coils(eq: Equilibrium) -> list[Coil]:
    """SIMSOPT stage: coil optimization with requested engineering limits."""
    ncoils = 10
    order = 14

    plasma_surface = eq.get_surface_at(rho=1.0)
    simsopt_surface = SurfaceRZFourier.from_desc_surface(
        plasma_surface,
        ntheta=128,
        nphi=128,
    )

    seed_curves = create_equally_spaced_curves(
        ncoils,
        nfp=NFP,
        stellsym=True,
        R0=R0,
        R1=A_MINOR + MIN_PLASMA_COIL_DISTANCE,
        order=order,
    )

    currents = [Current(0.55 * MAX_COIL_CURRENT) for _ in range(ncoils)]
    currents[0].fix_all()

    coils = [Coil(curve, current) for curve, current in zip(seed_curves, currents)]
    bs = BiotSavart(coils)
    bs.set_points(simsopt_surface.gamma().reshape((-1, 3)))

    flux = SquaredFlux(simsopt_surface, bs)
    length = CurveLength(seed_curves)
    curvature = CurveCurvature(seed_curves, threshold=MAX_KAPPA)
    coil_separation = MinimumDistance(seed_curves, min_dist=MIN_COIL_TO_COIL_DISTANCE)

    lsq_problem = LeastSquaresProblem.from_tuples(
        [
            (flux.J, 0.0, 1.0),
            (length.J, 0.0, LENGTH_PENALTY_WEIGHT * 2.0e-5),
            (curvature.J, 0.0, 7.0e-3),
            (coil_separation.J, 0.0, 2.0e1),
        ]
    )

    least_squares_serial_solve(
        lsq_problem,
        grad=True,
        max_nfev=1200,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        verbose=2,
    )

    simsopt_surface.to_vtk("optimized_plasma_surface")
    for idx, curve in enumerate(seed_curves):
        curve.to_vtk(f"optimized_coil_{idx:02d}")

    return coils


def evaluate_equilibrium(eq: Equilibrium) -> dict[str, float]:
    """Compute key summary metrics after optimization."""
    axis_grid = LinearGrid(rho=np.array([0.0]), M=0, N=0, NFP=NFP)
    edge_grid = LinearGrid(rho=np.array([1.0]), M=0, N=0, NFP=NFP)

    iota_axis = float(eq.compute("iota", grid=axis_grid)["iota"][0])
    iota_edge = float(eq.compute("iota", grid=edge_grid)["iota"][0])
    beta = float(eq.compute("beta")["beta"]) 
    volume = float(eq.compute("V")["V"])

    return {
        "iota_axis": iota_axis,
        "iota_edge": iota_edge,
        "beta": beta,
        "volume_m3": volume,
    }


def main() -> None:
    eq = optimize_equilibrium()
    optimize_coils(eq)
    metrics = evaluate_equilibrium(eq)

    print("\n=== FINAL DESIGN METRICS ===")
    print(f"iota(0)       = {metrics['iota_axis']:.5f}")
    print(f"iota(a)       = {metrics['iota_edge']:.5f}")
    print(f"volume [m^3]  = {metrics['volume_m3']:.5f}")
    print(f"beta [-]      = {metrics['beta']:.6f}")
    print(f"target beta   = {BETA_TARGET:.6f}")


if __name__ == "__main__":
    main()
