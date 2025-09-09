#!/usr/bin/env python3
"""
DESC + SIMSOPT integrated workflow for compact (spherical) stellarator:
 - setup a low-aspect-ratio boundary
 - build an initial DESC equilibrium and solve
 - define multi-objective optimization (QS triple product, EffectiveRipple, Ballooning)
 - run DESC optimization (fixed-boundary, gradient-based)
 - create filamentary coils in simsopt (via symmetry), define coil objectives
 - optimize coils (squared flux + regularization)
 - save outputs for post-processing (h5/json)


"""
import os
import numpy as np
import logging

# DESC imports (equilibrium, geometry, profiles, objectives, optimizer)
from desc.geometry import FourierRZToroidalSurface
from desc.equilibrium import Equilibrium
from desc.profiles import MTanhProfile, PowerSeriesProfile
from desc.objectives import (
    QuasisymmetryTripleProduct,
    EffectiveRipple,
    BallooningStability,
    ObjectiveFunction
)
from desc.optimize import Optimizer

# SIMSOPT imports (coils, fields, curves, objectives)
from simsopt.geo import create_equally_spaced_curves, CurveLength, MeanSquaredCurvature, CurveSurfaceDistance, CurveCurveDistance
from simsopt.field import BiotSavart, Current
from simsopt.objectives import (
    SquaredFlux,
    QuadraticPenalty,

)

from scipy.optimize import minimize

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("desc_simsopt_integration")

# -------------------------
# User / physical settings
# -------------------------
NFP = 2                      # number of field periods (example)
major_R = 1.5                # meters, compact / spherical-ish
aspect_ratio = 1.5           # low aspect ratio
Psi_total = 1.0              # total toroidal flux (Weber) - scale parameter
target_beta = 0.05           # desired volume-averaged beta (~5%) -- tune carefully
ntheta = 64                  # real-space poloidal resolution
nzeta = 64                   # real-space toroidal resolution

# DESC spectral resolution (start modest, raise for production)
M = 6    # poloidal spectral resolution
N = 6    # toroidal spectral resolution
L = 4    # radial resolution (Zernike radial order)

# Optimization weights (tune strongly for your problem)
W_QS = 1.0e-3
W_RIPPLE = 5.0
W_BALLOON = 20.0

# Coil design parameters (stage-two)
n_base_coils = 4     # number of unique base coils (will be replicated by symmetries)
init_coil_R0 = 1.0   # starting circle major location (m)
init_coil_R1 = 0.5   # starting circle minor radius (m)
coil_order = 6       # Fourier order for coil curves
LENGTH_WEIGHT = 1.0
CURV_WEIGHT = 1e-2
SURF_DIST_TARGET = 0.35  # target min coil-plasma distance (m)
COIL_MIN_DIST = 0.05     # min coil-to-coil distance enforced via penalty
COIL_ITER_MAX = 300

# output folder
OUTDIR = "desc_simsopt_output"
os.makedirs(OUTDIR, exist_ok=True)


# -------------------------
# Helper: Build DESC equilibrium
# -------------------------
def build_and_solve_desc_equilibrium():
    log.info("Building initial boundary surface (FourierRZToroidalSurface.from_shape_parameters)...")
    # use the convenient shape-parameter generator (generalized Miller-like)
    surface = FourierRZToroidalSurface.from_shape_parameters(
        major_radius=major_R,
        aspect_ratio=aspect_ratio,
        elongation=1.0,
        triangularity=0.0,
        squareness=0.0,
        NFP=NFP,
        sym=True
    )

    log.info("Creating Equilibrium object with initial profiles...")
    # Pressure profile: MTanhProfile gives pedestal-like shapes; for core optimization we set a
    # simple monotonic profile scaled to target_beta (user should calibrate units)
    # NOTE: DESC uses SI units for pressure (Pa). Here we supply a placeholder profile.
    p_profile = PowerSeriesProfile(params=[0.0, 0.0])  # start with small pressure, we'll ramp later

    # Rotational transform profile (iota) -- choose a modest vacuum profile if unknown
    iota_profile = PowerSeriesProfile(params=[0.0, 0.5])  # iota(0) ~ 0.5

    eq = Equilibrium(
        Psi=Psi_total,
        NFP=NFP,
        L=L, M=M, N=N,
        L_grid=L, M_grid=ntheta, N_grid=nzeta,
        pressure=p_profile,
        iota=iota_profile,
        surface=surface,
        ensure_nested=True
    )

    log.info("Solving initial equilibrium (eq.solve()) ...")
    # Initial solve: may require tuning of solver tolerances in real problems
    eq.solve()

    log.info("Initial equilibrium solved. Saving baseline equilibrium.")
    eq.save(os.path.join(OUTDIR, "eq_initial.h5"))

    return eq


# -------------------------
# Helper: DESC optimization (equilibrium-level)
# -------------------------
def optimize_equilibrium(eq: Equilibrium):
    """
    Build a composite objective in DESC and optimize the equilibrium:
     - Quasisymmetry triple product (volume metric)
     - Effective ripple (neoclassical proxy)
     - Ballooning stability (infinite-n ideal ballooning)
    """

    log.info("Building DESC objectives (Quasisymmetry, EffectiveRipple, Ballooning)...")

    # Quasisymmetry triple product objective (volume objective)
    qs = QuasisymmetryTripleProduct(eq, weight=W_QS)

    # Effective ripple (neoclassical proxy) - the tutorial suggests aiming for epsilon < ~0.02
    eff_rip = EffectiveRipple(eq, weight=W_RIPPLE, X=16, Y=32)

    # Ballooning stability objective - penalize unstable growth rates (lambda>lambda0)
    balloon = BallooningStability(eq, weight=W_BALLOON, rho=np.array([0.5]),
                                  alpha=np.linspace(0.0, np.pi, 8, endpoint=False),
                                  nturns=3, nzetaperturn=200, lambda0=0.0)

    # Combine objectives into single ObjectiveFunction (DESC wrapper)
    obj = ObjectiveFunction([qs, eff_rip, balloon])

    # Use DESC's Optimizer wrapper (lsq trust region or fmintr). We use lsqtr for robust least-squares.
    opt = Optimizer("lsq-exact") # Changed from "lsqtr" to "lsq-exact"

    log.info("Running DESC equilibrium optimization (this can use AD and GPUs if configured)...")
    res = eq.optimize(obj, optimizer=opt, maxiter=200, jac=True)

    log.info("DESC optimization finished. Saving optimized equilibrium.")
    eq.save(os.path.join(OUTDIR, "eq_optimized.h5"))

    # compute and log a few diagnostic scalar values
    try:
        eff_rip_vals = eff_rip.compute_scaled()
        qs_vals = qs.compute_scaled()
        # ballooning returns scalars aggregated according to its design
        balloon_val = balloon.compute_scaled()
        log.info(f"Post-opt diagnostics: mean(epsilon) ~ {np.mean(eff_rip_vals):.4e}, "
                 f"QS norm ~ {np.linalg.norm(qs_vals):.4e}, ballooning {balloon_val:.4e}")
    except Exception as e:
        log.warning(f"Could not compute post-opt diagnostics (best-effort): {e}")

    return eq, res


# -------------------------
# Helper: Create and optimize coils with simsopt
# -------------------------
def design_and_optimize_coils(eq: Equilibrium):
    """
    Use simsopt to build base coil shapes (equally spaced circles -> Fourier curves),
    replicate them using stellarator symmetry, and build a BiotSavart object.
    Optimize coils to reduce B·n on LCFS while satisfying regularization constraints.
    """
    log.info("Creating target surface geometry for simsopt (export from DESC surface) ...")
    # Extract real-space target surface points from DESC surface object (gamma())
    s = eq.surface

    # simsopt expects a Surface-like object; for the example we call create_equally_spaced_curves
    # to generate base coil curves. This uses the surface.nfp metadata in simsopt examples.
    log.info("Creating initial base coils (equally spaced circles) ...")
    base_curves = create_equally_spaced_curves(
        n_base_coils, s.NFP, stellsym=True, R0=init_coil_R0, R1=init_coil_R1, order=coil_order
    )

    # Create currents (per-base-coil). Use Current from simsopt; typical scaling: 1e5 for Tesla-level fields.
    base_currents = [Current(1.0) * 1e5 for _ in range(n_base_coils)]
    # Optionally fix the first current to avoid trivial zero-current solution:
    base_currents[0].fix_all()

    # Replicate base coils across symmetry to obtain full set
    coils = coils_via_symmetries(base_curves, base_currents, s.NFP, True)

    log.info(f"Total coils created: {len(coils)} (with symmetry replication). Building BiotSavart ...")
    bs = BiotSavart(coils)

    # set evaluation points on the DESC surface (flatten gamma())
    gamma = s.gamma()  # (nphi, ntheta, 3) array in simsopt surface API
    pts = gamma.reshape((-1, 3))
    bs.set_points(pts)

    # Define the coil objective: squared flux (B·n) + length penalty + curvature + min distances
    J_flux = SquaredFlux(s, bs)  # main matching term
    J_lengths = sum(CurveLength(c.curve) for c in base_curves)
    J_len_pen = QuadraticPenalty(J_lengths, LENGTH_WEIGHT * J_lengths.J(), "max")

    # curvature and minimum distances (stabilizing/regularizing terms)
    J_curv = sum(MeanSquaredCurvature(c.curve) for c in base_curves)
    J_surf_dist = sum(CurveSurfaceDistance(c.curve, s, target=SURF_DIST_TARGET) for c in base_curves)
    # coil-to-coil min distance penalty - one can build CurveCurveDistance terms, approximate with sum
    J_coil_coil = sum(CurveCurveDistance(c1.curve, c2.curve, target=COIL_MIN_DIST)
                      for i, c1 in enumerate(base_curves) for c2 in base_curves[i + 1:])

    # full objective: weighted sum (weights are examples; tune them)
    objective = J_flux + LENGTH_WEIGHT * J_len_pen + CURV_WEIGHT * J_curv + \
                100.0 * J_surf_dist + 1000.0 * J_coil_coil

    log.info("Preparing simsopt objective vector and calling SciPy optimizer (L-BFGS-B)...")

    # wrapper objective function compatible with scipy.minimize
    def fun(dofs):
        objective.x = dofs
        Jval = objective.J()
        dJ = objective.dJ()
        return float(Jval), np.asarray(dJ, dtype=float)

    x0 = objective.x.copy()
    res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                   options={'maxiter': COIL_ITER_MAX, 'disp': True})

    log.info("Coil optimization complete. Saving BiotSavart JSON + coil shapes.")
    bs.save(os.path.join(OUTDIR, "biot_savart_opt.json"))

    # also save coil curve parameters to numpy for reproducibility
    np.save(os.path.join(OUTDIR, "coils_dofs.npy"), res.x)

    return coils, bs, res


# -------------------------
# Main workflow
# -------------------------
def main():
    eq0 = build_and_solve_desc_equilibrium()

    # optionally ramp pressure to desired beta (this is problem-specific)
    # A proper mapping from target_beta to SI pressure profile must be supplied by user.
    # For demonstration we modestly raise core pressure coefficients and re-solve
    try:
        eq0.pressure.set_params([0.0, target_beta * 1e5])  # placeholder scaling
    except Exception:
        pass

    eq0.solve()
    log.info("Running equilibrium-level optimization in DESC ...")
    eq_opt, desc_res = optimize_equilibrium(eq0)

    log.info("Running coil design + optimization (simsopt) ...")
    coils, bs, coil_res = design_and_optimize_coils(eq_opt)

    log.info("Workflow complete. Results are in: %s", OUTDIR)


if __name__ == "__main__":
    main()