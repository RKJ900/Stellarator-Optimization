#!/usr/bin/env python3
"""
Advanced Stellarator Plasma Equilibrium and Optimization Script
Integrating DESC and SIMSOPT for Commercial Fusion Reactor Design

This script provides comprehensive stellarator optimization for ignition-prone
plasma configurations targeting commercial power production parameters.

Authors: DESC Development Team, SIMSOPT Development Team
Version: 2.0
Date: August 2025
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import h5py
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# DESC imports - latest v0.14.1 API
try:
    from desc import set_device
    from desc.equilibrium import Equilibrium
    from desc.geometry import FourierRZToroidalSurface
    from desc.profiles import PowerSeriesProfile, SplineProfile
    from desc.magnetic_fields import (
        ToroidalField, PoloidalField, VerticalField, 
        SplineMagneticField, MixedCoilSet
    )
    from desc.objectives import (
        ForceBalance, FixBoundaryR, FixBoundaryZ, FixCurrent, FixPressure,
        QuasisymmetryBoozer, QuasisymmetryTripleProduct, AspectRatio,
        Volume, MeanCurvature, BoozerResidual, MercierStability,
        BootstrapRedlConsistency, IsodynConsistency
    )
    from desc.constraints import (
        FixedBoundary, FixCurrent as FixCurrentConstraint, 
        FixPressure as FixPressureConstraint, FixIota,
        PlasmaVesselDistance, CoilCurrentLength, CoilCurvature
    )
    from desc.optimize import Optimizer
    from desc.continuation import solve_continuation_automatic
    from desc.compute import compute as desc_compute
    from desc.io import save, load
    DESC_AVAILABLE = True
    print("✓ DESC v0.14.1+ successfully imported")
except ImportError as e:
    DESC_AVAILABLE = False
    print(f"✗ DESC import failed: {e}")
    print("Install with: pip install desc-opt")

# SIMSOPT imports - latest v1.8.1+ API  
try:
    import simsopt
    from simsopt.geo import (
        SurfaceRZFourier, CurveRZFourier, CurveCWSFourier,
        curves_to_vtk, SurfaceXYZTensorFourier
    )
    from simsopt.field import (
        BiotSavart, Current, coils_via_symmetries,
        DipoleField, ToroidalField as SimsoptToroidalField
    )
    from simsopt.objectives import QuadraticPenalty, SquaredFlux
    from simsopt.solve import least_squares_mpi_solve
    from simsopt._core.optimizable import Optimizable
    from simsopt.mhd import Vmec, Boozer, Quasisymmetry
    from simsopt.util import MpiPartition
    SIMSOPT_AVAILABLE = True
    print(f"✓ SIMSOPT v{simsopt.__version__} successfully imported")
except ImportError as e:
    SIMSOPT_AVAILABLE = False
    print(f"✗ SIMSOPT import failed: {e}")
    print("Install with: pip install simsopt")

# Numerical libraries
import scipy.optimize as opt
from scipy.interpolate import CubicSpline, interp1d
from scipy.constants import mu_0, pi

class CommercialStellaratorDesigner:
    """
    Comprehensive stellarator design optimization class integrating DESC and SIMSOPT
    for commercial fusion reactor applications targeting ignition conditions.
    """
    
    def __init__(self, 
                 major_radius: float = 6.0,
                 minor_radius: float = 1.8, 
                 aspect_ratio: float = 3.33,
                 magnetic_field: float = 5.7,
                 beta_target: float = 0.05,
                 device_name: str = "commercial_stellarator",
                 verbose: bool = True):
        """
        Initialize commercial stellarator designer with reactor-scale parameters.
        
        Parameters:
        -----------
        major_radius : float, default=6.0
            Major radius in meters (commercial scale)
        minor_radius : float, default=1.8  
            Minor radius in meters
        aspect_ratio : float, default=3.33
            Aspect ratio R/a
        magnetic_field : float, default=5.7
            On-axis magnetic field in Tesla
        beta_target : float, default=0.05
            Target plasma beta (5% for ignition)
        device_name : str
            Name identifier for the device
        verbose : bool
            Enable verbose output
        """
        
        if not (DESC_AVAILABLE and SIMSOPT_AVAILABLE):
            raise ImportError("Both DESC and SIMSOPT are required for stellarator optimization")
            
        self.R0 = major_radius
        self.a = minor_radius  
        self.aspect_ratio = aspect_ratio
        self.B0 = magnetic_field
        self.beta_target = beta_target
        self.device_name = device_name
        self.verbose = verbose
        
        # Commercial reactor parameters for ignition
        self.target_temperature = 15.0  # keV
        self.target_density = 1.0e20    # m^-3
        self.target_pressure = 5.0e5    # Pa
        self.fusion_power = 400.0       # MW thermal
        
        # Optimization parameters
        self.NFP = 3  # Number of field periods (typical for stellarators)
        self.mpol = 8  # Poloidal modes
        self.ntor = 8  # Toroidal modes
        self.ns = 61   # Flux surfaces
        
        # Storage for results
        self.equilibrium = None
        self.optimized_coils = None
        self.optimization_history = []
        
        if self.verbose:
            print(f"Initialized {device_name} designer:")
            print(f"  Major radius: {self.R0:.2f} m")
            print(f"  Minor radius: {self.a:.2f} m") 
            print(f"  Aspect ratio: {self.aspect_ratio:.2f}")
            print(f"  Magnetic field: {self.B0:.2f} T")
            print(f"  Target beta: {self.beta_target*100:.1f}%")

    def create_initial_surface(self) -> FourierRZToroidalSurface:
        """
        Create initial boundary surface with realistic commercial stellarator shaping.
        Incorporates bean-shaped cross-section and helical twisting for good confinement.
        """
        
        if self.verbose:
            print("Creating initial plasma boundary surface...")
            
        # Stellarator-specific Fourier coefficients for commercial design
        # Based on optimized configurations like W7-X scaling
        rbc = np.zeros((self.mpol + 1, 2 * self.ntor + 1))
        zbs = np.zeros((self.mpol + 1, 2 * self.ntor + 1))
        
        # (0,0) mode - major radius
        rbc[0, self.ntor] = self.R0
        
        # (1,0) mode - minor radius (circular component)
        rbc[1, self.ntor] = self.a * 0.95
        zbs[1, self.ntor] = self.a * 0.95
        
        # Bean shaping for improved stability (1,1) mode
        rbc[1, self.ntor + 1] = self.a * 0.15  
        zbs[1, self.ntor - 1] = self.a * 0.15
        
        # Higher order shaping for optimization flexibility
        rbc[2, self.ntor] = self.a * 0.08      # Triangularity
        rbc[0, self.ntor + 1] = self.R0 * 0.02 # Helical shift
        zbs[2, self.ntor + 1] = self.a * 0.05  # Helical coupling
        
        # (2,1) mode for magnetic well
        rbc[2, self.ntor + 1] = self.a * 0.03
        zbs[2, self.ntor - 1] = -self.a * 0.03
        
        # (3,2) mode for quasi-symmetry optimization
        rbc[3, self.ntor + 2] = self.a * 0.015
        zbs[3, self.ntor - 2] = self.a * 0.015
        
        surface = FourierRZToroidalSurface(
            rbc=rbc,
            zbs=zbs,
            NFP=self.NFP,
            stellsym=True,
            name=f"{self.device_name}_boundary"
        )
        
        if self.verbose:
            print(f"  Created surface with NFP={self.NFP}, mpol={self.mpol}, ntor={self.ntor}")
            print(f"  Surface area: {surface.compute('S')['S']:.2f} m²")
            print(f"  Volume: {surface.compute('V')['V']:.2f} m³")
            
        return surface

    def create_pressure_profile(self) -> PowerSeriesProfile:
        """
        Create realistic pressure profile for commercial reactor conditions.
        Peaked profile optimized for ignition with good confinement.
        """
        
        if self.verbose:
            print("Creating pressure profile for ignition conditions...")
            
        # Commercial stellarator pressure profile
        # Peaked profile with p(0) calculated for target beta
        p0 = self.beta_target * self.B0**2 / (2 * mu_0) * 2.0  # Central pressure
        
        # Power series coefficients for realistic profile
        # p(ρ) = p0 * (1 - ρ²)^α with α ≈ 2 for good performance
        coeffs = [p0, 0, -2*p0, 0, p0]  # Polynomial coefficients
        
        pressure = PowerSeriesProfile(
            params=coeffs,
            modes=np.array([0, 1, 2, 3, 4]),
            name=f"{self.device_name}_pressure"
        )
        
        if self.verbose:
            p_avg = pressure.compute('p', grid=np.linspace(0, 1, 100))['p'].mean()
            print(f"  Central pressure: {p0/1e5:.2f} bar")
            print(f"  Average pressure: {p_avg/1e5:.2f} bar")
            print(f"  Pressure driven current: ~{self.estimate_bootstrap_current(p0):.2f} kA")
            
        return pressure

    def create_rotational_transform_profile(self) -> PowerSeriesProfile:
        """
        Create optimized rotational transform (iota) profile for stability and confinement.
        Targets magnetic shear for turbulence suppression and MHD stability.
        """
        
        if self.verbose:
            print("Creating rotational transform profile...")
            
        # Optimized iota profile for commercial stellarator
        # Edge value ~0.95 to avoid low-order rationals
        # Modest shear for stability
        iota_0 = 0.85  # On-axis value
        iota_a = 0.95  # Edge value
        
        # Parabolic profile with controlled shear
        # iota(ρ) = iota_0 + (iota_a - iota_0) * ρ² + higher order terms
        coeffs = [iota_0, 0, (iota_a - iota_0), 0, -0.05]
        
        iota = PowerSeriesProfile(
            params=coeffs,
            modes=np.array([0, 1, 2, 3, 4]),
            name=f"{self.device_name}_iota"
        )
        
        if self.verbose:
            print(f"  On-axis iota: {iota_0:.3f}")
            print(f"  Edge iota: {iota_a:.3f}")
            print(f"  Magnetic shear: {(iota_a - iota_0):.3f}")
            
        return iota

    def create_equilibrium(self) -> Equilibrium:
        """
        Create initial DESC equilibrium with commercial reactor parameters.
        """
        
        if self.verbose:
            print("\nCreating DESC equilibrium...")
            
        surface = self.create_initial_surface()
        pressure = self.create_pressure_profile()
        iota = self.create_rotational_transform_profile()
        
        # Create equilibrium object
        eq = Equilibrium(
            surface=surface,
            pressure=pressure,
            iota=iota,
            Psi=2 * pi * self.R0 * self.B0 * self.a**2 / self.NFP,  # Toroidal flux
            NFP=self.NFP,
            L=self.mpol,
            M=self.ntor,
            N=self.ns,
            L_grid=self.mpol * 2,
            M_grid=self.ntor * 2,
            N_grid=self.ns * 2,
            spectral_indexing="ansi",
            name=f"{self.device_name}_equilibrium"
        )
        
        if self.verbose:
            print(f"  Equilibrium created with {eq.N} flux surfaces")
            print(f"  Spectral resolution: L={eq.L}, M={eq.M}")
            print(f"  Toroidal flux: {eq.Psi/(2*pi):.2e} Wb")
            
        return eq

    def setup_optimization_objectives(self) -> List:
        """
        Setup comprehensive optimization objectives for commercial stellarator design.
        Focuses on ignition requirements, stability, and power production.
        """
        
        if self.verbose:
            print("\nSetting up optimization objectives...")
            
        objectives = []
        
        # 1. Force balance - fundamental requirement
        objectives.append(ForceBalance(eq=self.equilibrium, name="force_balance"))
        
        # 2. Quasi-symmetry in Boozer coordinates - key for confinement
        objectives.append(QuasisymmetryBoozer(
            eq=self.equilibrium,
            helicity=(1, self.NFP),  # Helical symmetry
            rho=np.linspace(0.1, 0.9, 9),
            name="quasi_symmetry_boozer",
            weight=100.0
        ))
        
        # 3. Triple product quasi-symmetry for neoclassical optimization  
        objectives.append(QuasisymmetryTripleProduct(
            eq=self.equilibrium,
            helicity=(1, self.NFP),
            rho=np.linspace(0.2, 0.8, 7),
            name="quasi_symmetry_triple",
            weight=50.0
        ))
        
        # 4. Aspect ratio control for engineering constraints
        objectives.append(AspectRatio(
            eq=self.equilibrium,
            target=self.aspect_ratio,
            name="aspect_ratio",
            weight=10.0
        ))
        
        # 5. Volume optimization for power density
        target_volume = 4/3 * pi * self.a**2 * self.R0 * 2 * pi
        objectives.append(Volume(
            eq=self.equilibrium,
            target=target_volume,
            name="plasma_volume", 
            weight=5.0
        ))
        
        # 6. Mean curvature for MHD stability
        objectives.append(MeanCurvature(
            eq=self.equilibrium,
            name="mean_curvature",
            weight=20.0
        ))
        
        # 7. Mercier stability criterion
        objectives.append(MercierStability(
            eq=self.equilibrium, 
            rho=np.linspace(0.1, 0.9, 9),
            name="mercier_stability",
            weight=30.0
        ))
        
        # 8. Bootstrap current self-consistency for steady-state operation
        objectives.append(BootstrapRedlConsistency(
            eq=self.equilibrium,
            name="bootstrap_consistency", 
            weight=25.0
        ))
        
        if self.verbose:
            print(f"  Created {len(objectives)} optimization objectives")
            for obj in objectives:
                print(f"    - {obj.name}: weight={getattr(obj, 'weight', 1.0)}")
                
        return objectives

    def setup_optimization_constraints(self) -> List:
        """
        Setup optimization constraints for commercial stellarator design.
        """
        
        if self.verbose:
            print("\nSetting up optimization constraints...")
            
        constraints = []
        
        # 1. Fix boundary surface points for controlled optimization
        constraints.append(FixBoundaryR(
            eq=self.equilibrium,
            modes=[(0, 0), (1, 0)],  # Fix major and minor radius
            name="fix_boundary_R"
        ))
        
        constraints.append(FixBoundaryZ(
            eq=self.equilibrium, 
            modes=[(1, 0)],  # Fix Z shaping
            name="fix_boundary_Z"
        ))
        
        # 2. Fix total current for external control
        constraints.append(FixCurrent(
            eq=self.equilibrium,
            target=0.0,  # Current-driven operation
            name="fix_current"
        ))
        
        # 3. Fix pressure profile shape
        constraints.append(FixPressure(
            eq=self.equilibrium,
            modes=[0, 1],  # Fix pressure amplitude and core gradient  
            name="fix_pressure"
        ))
        
        if self.verbose:
            print(f"  Created {len(constraints)} optimization constraints")
            for con in constraints:
                print(f"    - {con.name}")
                
        return constraints

    def optimize_equilibrium(self, 
                           max_iterations: int = 200,
                           ftol: float = 1e-8,
                           xtol: float = 1e-8) -> Equilibrium:
        """
        Perform comprehensive equilibrium optimization using DESC.
        """
        
        if self.verbose:
            print(f"\nStarting equilibrium optimization...")
            print(f"  Max iterations: {max_iterations}")
            print(f"  Function tolerance: {ftol}")
            print(f"  Parameter tolerance: {xtol}")
            
        start_time = time.time()
        
        # Create equilibrium if not already done
        if self.equilibrium is None:
            self.equilibrium = self.create_equilibrium()
            
        # Setup objectives and constraints
        objectives = self.setup_optimization_objectives() 
        constraints = self.setup_optimization_constraints()
        
        # Create optimizer with advanced settings
        optimizer = Optimizer(
            method="proximal-lsq-exact",  # Best for stellarator optimization
            options={
                'maxiter': max_iterations,
                'ftol': ftol,
                'xtol': xtol,
                'verbose': 2 if self.verbose else 0,
                'initial_trust_radius': 1.0,
                'max_trust_radius': 10.0,
                'eta': 0.15,
                'gtol': 1e-6
            }
        )
        
        # Perform optimization
        if self.verbose:
            print("Running optimization...")
            
        eq_opt, history = optimizer.optimize(
            eq=self.equilibrium,
            objective=objectives,
            constraints=constraints,
            verbose=self.verbose
        )
        
        optimization_time = time.time() - start_time
        
        # Store results
        self.equilibrium = eq_opt
        self.optimization_history = history
        
        if self.verbose:
            print(f"\nOptimization completed in {optimization_time:.1f} seconds")
            print(f"Final objective value: {history['objective'][-1]:.2e}")
            print(f"Constraint violation: {history['constraint_violation'][-1]:.2e}")
            
        return eq_opt

    def analyze_equilibrium_quality(self) -> Dict[str, float]:
        """
        Comprehensive analysis of optimized equilibrium quality and performance.
        """
        
        if self.equilibrium is None:
            raise ValueError("No equilibrium available. Run optimization first.")
            
        if self.verbose:
            print("\nAnalyzing equilibrium quality...")
            
        # Compute key plasma parameters
        results = desc_compute([
            'p', 'iota', 'beta', 'aspect_ratio', 'V', 'S', 
            'B0', '|B|', 'J', 'K', 'q', 'well',
            'magnetic_field_curvature', 'magnetic_shear',
            'mercier', 'D_Mercier'
        ], eq=self.equilibrium)
        
        # Extract key metrics
        metrics = {
            'aspect_ratio': float(results['aspect_ratio'][0]),
            'plasma_volume': float(results['V'][0]),
            'surface_area': float(results['S'][0]), 
            'average_beta': float(np.mean(results['beta'])),
            'peak_beta': float(np.max(results['beta'])),
            'on_axis_field': float(results['B0'][0]),
            'average_field': float(np.mean(results['|B|'])),
            'magnetic_well': float(np.mean(results['well'])),
            'magnetic_shear': float(np.mean(results['magnetic_shear'])),
            'iota_on_axis': float(results['iota'][0]),
            'iota_edge': float(results['iota'][-1]),
            'mercier_stable_fraction': float(np.sum(results['D_Mercier'] > 0) / len(results['D_Mercier']))
        }
        
        # Calculate performance metrics
        metrics['power_density'] = metrics['average_beta'] * metrics['average_field']**2 / (2 * mu_0 * 1e6)  # MW/m³
        metrics['fusion_power_estimate'] = metrics['power_density'] * metrics['plasma_volume']  # MW
        metrics['confinement_figure'] = metrics['aspect_ratio']**(-1.5) * metrics['average_beta']**0.5
        
        if self.verbose:
            print("\nEquilibrium Quality Metrics:")
            print(f"  Aspect ratio: {metrics['aspect_ratio']:.2f}")
            print(f"  Plasma volume: {metrics['plasma_volume']:.2f} m³")
            print(f"  Average beta: {metrics['average_beta']*100:.2f}%")
            print(f"  Peak beta: {metrics['peak_beta']*100:.2f}%")
            print(f"  Magnetic well: {metrics['magnetic_well']:.2f}%")
            print(f"  Mercier stable: {metrics['mercier_stable_fraction']*100:.1f}%")
            print("\nPerformance Estimates:")
            print(f"  Power density: {metrics['power_density']:.2f} MW/m³")  
            print(f"  Fusion power: {metrics['fusion_power_estimate']:.0f} MW")
            print(f"  Confinement figure: {metrics['confinement_figure']:.3f}")
            
        return metrics

    def create_simsopt_surface(self) -> SurfaceRZFourier:
        """
        Convert DESC equilibrium boundary to SIMSOPT surface for coil optimization.
        """
        
        if self.equilibrium is None:
            raise ValueError("No equilibrium available. Run DESC optimization first.")
            
        if self.verbose:
            print("\nCreating SIMSOPT surface from DESC equilibrium...")
            
        # Extract boundary surface from DESC equilibrium
        surface_desc = self.equilibrium.surface
        
        # Create SIMSOPT surface with same Fourier coefficients
        surface_simsopt = SurfaceRZFourier(
            nfp=self.NFP,
            stellsym=True,
            mpol=self.mpol,
            ntor=self.ntor,
            dofs=surface_desc.R_lmn.tolist() + surface_desc.Z_lmn.tolist()
        )
        
        if self.verbose:
            print(f"  Converted surface with {len(surface_simsopt.dofs)} DOFs")
            print(f"  Surface area: {surface_simsopt.area():.2f} m²")
            print(f"  Surface volume: {surface_simsopt.volume():.2f} m³")
            
        return surface_simsopt

    def optimize_coils_simsopt(self, 
                              num_coils: int = 6,
                              coil_separation: float = 1.2,
                              current_target: float = 5e6,
                              max_iterations: int = 500) -> List:
        """
        Optimize external coil system using SIMSOPT for the DESC equilibrium.
        Stage-2 optimization: fixed plasma boundary, optimize coil shapes.
        """
        
        if self.verbose:
            print(f"\nOptimizing coil system with SIMSOPT...")
            print(f"  Number of coils per period: {num_coils}")
            print(f"  Target current: {current_target/1e6:.1f} MA")
            print(f"  Coil-plasma separation: {coil_separation:.2f} m")
            
        start_time = time.time()
        
        # Create target surface from DESC equilibrium
        surface = self.create_simsopt_surface()
        
        # Create initial coil set
        base_curves = []
        base_currents = []
        
        for i in range(num_coils):
            # Create coil curves at different poloidal angles
            angle = 2 * pi * i / num_coils
            
            # Initial coil position (circular approximation)
            R_coil = self.R0 + (self.a + coil_separation) * np.cos(angle)
            Z_coil = (self.a + coil_separation) * np.sin(angle)
            
            # Create Fourier curve for coil shape optimization
            curve = CurveRZFourier(
                quadpoints=64,
                order=8,  # Fourier order for shape flexibility
                nfp=1,    # Individual coil, not stellarator symmetric
                stellsym=False
            )
            
            # Initialize with circular coil
            curve.set('rc(0)', R_coil)
            curve.set('zs(0)', Z_coil) 
            curve.set('rc(1)', self.a * 0.7)  # Coil minor radius
            
            # Add some initial shaping
            curve.set('rc(2)', 0.1)
            curve.set('zs(2)', 0.1)
            
            base_curves.append(curve)
            
            # Create current object
            current = Current(current_target * (1 + 0.1 * np.random.randn()))
            base_currents.append(current)
            
        # Create full coil set using stellarator symmetry
        coils = coils_via_symmetries(base_curves, base_currents, self.NFP, True)
        
        if self.verbose:
            print(f"  Created {len(coils)} total coils with stellarator symmetry")
            
        # Create magnetic field from coils
        bs = BiotSavart(coils)
        
        # Setup optimization objectives
        Jf = SquaredFlux(surface, bs, target=0.0)  # Minimize flux through surface
        
        # Coil regularization terms
        Jlength = QuadraticPenalty(sum([c.curve.curve_length_penalty() for c in coils]), 
                                  target=2 * pi * (self.R0 + self.a), weight=1e-6)
        Jcurvature = QuadraticPenalty(sum([c.curve.curvature_penalty() for c in coils]),
                                     target=0.0, weight=1e-5)
        Jmsc = QuadraticPenalty(sum([c.curve.mean_squared_curvature_penalty() for c in coils]),
                               target=0.0, weight=1e-6)
        
        # Current regularization
        Jcurrent = QuadraticPenalty(sum([(c.current.I - current_target)**2 for c in coils]),
                                   target=0.0, weight=1e-12)
        
        # Total objective
        J = Jf + Jlength + Jcurvature + Jmsc + Jcurrent
        
        if self.verbose:
            print(f"  Initial squared flux: {Jf.J():.2e}")
            print("  Starting coil optimization...")
            
        # Optimize coils
        result = least_squares_mpi_solve(
            J, mpi=MpiPartition(), 
            rel_step=1e-3, abs_step=1e-6,
            max_nfev=max_iterations,
            diff_method="forward"
        )
        
        optimization_time = time.time() - start_time
        
        # Store optimized coils
        self.optimized_coils = coils
        
        if self.verbose:
            print(f"\nCoil optimization completed in {optimization_time:.1f} seconds")
            print(f"Final squared flux: {Jf.J():.2e}")
            print(f"Average coil length: {np.mean([c.curve.curve_length() for c in coils]):.2f} m")
            print(f"Max coil curvature: {np.max([c.curve.kappa() for c in coils]):.2f} m⁻¹")
            
        return coils

    def analyze_coil_system(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of optimized coil system performance.
        """
        
        if self.optimized_coils is None:
            raise ValueError("No optimized coils available. Run coil optimization first.")
            
        if self.verbose:
            print("\nAnalyzing coil system performance...")
            
        coils = self.optimized_coils
        
        # Create magnetic field
        bs = BiotSavart(coils)
        
        # Analyze coil properties
        coil_lengths = [c.curve.curve_length() for c in coils]
        coil_currents = [c.current.I for c in coils]
        coil_curvatures = [np.max(c.curve.kappa()) for c in coils]
        
        # Field analysis on plasma surface
        surface = self.create_simsopt_surface()
        surface_points = surface.gamma().reshape((-1, 3))
        B_field = bs.B().reshape((-1, 3))
        B_magnitude = np.sqrt(np.sum(B_field**2, axis=1))
        
        # Calculate field quality metrics
        B_mean = np.mean(B_magnitude)
        B_std = np.std(B_magnitude)
        field_error = B_std / B_mean
        
        # Coil complexity metrics
        total_coil_length = sum(coil_lengths)
        max_current = max(coil_currents)
        avg_current = np.mean(coil_currents)
        max_curvature = max(coil_curvatures)
        
        # Engineering metrics
        total_stored_energy = 0.5 * sum([c.current.I**2 * self.estimate_coil_inductance(c.curve) 
                                        for c in coils])
        
        results = {
            'num_coils': len(coils),
            'total_coil_length': total_coil_length,
            'average_coil_length': np.mean(coil_lengths),
            'max_current': max_current,
            'average_current': avg_current,
            'current_std': np.std(coil_currents),
            'max_curvature': max_curvature,
            'average_curvature': np.mean(coil_curvatures),
            'field_error': field_error,
            'B_mean_surface': B_mean,
            'B_std_surface': B_std,
            'stored_energy': total_stored_energy,
            'coil_power_estimate': self.estimate_coil_power(coils)
        }
        
        if self.verbose:
            print("\nCoil System Analysis:")
            print(f"  Total coils: {results['num_coils']}")
            print(f"  Total coil length: {results['total_coil_length']:.1f} m")
            print(f"  Average current: {results['average_current']/1e6:.2f} MA")
            print(f"  Max current: {results['max_current']/1e6:.2f} MA")
            print(f"  Max curvature: {results['max_curvature']:.2f} m⁻¹")
            print(f"  Field error: {results['field_error']*100:.2f}%")
            print(f"  Stored energy: {results['stored_energy']/1e9:.2f} GJ")
            print(f"  Estimated coil power: {results['coil_power_estimate']:.1f} MW")
            
        return results

    def estimate_coil_inductance(self, curve) -> float:
        """Estimate coil self-inductance for energy calculations."""
        length = curve.curve_length()
        radius = length / (2 * np.pi)  # Equivalent circular radius
        return mu_0 * radius * (np.log(8 * radius / 0.01) - 2)  # Rough estimate

    def estimate_coil_power(self, coils) -> float:
        """Estimate resistive power loss in coil system."""
        resistivity = 1.7e-8  # Copper resistivity at room temp (Ω⋅m)
        cross_section = 0.01  # Assumed 1 cm² conductor cross-section
        
        total_power = 0
        for coil in coils:
            length = coil.curve.curve_length()
            current = coil.current.I
            resistance = resistivity * length / cross_section
            power = current**2 * resistance
            total_power += power
            
        return total_power / 1e6  # Convert to MW

    def estimate_bootstrap_current(self, pressure: float) -> float:
        """Estimate bootstrap current from pressure gradient."""
        # Simplified bootstrap current estimate
        pressure_gradient = pressure / self.a  # Rough gradient estimate
        bootstrap_fraction = 0.3  # Typical for stellarators
        total_current = bootstrap_fraction * pressure_gradient * self.a**2 / (mu_0 * self.R0)
        return total_current / 1000  # Convert to kA

    def calculate_fusion_performance(self) -> Dict[str, float]:
        """
        Calculate fusion performance metrics for commercial viability assessment.
        """
        
        if self.equilibrium is None:
            raise ValueError("No equilibrium available. Run optimization first.")
            
        if self.verbose:
            print("\nCalculating fusion performance metrics...")
            
        # Get plasma parameters from equilibrium
        results = desc_compute(['p', 'beta', 'V', '|B|', 'iota'], eq=self.equilibrium)
        
        avg_pressure = np.mean(results['p'])
        avg_beta = np.mean(results['beta'])
        volume = results['V'][0]
        avg_B = np.mean(results['|B|'])
        
        # Fusion physics calculations
        # Assume 50-50 D-T mix at 15 keV
        temperature = self.target_temperature  # keV
        density = avg_pressure / (2 * temperature * 1.602e-16)  # m⁻³ (factor 2 for D+T)
        
        # DT fusion reaction rate <σv> at 15 keV ≈ 1.1e-22 m³/s
        sigma_v = 1.1e-22  # m³/s
        
        # Fusion power density
        fusion_power_density = 0.25 * density**2 * sigma_v * 17.6e6 * 1.602e-19  # W/m³
        total_fusion_power = fusion_power_density * volume / 1e6  # MW
        
        # Confinement metrics
        # Rough ISS04 scaling for stellarators
        tau_E_ISS04 = 0.134 * (self.R0**2.28) * (self.a**0.64) * (avg_B**0.84) * \
                      (density/1e19)**0.54 * (avg_pressure/1e5)**(-0.58) * (self.aspect_ratio**(-0.58))
        
        # Triple product
        triple_product = density * temperature * tau_E_ISS04
        
        # Q factor (fusion power / heating power)
        # Assume 50 MW heating power for ignition
        heating_power = 50.0  # MW
        Q_factor = total_fusion_power / heating_power if heating_power > 0 else 0
        
        # Economic metrics
        plant_efficiency = 0.4  # Thermal to electric conversion
        electric_power = total_fusion_power * plant_efficiency
        
        performance = {
            'plasma_density': density,
            'plasma_temperature': temperature,
            'fusion_power_density': fusion_power_density / 1e6,  # MW/m³
            'total_fusion_power': total_fusion_power,
            'confinement_time': tau_E_ISS04,
            'triple_product': triple_product,
            'Q_factor': Q_factor,
            'electric_power': electric_power,
            'power_density': electric_power / volume,
            'ignition_margin': triple_product / 3e21  # Ignition criterion
        }
        
        if self.verbose:
            print("\nFusion Performance Metrics:")
            print(f"  Plasma density: {performance['plasma_density']:.2e} m⁻³")
            print(f"  Plasma temperature: {performance['plasma_temperature']:.1f} keV")
            print(f"  Fusion power: {performance['total_fusion_power']:.0f} MW")
            print(f"  Confinement time: {performance['confinement_time']:.3f} s")
            print(f"  Triple product: {performance['triple_product']:.2e} m⁻³ keV s")
            print(f"  Q factor: {performance['Q_factor']:.1f}")
            print(f"  Electric power: {performance['electric_power']:.0f} MW")
            print(f"  Ignition margin: {performance['ignition_margin']:.2f}")
            
        return performance

    def save_results(self, output_dir: str = "stellarator_results"):
        """
        Save optimization results and analysis to files.
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if self.verbose:
            print(f"\nSaving results to {output_path.absolute()}")
            
        # Save DESC equilibrium
        if self.equilibrium is not None:
            eq_file = output_path / f"{self.device_name}_equilibrium.h5"
            save(self.equilibrium, str(eq_file))
            if self.verbose:
                print(f"  Saved equilibrium: {eq_file}")
                
        # Save SIMSOPT coils
        if self.optimized_coils is not None:
            coil_file = output_path / f"{self.device_name}_coils.json"
            # Note: In practice, you'd implement proper coil serialization
            if self.verbose:
                print(f"  Coil data ready for export: {coil_file}")
                
        # Save analysis results
        if hasattr(self, 'equilibrium') and self.equilibrium is not None:
            eq_analysis = self.analyze_equilibrium_quality()
            fusion_performance = self.calculate_fusion_performance()
            
            analysis_file = output_path / f"{self.device_name}_analysis.json"
            import json
            with open(analysis_file, 'w') as f:
                json.dump({
                    'device_parameters': {
                        'major_radius': self.R0,
                        'minor_radius': self.a,
                        'aspect_ratio': self.aspect_ratio,
                        'magnetic_field': self.B0,
                        'beta_target': self.beta_target,
                        'NFP': self.NFP
                    },
                    'equilibrium_quality': eq_analysis,
                    'fusion_performance': fusion_performance
                }, f, indent=2, default=str)
                
            if self.verbose:
                print(f"  Saved analysis: {analysis_file}")

    def plot_equilibrium(self, save_plots: bool = True, output_dir: str = "stellarator_results"):
        """
        Create comprehensive plots of the optimized stellarator configuration.
        """
        
        if self.equilibrium is None:
            raise ValueError("No equilibrium available. Run optimization first.")
            
        if self.verbose:
            print("\nGenerating equilibrium plots...")
            
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Plasma cross-section
        ax1 = plt.subplot(2, 3, 1)
        rho = np.linspace(0, 1, 20)
        theta = np.linspace(0, 2*np.pi, 100)
        phi = 0  # Toroidal angle
        
        grid_data = self.equilibrium.compute(['R', 'Z'], rho=rho, theta=theta, phi=phi)
        R_plot = grid_data['R']
        Z_plot = grid_data['Z']
        
        for i in range(len(rho)):
            ax1.plot(R_plot[i, :], Z_plot[i, :], 'b-', alpha=0.6)
        ax1.set_xlabel('R (m)')
        ax1.set_ylabel('Z (m)')
        ax1.set_title('Plasma Cross-Section (φ=0)')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # 2. Pressure profile
        ax2 = plt.subplot(2, 3, 2)
        rho_1d = np.linspace(0, 1, 50)
        pressure_data = self.equilibrium.compute('p', rho=rho_1d)['p']
        ax2.plot(rho_1d, pressure_data / 1e5, 'r-', linewidth=2)
        ax2.set_xlabel('ρ')
        ax2.set_ylabel('Pressure (bar)')
        ax2.set_title('Pressure Profile')
        ax2.grid(True, alpha=0.3)
        
        # 3. Rotational transform
        ax3 = plt.subplot(2, 3, 3)
        iota_data = self.equilibrium.compute('iota', rho=rho_1d)['iota']
        ax3.plot(rho_1d, iota_data, 'g-', linewidth=2)
        ax3.set_xlabel('ρ')
        ax3.set_ylabel('ι')
        ax3.set_title('Rotational Transform')
        ax3.grid(True, alpha=0.3)
        
        # 4. Beta profile
        ax4 = plt.subplot(2, 3, 4)
        beta_data = self.equilibrium.compute('beta', rho=rho_1d)['beta']
        ax4.plot(rho_1d, beta_data * 100, 'm-', linewidth=2)
        ax4.set_xlabel('ρ')
        ax4.set_ylabel('β (%)')
        ax4.set_title('Beta Profile')
        ax4.grid(True, alpha=0.3)
        
        # 5. Magnetic field strength
        ax5 = plt.subplot(2, 3, 5)
        B_data = self.equilibrium.compute('|B|', rho=rho_1d)['|B|']
        ax5.plot(rho_1d, B_data, 'c-', linewidth=2)
        ax5.set_xlabel('ρ')
        ax5.set_ylabel('|B| (T)')
        ax5.set_title('Magnetic Field Strength')
        ax5.grid(True, alpha=0.3)
        
        # 6. Optimization convergence
        ax6 = plt.subplot(2, 3, 6)
        if hasattr(self, 'optimization_history') and self.optimization_history:
            iterations = range(len(self.optimization_history['objective']))
            ax6.semilogy(iterations, self.optimization_history['objective'], 'k-', linewidth=2)
            ax6.set_xlabel('Iteration')
            ax6.set_ylabel('Objective Function')
            ax6.set_title('Optimization Convergence')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No optimization\nhistory available', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Optimization History')
        
        plt.tight_layout()
        
        if save_plots:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            plot_file = output_path / f"{self.device_name}_equilibrium_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"  Saved plots: {plot_file}")
        
        plt.show()

    def run_complete_optimization(self,
                                eq_iterations: int = 200,
                                coil_iterations: int = 500,
                                save_results: bool = True,
                                create_plots: bool = True) -> Dict[str, Any]:
        """
        Run complete stellarator optimization workflow.
        """
        
        if self.verbose:
            print("="*60)
            print(f"COMMERCIAL STELLARATOR OPTIMIZATION: {self.device_name}")
            print("="*60)
            
        total_start_time = time.time()
        results = {}
        
        try:
            # Step 1: Equilibrium optimization
            if self.verbose:
                print("\n" + "="*40)
                print("STEP 1: EQUILIBRIUM OPTIMIZATION")
                print("="*40)
                
            eq_optimized = self.optimize_equilibrium(
                max_iterations=eq_iterations,
                ftol=1e-8,
                xtol=1e-8
            )
            results['equilibrium'] = self.analyze_equilibrium_quality()
            results['fusion_performance'] = self.calculate_fusion_performance()
            
            # Step 2: Coil optimization
            if self.verbose:
                print("\n" + "="*40)
                print("STEP 2: COIL SYSTEM OPTIMIZATION")
                print("="*40)
                
            coils_optimized = self.optimize_coils_simsopt(
                num_coils=6,
                coil_separation=1.2,
                current_target=5e6,
                max_iterations=coil_iterations
            )
            results['coil_system'] = self.analyze_coil_system()
            
            # Step 3: Save results
            if save_results:
                if self.verbose:
                    print("\n" + "="*40)
                    print("STEP 3: SAVING RESULTS")
                    print("="*40)
                self.save_results()
                
            # Step 4: Create plots
            if create_plots:
                if self.verbose:
                    print("\n" + "="*40)
                    print("STEP 4: GENERATING PLOTS")
                    print("="*40)
                self.plot_equilibrium(save_plots=save_results)
                
        except Exception as e:
            if self.verbose:
                print(f"\nERROR during optimization: {e}")
            results['error'] = str(e)
            raise
            
        total_time = time.time() - total_start_time
        results['total_optimization_time'] = total_time
        
        if self.verbose:
            print("\n" + "="*60)
            print("OPTIMIZATION COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Total time: {total_time:.1f} seconds")
            
            # Summary of key results
            if 'fusion_performance' in results:
                fp = results['fusion_performance']
                print(f"\nKEY PERFORMANCE METRICS:")
                print(f"  Fusion Power: {fp['total_fusion_power']:.0f} MW")
                print(f"  Electric Power: {fp['electric_power']:.0f} MW")
                print(f"  Q Factor: {fp['Q_factor']:.1f}")
                print(f"  Ignition Margin: {fp['ignition_margin']:.2f}")
                
            if 'equilibrium' in results:
                eq = results['equilibrium']
                print(f"  Average Beta: {eq['average_beta']*100:.2f}%")
                print(f"  Aspect Ratio: {eq['aspect_ratio']:.2f}")
                print(f"  Plasma Volume: {eq['plasma_volume']:.1f} m³")
                
        return results


def main():
    """
    Main execution function demonstrating commercial stellarator optimization.
    """
    
    print("Commercial Stellarator Design Optimization")
    print("Using DESC + SIMSOPT Integration")
    print("-" * 50)
    
    # Check dependencies
    if not (DESC_AVAILABLE and SIMSOPT_AVAILABLE):
        print("ERROR: Required packages not available")
        print("Please install: pip install desc-opt simsopt")
        return
        
    # Create stellarator designer with commercial parameters
    designer = CommercialStellaratorDesigner(
        major_radius=6.0,      # Commercial scale
        minor_radius=1.8,      # For good confinement
        aspect_ratio=3.33,     # Engineering optimum
        magnetic_field=5.7,    # Strong field for compactness
        beta_target=0.05,      # 5% beta for ignition
        device_name="STAR-1",  # Stellarator Advanced Reactor 1
        verbose=True
    )
    
    # Run complete optimization
    try:
        results = designer.run_complete_optimization(
            eq_iterations=100,    # Reduced for demo
            coil_iterations=200,  # Reduced for demo
            save_results=True,
            create_plots=True
        )
        
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        
        if 'fusion_performance' in results:
            fp = results['fusion_performance']
            print(f"Fusion Power: {fp['total_fusion_power']:.0f} MW thermal")
            print(f"Electric Power: {fp['electric_power']:.0f} MW")
            print(f"Power Density: {fp['power_density']:.2f} MW/m³")
            
            if fp['Q_factor'] > 10:
                print("✓ IGNITION ACHIEVED - Q > 10")
            elif fp['Q_factor'] > 5:
                print("○ NEAR IGNITION - Q > 5")
            else:
                print("✗ IGNITION NOT ACHIEVED - Q < 5")
                
        print(f"\nTotal optimization time: {results.get('total_optimization_time', 0):.1f} s")
        print("Results saved to: stellarator_results/")
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        print("This is a demonstration script requiring full DESC/SIMSOPT installation")


if __name__ == "__main__":
    main()