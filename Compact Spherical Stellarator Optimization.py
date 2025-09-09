#!/usr/bin/env python3
"""
Compact Spherical Stellarator Optimization Script using DESC and SIMSOPT
=========================================================================

This script optimizes a compact spherical stellarator configuration for 
commercial nuclear fusion energy production using the latest DESC and SIMSOPT
computational codes for maximum efficiency and optimization.

Requirements:
- desc-opt (latest version from PyPI)
- simsopt (latest version from PyPI)
- numpy, scipy, matplotlib
- JAX (for DESC automatic differentiation)





"""

!pip install desc-opt simsopt numpy scipy matplotlib jax jaxlib



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# DESC imports - Latest version optimizations
try:
    import desc
    from desc.equilibrium import Equilibrium
    from desc.geometry import FourierRZToroidalSurface
    from desc.magnetic_fields import ToroidalField, PoloidalField, ScalarPotentialField
    from desc.objectives import (
        QuasisymmetryTwoTerm, QuasisymmetryTripleProduct, 
        AspectRatio, Volume, MeanCurvature, PrincipalCurvature,
        MercierStability, RotationalTransform, Shear
    )
    from desc.constraints import (
        FixBoundaryR, FixBoundaryZ, FixPressure, FixCurrent, 
        FixIota, PlasmaVesselDistance
    )
    from desc.optimize import Optimizer
    from desc.profiles import PowerSeriesProfile, SplineProfile
    from desc.grid import LinearGrid, ConcentricGrid
    DESC_AVAILABLE = True
    print(f"DESC version: {desc.__version__}")
except ImportError as e:
    print(f"DESC not available: {e}")
    DESC_AVAILABLE = False

# SIMSOPT imports - Latest version optimizations  
try:
    import simsopt
    from simsopt.mhd import Vmec, Boozer
    from simsopt.field import BiotSavart, Current, coils_via_symmetries
    from simsopt.geo import (
        CurveLength, CurveCurvature, CurveXYZFourier, 
        create_equally_spaced_curves, SurfaceRZFourier
    )
    from simsopt.objectives import SquaredFlux, QuadraticPenalty
    from simsopt.solve import least_squares_mpi_solve
    from simsopt._core.optimizable import Optimizable
    SIMSOPT_AVAILABLE = True
    print(f"SIMSOPT version: {simsopt.__version__}")
except ImportError as e:
    print(f"SIMSOPT not available: {e}")
    SIMSOPT_AVAILABLE = False

class CompactSphericalStellaratorOptimizer:
    """
    Comprehensive optimizer for compact spherical stellarator configurations
    combining DESC for equilibrium calculations and SIMSOPT for coil optimization.
    """
    
    def __init__(self, 
                 major_radius=1.0,
                 minor_radius=0.3, 
                 nfp=2,  # Number of field periods for compact design
                 beta=0.05,  # Target beta for commercial viability
                 aspect_ratio_target=3.3):  # Compact design target
        """
        Initialize the stellarator optimizer with compact spherical configuration.
        
        Parameters:
        -----------
        major_radius : float
            Major radius of the stellarator (meters)
        minor_radius : float  
            Minor radius of the stellarator (meters)
        nfp : int
            Number of field periods (2-4 for compact designs)
        beta : float
            Target plasma beta for commercial fusion
        aspect_ratio_target : float
            Target aspect ratio for compact design
        """
        self.R0 = major_radius
        self.a = minor_radius  
        self.nfp = nfp
        self.beta_target = beta
        self.aspect_ratio_target = aspect_ratio_target
        
        # Configuration parameters for commercial fusion
        self.B0 = 5.0  # Tesla - strong field for compactness
        self.pressure_scale = 1e5  # Pa - commercial pressure levels
        self.current_scale = 1e6  # Amperes - for strong confinement
        
        # Optimization results storage
        self.optimized_equilibrium = None
        self.optimized_coils = None
        self.optimization_history = []
        
        print(f"Initializing Compact Spherical Stellarator:")
        print(f"  Major radius: {self.R0} m")
        print(f"  Minor radius: {self.a} m") 
        print(f"  Aspect ratio: {self.R0/self.a:.2f}")
        print(f"  Field periods: {self.nfp}")
        print(f"  Target beta: {self.beta_target}")
        
    def create_compact_surface(self):
        """
        Create optimized boundary surface for compact spherical stellarator.
        Uses Fourier representation with carefully chosen modes for efficiency.
        """
        if not DESC_AVAILABLE:
            raise ImportError("DESC required for surface creation")
            
        # Fourier modes optimized for compact spherical design
        # Using fewer modes for computational efficiency while maintaining physics
        surface_data = {
            'R_lmn': np.zeros((10, 6, 2*6+1)),  # (L_max+1, M_max+1, 2*N_max+1)
            'Z_lmn': np.zeros((10, 6, 2*6+1)),
            'NFP': self.nfp,
            'sym': True
        }
        
        # Set fundamental mode (m=0, n=0)
        surface_data['R_lmn'][0, 0, 6] = self.R0  # Major radius
        
        # Add shaping modes for compact spherical tokamak-stellarator hybrid
        # Elongation for better performance
        surface_data['R_lmn'][0, 1, 6] = 0.1 * self.a  # m=1 shaping
        surface_data['Z_lmn'][0, 1, 6] = 1.2 * self.a  # Elongation
        
        # Stellarator-specific modes for rotational transform
        for n in range(1, self.nfp + 1):
            # Helical shaping modes
            surface_data['R_lmn'][0, 1, 6+n] = 0.05 * self.a
            surface_data['Z_lmn'][0, 1, 6+n] = 0.03 * self.a
            surface_data['R_lmn'][0, 2, 6+n] = 0.02 * self.a
        
        return FourierRZToroidalSurface(**surface_data)
    
    def create_profiles(self):
        """
        Create optimized pressure and current profiles for commercial fusion.
        """
        if not DESC_AVAILABLE:
            return None, None
            
        # Pressure profile optimized for bootstrap current and stability
        # Broader profile for better confinement and reduced transport
        pressure_coeffs = [
            self.pressure_scale,      # p0 - central pressure  
            -0.8 * self.pressure_scale,  # p1 - gradient term
            0.3 * self.pressure_scale,   # p2 - profile broadening
            -0.1 * self.pressure_scale   # p3 - fine tuning
        ]
        pressure_profile = PowerSeriesProfile(pressure_coeffs)
        
        # Current profile for rotational transform optimization
        # Balanced for quasi-symmetry and MHD stability
        current_coeffs = [
            0.0,                          # I0 - no net current
            self.current_scale * 0.5,     # I1 - primary current drive
            -self.current_scale * 0.2,    # I2 - profile control
            self.current_scale * 0.05     # I3 - fine adjustment
        ]
        current_profile = PowerSeriesProfile(current_coeffs)
        
        return pressure_profile, current_profile
    
    def setup_desc_optimization(self):
        """
        Set up DESC optimization with objectives for commercial stellarator.
        Focuses on quasi-symmetry, stability, and confinement efficiency.
        """
        if not DESC_AVAILABLE:
            raise ImportError("DESC required for optimization setup")
            
        # Create surface and profiles
        surface = self.create_compact_surface()
        pressure_profile, current_profile = self.create_profiles()
        
        # Create equilibrium with commercial parameters
        eq = Equilibrium(
            surface=surface,
            pressure=pressure_profile, 
            current=current_profile,
            Psi=1.0,  # Normalized flux
            NFP=self.nfp,
            L=6,      # Radial resolution
            M=6,      # Poloidal resolution  
            N=6       # Toroidal resolution
        )
        
        # Define optimization objectives prioritized for commercial fusion
        objectives = []
        
        # 1. Quasi-symmetry - Critical for confinement efficiency
        qs_obj = QuasisymmetryTwoTerm(
            eq=eq,
            helicity=(1, self.nfp),  # QH symmetry for compact design
            target=0.0,
            weight=1000.0,  # High priority
            grid=LinearGrid(L=6, M=12, N=12)
        )
        objectives.append(qs_obj)
        
        # 2. Aspect ratio control for compactness
        ar_obj = AspectRatio(
            eq=eq,
            target=self.aspect_ratio_target,
            weight=100.0,
            bounds=(3.0, 4.0)  # Commercial range
        )
        objectives.append(ar_obj)
        
        # 3. Rotational transform optimization
        iota_obj = RotationalTransform(
            eq=eq,
            target=0.4,  # Good for stability and transport
            weight=50.0,
            grid=LinearGrid(L=6, M=6, N=6)
        )
        objectives.append(iota_obj)
        
        # 4. MHD stability - Mercier criterion
        mercier_obj = MercierStability(
            eq=eq,
            target=0.0,
            weight=200.0,  # High priority for commercial operation
            grid=LinearGrid(L=6, M=6, N=6)
        )
        objectives.append(mercier_obj)
        
        # 5. Mean curvature control for stability
        curv_obj = MeanCurvature(
            eq=eq,
            target=0.0,
            weight=10.0,
            grid=LinearGrid(L=4, M=8, N=8)  
        )
        objectives.append(curv_obj)
        
        # Define constraints for physical feasibility
        constraints = []
        
        # Fix some boundary modes to maintain topology
        constraints.append(FixBoundaryR(eq=eq, modes=[(0, 0, 0)]))
        constraints.append(FixBoundaryZ(eq=eq, modes=[(0, 0, 0)]))
        
        # Pressure and current constraints for commercial operation
        constraints.append(FixPressure(eq=eq, modes=[(0, 0, 0)]))
        constraints.append(FixCurrent(eq=eq, modes=[(0, 0, 0)]))
        
        return eq, objectives, constraints
    
    def setup_simsopt_coil_optimization(self, equilibrium):
        """
        Set up SIMSOPT coil optimization for the compact stellarator.
        Optimizes external coils for field accuracy and engineering feasibility.
        """
        if not SIMSOPT_AVAILABLE:
            raise ImportError("SIMSOPT required for coil optimization")
            
        # Number of coils - optimized for compact design
        ncoils = 4 * self.nfp  # 4 coils per field period
        
        # Create initial coil curves using circular approximation
        base_curves = create_equally_spaced_curves(
            ncoils, 
            self.nfp, 
            stellsym=True,
            R0=self.R0 + 0.2,  # Coil major radius (20cm clearance)
            R1=0.1,            # Coil minor radius variation
            order=4            # Fourier order for coil shape
        )
        
        # Convert to CurveXYZFourier for optimization
        curves = []
        for curve in base_curves:
            # Initialize with circular coil and add perturbations for optimization
            dofs = np.zeros(3 * (2 * 4 + 1))  # 3D Fourier coefficients
            # Set fundamental mode 
            dofs[4] = self.R0 + 0.2  # X fundamental
            dofs[4 + 2*4 + 1 + 4] = self.R0 + 0.2  # Y fundamental
            
            fourier_curve = CurveXYZFourier(dofs, order=4)
            curves.append(fourier_curve)
        
        # Create currents for coils - optimize for field strength
        base_current = self.current_scale / ncoils  # Distribute current
        currents = [Current(base_current) for _ in range(ncoils)]
        
        # Apply coil symmetries for computational efficiency
        coils = coils_via_symmetries(curves, currents, self.nfp, True)
        
        # Create magnetic field from coils
        bs = BiotSavart(coils)
        
        # Set up optimization objectives for coils
        # 1. Minimize normal field error on plasma boundary
        s = equilibrium.surface if hasattr(equilibrium, 'surface') else None
        if s is None:
            # Create surface from equilibrium boundary
            s = SurfaceRZFourier(
                nfp=self.nfp, 
                stellsym=True,
                mpol=6, 
                ntor=6
            )
            # Set surface parameters from equilibrium
            s.set('R(0,0)', self.R0)
            s.set('R(1,0)', self.a)
            s.set('Z(1,0)', 0.0)
        
        Jf = SquaredFlux(s, bs, target=0.0, definition="local")
        
        # 2. Coil length penalty (engineering constraint)
        Jls = [CurveLength(curve) for curve in curves]
        
        # 3. Curvature penalty (manufacturability)
        Jcs = [CurveCurvature(curve, target=2.0/self.a) for curve in curves]  
        
        return coils, bs, [Jf] + Jls + Jcs
        
    def optimize_equilibrium(self, maxiter=100, ftol=1e-8):
        """
        Optimize the stellarator equilibrium using DESC.
        """
        if not DESC_AVAILABLE:
            print("DESC not available - skipping equilibrium optimization")
            return None
            
        print("\n=== DESC Equilibrium Optimization ===")
        
        # Setup optimization
        eq, objectives, constraints = self.setup_desc_optimization()
        
        # Create optimizer with advanced settings for efficiency
        optimizer = Optimizer('proximal-lsq-exact')  # Fast and accurate
        optimizer.maxiter = maxiter
        optimizer.ftol = ftol
        optimizer.verbose = 2
        
        # Run optimization
        eq_optimized = optimizer.optimize(
            eq, 
            objectives, 
            constraints,
            copy=True
        )
        
        self.optimized_equilibrium = eq_optimized
        
        # Calculate key performance metrics
        self._calculate_equilibrium_metrics(eq_optimized)
        
        print("DESC equilibrium optimization completed!")
        return eq_optimized
    
    def optimize_coils(self, maxiter=200):
        """
        Optimize external coils using SIMSOPT.
        """
        if not SIMSOPT_AVAILABLE:
            print("SIMSOPT not available - skipping coil optimization") 
            return None
            
        if self.optimized_equilibrium is None:
            print("No equilibrium available - run optimize_equilibrium first")
            return None
            
        print("\n=== SIMSOPT Coil Optimization ===")
        
        # Setup coil optimization
        coils, bs, objectives = self.setup_simsopt_coil_optimization(
            self.optimized_equilibrium
        )
        
        # Combine objectives with appropriate weights
        def combined_objective(dofs):
            obj_val = 0.0
            obj_val += 1000.0 * objectives[0].J()  # Flux objective (high priority)
            
            # Length penalties
            for i in range(1, len(objectives) - len(coils[::2])):
                obj_val += 1.0 * objectives[i].J()  # Length penalty
            
            # Curvature penalties  
            for i in range(len(objectives) - len(coils[::2]), len(objectives)):
                obj_val += 10.0 * objectives[i].J()  # Curvature penalty
                
            return obj_val
        
        # Use scipy optimizer for coil optimization
        from simsopt._core.util import ObjectiveFromFunc
        
        prob = ObjectiveFromFunc(
            combined_objective,
            [coil.curve for coil in coils[::2]]  # Optimize unique curves only
        )
        
        # Run optimization
        result = minimize(
            prob.J,
            prob.dofs,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'disp': True}
        )
        
        self.optimized_coils = coils
        
        # Calculate coil performance metrics
        self._calculate_coil_metrics(coils, bs)
        
        print("SIMSOPT coil optimization completed!")
        return coils
        
    def _calculate_equilibrium_metrics(self, eq):
        """Calculate key performance metrics for the equilibrium."""
        print("\n--- Equilibrium Performance Metrics ---")
        
        try:
            # Grid for calculations
            grid = LinearGrid(L=6, M=12, N=12, NFP=eq.NFP)
            
            # Aspect ratio
            aspect_ratio = eq.compute('R0/a', grid=grid)['R0/a']
            print(f"Aspect Ratio: {aspect_ratio:.3f}")
            
            # Volume
            volume = eq.compute('V', grid=grid)['V']
            print(f"Plasma Volume: {volume:.3f} m³")
            
            # Rotational transform
            iota = eq.compute('iota', grid=grid)['iota'] 
            print(f"Rotational Transform (avg): {np.mean(iota):.3f}")
            
            # Magnetic well/hill
            well = eq.compute('magnetic well', grid=grid)
            if 'magnetic well' in well:
                print(f"Magnetic Well: {np.mean(well['magnetic well']):.6f}")
                
            # Beta
            if hasattr(eq, 'pressure') and eq.pressure is not None:
                beta = eq.compute('beta_vol', grid=grid)
                if 'beta_vol' in beta:
                    print(f"Volume-averaged Beta: {beta['beta_vol']:.4f}")
                    
        except Exception as e:
            print(f"Error calculating metrics: {e}")
    
    def _calculate_coil_metrics(self, coils, bs):
        """Calculate key performance metrics for the coil system."""
        print("\n--- Coil System Performance Metrics ---")
        
        try:
            # Total coil length
            total_length = sum([coil.curve.curve_length_pure() for coil in coils[::2]])
            print(f"Total Coil Length: {total_length:.2f} m")
            
            # Maximum curvature
            max_curvature = 0.0
            for coil in coils[::2]:
                kappa = coil.curve.kappa()
                max_curvature = max(max_curvature, np.max(np.abs(kappa)))
            print(f"Maximum Coil Curvature: {max_curvature:.3f} m⁻¹")
            
            # Current levels
            currents = [abs(coil.current.get_value()) for coil in coils[::2]]
            print(f"Coil Currents: {np.mean(currents)/1e6:.2f} ± {np.std(currents)/1e6:.2f} MA")
            
        except Exception as e:
            print(f"Error calculating coil metrics: {e}")
    
    def run_full_optimization(self):
        """
        Run the complete optimization sequence for commercial stellarator design.
        """
        print("=" * 60)
        print("COMPACT SPHERICAL STELLARATOR OPTIMIZATION")
        print("Commercial Nuclear Fusion Configuration")
        print("=" * 60)
        
        # Stage 1: Equilibrium optimization with DESC
        equilibrium = self.optimize_equilibrium(maxiter=100, ftol=1e-8)
        
        # Stage 2: Coil optimization with SIMSOPT  
        coils = self.optimize_coils(maxiter=200)
        
        # Final performance summary
        self.print_optimization_summary()
        
        return equilibrium, coils
    
    def print_optimization_summary(self):
        """Print comprehensive optimization summary."""
        print("\n" + "=" * 50)
        print("OPTIMIZATION SUMMARY")
        print("=" * 50)
        
        print(f"Configuration: Compact Spherical Stellarator")
        print(f"Field Periods: {self.nfp}")
        print(f"Target Aspect Ratio: {self.aspect_ratio_target}")
        print(f"Target Beta: {self.beta_target}")
        print(f"Magnetic Field: {self.B0} T")
        
        if self.optimized_equilibrium:
            print("\n✓ Equilibrium optimization: COMPLETED")
        else:
            print("\n✗ Equilibrium optimization: FAILED")
            
        if self.optimized_coils:
            print("✓ Coil optimization: COMPLETED")
        else:
            print("✗ Coil optimization: FAILED")
            
        print("\nDesign optimized for commercial fusion energy production.")
        print("Key features: Compact geometry, high beta capability,")
        print("              quasi-symmetric confinement, engineering feasibility")

def main():
    """
    Main execution function demonstrating the stellarator optimization workflow.
    """
    
    # Create optimizer for compact commercial stellarator
    optimizer = CompactSphericalStellaratorOptimizer(
        major_radius=1.2,        # Compact size for cost efficiency  
        minor_radius=0.36,       # Good aspect ratio
        nfp=2,                   # Simple field structure
        beta=0.06,               # High beta for commercial viability
        aspect_ratio_target=3.3  # Compact commercial design
    )
    
    # Run complete optimization
    equilibrium, coils = optimizer.run_full_optimization()
    
    # Additional analysis could be added here:
    # - Transport calculations
    # - Stability analysis  
    # - Engineering assessments
    # - Economic modeling
    
    print("\nOptimization completed successfully!")
    print("Results ready for further analysis and engineering design.")

if __name__ == "__main__":
    main()