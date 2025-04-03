=========================================================================================================
Wall-resolved LES database of A-Airfoil @ 13.3 deg, Re_c = 1.0E7, M = 0.15

Originally uploaded on October  26, 2022

 Copyright (c) Soshi Kawai. All Rights Reserved.
 Contact: kawai@tohoku.ac.jp
=========================================================================================================

Original paper:
  Yoshiharu Tamaki & Soshi Kawai, 
  "Wall-resolved large-eddy simulation of near-stall airfoil flow at Re_c=10^7"
  AIAA Journal, 61 (2), 698-711, (2023).
  https://doi.org/10.2514/1.J062066

Related paper (Wall-resolved and wall-modeled LES of the same flow at a lower Reynolds number Re_c=2.1x10^6):
1)  Kengo Asada & Soshi Kawai, 
   "Large-eddy simulation of airfoil flow near stall condition at Reynolds number 2.1 \times 10^6,"
    Physics of Fluids (30), 085103, 2018.
2)  Yoshiharu Tamaki, Yuma Fukushima, Yuichi Kuya & Soshi Kawai,
   "Physics and modeling of trailing-edge stall phenomena for wall-modeled large-eddy simulation"
    Physical Review Fluids (5), 074602, 2020.

=========================================================================================================
This database includes

# airfoil.dat (Aerospatiale A-airfoil, interpolated by cubic spline)

# surface data.dat (BL quantities over the lower surface are blanked)
 x/c
 C_p
 C_f
 H (shape factor)
 Re_theta (Re based on the momentum thickness)
 Re_delta_star (Re based on the displacement thickness)
 
# profiles_{x/c coordinate}.dat
 Y/c (wall normal coordinate)
 U (wall-parallel velocity, normalized by u_inf)
 U_rms/u_tau (Reynolds normal stress in the streamwise direction)
 V_rms/u_tau (Reynolds normal stress in the wall-normal direction)
 W_rms/u_tau (Reynolds normal stress in the spanwise direction)
 <U'V'>/u_tau^2 (Reynolds shear stress)
 y+
 u+

# momentum_budget_{x/c coordinate}.dat (definition as in Fig. 20 of Ref. 1) above)
# in the far field at x/c=0.05 and 0.08, the terms may be incorrect due to grid curvature
 Y/c
 convection    (normalized by u_inf^2/c)
 viscous           (normalized by u_inf^2/c)
 pressure_gradient (normalized by u_inf^2/c)
 Reynolds_stress   (normalized by u_inf^2/c)

# tke_budget_{x/c coordinate}.dat (see Appendix B of the original paper for the definition)
 Y/c
 C (convection)
 P (production)
 epsilon (dissipation)
 Td (turbulent diffusion)
 Tp (velocity-pressure interaction)
 Vd (viscous diffusion)
 phi(2) (filter contribution for momentum equation)
 residual

 Each term is normalized by tau_w^2/mu_w 
 M, Pi, and phi(1), which are analytically zero in incompressible flows, are not included here.

# 2D data (C binary, little-endian, single precision, near-field cropped)
# (imax, jmax) = 17095 x 591
- grid_2d.xyz
  read(iu) imax,jmax
  read(iu) ((x(i,j),i=1,imax),j=1,jmax),((y(i,j),i=1,imax),j=1,jmax)

- statistics_2d.fun 
# (imax, jmax) = 17095 x 591
# (nvar)       = 8
  read(iu) imax,jmax,nvar
  read(iu) (((data(i,j,n),i=1,imax),j=1,jmax),n=1,nvar)

  data(i,j,1): Mean density, rho/rho_inf
  data(i,j,2): Mean x-velocity, u/u_inf
  data(i,j,3): Mean y-velocity, v/u_inf
  data(i,j,4): Mean pressure, p/p_inf
  data(i,j,5-8): Reynolds stress components (u'u', v'v', w'w', u'v, normalized by u_inf^2)
  (u, v, w) are the velocities in the Cartesian coordinates (x, y, z) (w being the spanwise velocity)
  =========================================================================================================
