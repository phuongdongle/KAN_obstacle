# KAN-Free-Boundary-PDE


This repository implements Kolmogorov--Arnold Network (KAN)-based physics-informed solvers for free-boundary partial differential equations (PDEs). The main focus is on using structured KAN representations to approximate both the PDE solution and the evolving free boundary.

The repository includes examples for:

- The two-dimensional one-phase Stefan problem
- Elliptic obstacle problems
- Free-boundary PDEs with known analytical solutions
- Error visualization and time-dependent animations

---

## 1. KAN Solver for the Time-Dependent Stefan Problem

The main example in this repository is the two-dimensional one-phase Stefan problem. The goal is to learn both the time-dependent solution

\[
u(x_1,x_2,t)
\]

and the moving free boundary

\[
s(x_2,t),
\]

using a KAN-based physics-informed neural solver.

The computational domain is

\[
(x_1,x_2,t) \in [0,2.25]\times[0,1]\times[0,1],
\]

with the moving physical domain defined by

\[
0 < x_1 < s(x_2,t), \qquad 0 < x_2 < 1, \qquad 0 \leq t \leq 1.
\]

The KAN model is trained by minimizing residual losses associated with:

- the heat equation,
- the initial condition,
- boundary conditions,
- the interface condition,
- and the Stefan free-boundary condition.

---

## 2. Time-Dependent KAN Prediction and Error

The animation below shows the learned KAN solution \(u_{\mathrm{KAN}}(x_1,x_2,t)\) evolving from \(t=0\) to \(t=1\), together with the pointwise absolute error.

Left panel: predicted KAN solution as a 3D surface.  
Right panel: absolute error as a 2D heatmap.

```markdown
![KAN Stefan prediction and error](Stefan_Problem/KAN_model/gif/kan_stefan_upred_error_panel.gif)
