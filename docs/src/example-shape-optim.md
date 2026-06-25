```@meta
CurrentModule = LevelSetMethods
Draft = false
```

# [Shape optimization: a primer](@id example-shape-optim)

## Isoperimetric inequality as a shape optimization problem

We consider in this example the *isoperimetric inequality* which states that among all closed surfaces enclosing a fixed volume ``V_0 > 0``, the sphere is the one with minimal perimeter.
We show here how to demonstrate this result through numerical optimization.

!!! warning
    This example is purely illustrative. The optimization method used here has not been extensively tested.
    Coupling the [LevelSetMethods](https://github.com/maltezfaria/LevelSetMethods.jl) toolbox to any simulation package makes it possible to solve PDE-constrained optimization problems (see for instance [allaire2004structural](@cite)).

To do this, we first define the problem mathematically:

```math
    \begin{array}{rl}
        \displaystyle\min_{\Omega \subset \mathbb{R}^d} & P(\Omega)
        \\
        \text{s.t.} & V(\Omega) = V_0
    \end{array},\qquad\text{(1)}
```

where ``P(\Omega), V(\Omega)`` are the perimeter and volume of ``\Omega`` defined by

```math
    V(\Omega) = \int_{\Omega} \:\text{d}\mathbf{x}
    \quad\text{and}\quad
    P(\Omega) = \int_{\partial \Omega} \:\text{d}\mathbf{s}
```

The optimization problem ``\text{(1)}`` can be solved using the augmented Lagrangian approach by minimizing iteratively the following functional:

```math
    f(\Omega) = P(\Omega) + \lambda (V(\Omega) - V_0) + \frac{\mu}{2} (V(\Omega) - V_0)^2
    \qquad\text{(2)}
```

where ``\mu`` is a parameter updated during the course of the optimization.
To minimize ``\text{(2)}``, we use a gradient-based algorithm.
For this, we need to define what a *small variation* of ``\Omega`` is.
As such, for any shape ``\Omega \subset \mathbb{R}^d`` we define (following the Hadamard method) its deformation ``\Omega_{\boldsymbol{\theta}}`` by a small vector field ``\boldsymbol{\theta} \in W^{1,\infty}(\mathbb{R}^d, \mathbb{R}^d)`` as:

```math
    \Omega_{\boldsymbol{\theta}}
    = (\text{Id} + \boldsymbol{\theta})(\Omega)
    = \{\mathbf{x} + \boldsymbol{\theta}(\mathbf{x}), \mathbf{x} \in \Omega\}.
```

The following first-order Taylor expansion can then be obtained:

```math
    f(\Omega_{\boldsymbol{\theta}})
    =
    f(\Omega)
    +
    \int_{\partial \Omega}
    \left(
        \kappa + (\lambda + \mu (V(\Omega) - V_0))
    \right) \boldsymbol{\theta} \cdot \mathbf{n}
    \:\text{d}\mathbf{s}
    + o(\boldsymbol{\theta})
    .
```

In other words, using ``\boldsymbol{\theta} = - (\kappa + (\lambda + \mu (V(\Omega) - V_0))) \mathbf{n}`` and a small enough coefficient ``\tau > 0``, ``f(\Omega_{\tau\boldsymbol{\theta}})`` is necessarily smaller than ``f(\Omega)``.

## Numerical solution using the level-set method

If ``\Omega`` is given by the level-set function ``\phi_0 : \mathbb{R}^d \to \mathbb{R}`` then one associated with ``\Omega_{\tau\boldsymbol{\theta}}`` is given by ``\phi(\cdot, \tau)``, solution of

```math
    \partial_t \phi - \kappa |\nabla \phi| - (\lambda + \mu (V(\Omega) - V_0)) |\nabla \phi| = 0
```

with ``\phi(\cdot, t = 0) = \phi_0``.
The descent speed ``-(\lambda + \mu(V - V_0))`` depends on the current volume, so we let it track the shape as it moves and update the multipliers ``\lambda, \mu`` once per outer iteration. Two features of the library make this clean: a [`NormalMotionTerm`](@ref) can refresh its own speed from the current level set through an *update hook* called before every stage, and [`integrate!`](@ref) chooses stable time steps automatically — so there is no manual time-stepping to manage.

This optimization method is implemented in the following Julia code:

```@example optimization
using LevelSetMethods, LinearAlgebra, CairoMakie
LevelSetMethods.set_makie_theme!()

grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (50, 50))

ic = MeshField(grid) do x # a star; see the geometry page
    r, θ = norm(x), atan(x[2], x[1])
    return r - (1 + 0.25 * cos(5θ))
end

V0 = 0.5                            # target area
λ, μ, c = Ref(0.0), Ref(0.1), 1.1  # multiplier, penalty weight, penalty growth

# evolution ϕₜ - κ|∇ϕ| - (λ + μ(V - V₀))|∇ϕ| = 0: a fixed mean-curvature term plus a
# normal-motion term whose speed tracks the current volume, refreshed before each stage by
# the term's update hook
speed = MeshField(x -> 0.0, grid)
update_speed!(v, ϕ, t) = fill!(values(v), -(λ[] + μ[] * (LevelSetMethods.volume(ϕ) - V0)))
eq = LevelSetEquation(;
    terms = (NormalMotionTerm(speed, update_speed!), CurvatureTerm(MeshField(x -> -1.0, grid))),
    ic,
    bc = LinearExtrapolationBC(),
)

obs = Observable(eq)
fig = Figure()
ax = Axis(fig[1, 1])
plot!(ax, obs)
arc!(ax, Point2f(0, 0), sqrt(V0 / π), 0, 2π; linestyle = :dash, color = :red) # optimal disk

# each frame advances the shape under the current multipliers, then takes one dual-ascent step
record(fig, joinpath(@__DIR__, "optimization.gif"), range(0, 0.06; length = 100)) do t
    integrate!(eq, t, 0.1)
    V = LevelSetMethods.volume(eq)
    λ[] += μ[] * (V - V0)
    μ[] *= c
    obs[] = eq
end
```

![Optimization](optimization.gif)

The multipliers ``\lambda`` and ``\mu`` (and the growth factor `c`) trade off how much the optimization focuses on minimizing the objective versus satisfying the constraint. Refining the grid sharpens the final shape further, since the curvature that drives the rounding is then better resolved.
