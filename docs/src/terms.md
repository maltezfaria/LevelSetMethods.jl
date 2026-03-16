# [Level-set terms](@id terms)

A level-set equation is given by

```math
  \phi_t + \sum_n \texttt{term}_n = 0
```

where each ``\texttt{term}_n`` is a `LevelSetTerm` object. The following terms are
available:

```@example
using LevelSetMethods
using InteractiveUtils # hide
subtypes(LevelSetMethods.LevelSetTerm)
```

Here we investigate the meaning of each term, and how they can be used to model different
phenomena.

## [Advection](@id advection)

The simplest term is the advection term, which is given by

```math
  \mathbf{u} \cdot \nabla \phi
```

where ``\mathbf{u}`` is a velocity field. This term models the transport of the level-set by
an *external* velocity field (see [osher2003level; Chapter 3](@cite)). You can construct an advection term using the `AdvectionTerm`
structure:

```@example advection-term
using LevelSetMethods, StaticArrays
grid = CartesianGrid((-1,-1), (1,1), (100, 100))
𝐮 = MeshField(x -> SVector(1,0), grid)
AdvectionTerm(𝐮)
```

In the example above we passed a [`MeshField`](@ref) object to the `AdvectionTerm`
constructor, meaning that the velocity field is simply a vector of values at each grid
point. This is useful if your velocity field is time-independent, or if you only know it at
grid points. Lets construct a level-set equation with an advection term:

```@example advection-term
ϕ₀ = LevelSet(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)
eq = LevelSetEquation(; terms = (AdvectionTerm(𝐮),), levelset = ϕ₀, bc = PeriodicBC())
```

To see how the advection term affects the level-set, we can solve the equation for a few
time steps:

```@example advection-term
using GLMakie
LevelSetMethods.set_makie_theme!()
fig = Figure(; size = (1200, 300))
# create a 2 x 2 figure
for (n,t) in enumerate([0.0, 0.5, 0.75, 1.0])
    I = CartesianIndices((2,2))[n]
    integrate!(eq, t)
    # ax = Axis(fig[I[1],I[2]])
    ax = Axis(fig[1,n], title = "t = $t")
    plot!(ax, eq)
end
fig
```

In the example above we see that the level-set is advected to the right. If we wanted to
have instead a time-dependent velocity field, we could pass a function to the
`AdvectionTerm`, and the velocity field would be computed at each time step. For example:

```@example advection-term
ϕ₀ = LevelSet(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)
eq = LevelSetEquation(; terms = (AdvectionTerm((x,t) -> SVector(x[1]^2, 0)),), levelset = ϕ₀, bc = PeriodicBC())
fig = Figure(; size = (1200, 300))
# create a 2 x 2 figure
for (n,t) in enumerate([0.0, 0.5, 0.75, 1.0])
    I = CartesianIndices((2,2))[n]
    integrate!(eq, t)
    ax = Axis(fig[1,n], title = "t = $t")
    plot!(ax, eq)
end
fig
```

Note that the velocity function must accept two arguments: the spatial coordinates `x`,
which is an abstract vector of length `d`, and the time `t`. Furthermore, it should return a
vector of length `d`.

Besides the velocity field, the `AdvectionTerm` constructor also accepts a `scheme` as a
second argument to specify the discretization scheme. The available options are:

- `Upwind()`: first-order upwind scheme
- `WENO5()`: fifth-order WENO scheme (default)

The WENO scheme is more expensive but much more accurate and is usually preferable to the
upwind scheme, which introduces significant numerical diffusion. To see their differences,
let us compare both schemes for a purely rotational velocity field:

```@example advection-term
ϕ₀ = LevelSetMethods.dumbbell(grid) # pre-defined level-set
𝐮  = MeshField(grid) do (x,y)
    SVector(-y, x)
end
eq_upwind = LevelSetEquation(; terms = AdvectionTerm(𝐮, Upwind()), levelset = deepcopy(ϕ₀), bc = PeriodicBC())
eq_weno   = LevelSetEquation(; terms = AdvectionTerm(𝐮), levelset  = deepcopy(ϕ₀), bc = PeriodicBC())
fig = Figure(size = (1000, 400))
ax = Axis(fig[1,1], title = "Initial")
plot!(ax, eq_upwind)
# do half a revolution
tf = π
ax = Axis(fig[1,2], title = "Upwind (final time)")
integrate!(eq_upwind, tf)
plot!(ax, eq_upwind)
ax = Axis(fig[1,3], title = "WENO5 (final time)")
integrate!(eq_weno, tf)
plot!(ax, eq_weno)
fig
```

## [Normal motion](@id normal-motion)

The normal motion term is given by

```math
  v |\nabla \phi|
```

where ``v`` is a scalar field. This term models the motion of the level-set in the normal
direction (see [osher2003level; Chapter 6](@cite)). Here is an example of how to use it:

```@example normal-motion-term
using LevelSetMethods
using GLMakie
grid = CartesianGrid((-2,-2), (2,2), (100, 100))
ϕ = LevelSetMethods.star(grid)
eq = LevelSetEquation(; terms = (NormalMotionTerm((x,t) -> 0.5),), levelset = ϕ, bc = PeriodicBC())
fig = Figure(; size = (1200, 300))
for (n,t) in enumerate([0.0, 0.5, 0.75, 1.0])
    integrate!(eq, t)
    ax = Axis(fig[1,n], title = "t = $t")
    plot!(ax, eq)
end
fig
```

As with `AdvectionTerm`, you can provide an update callback to mutate a mesh-based speed
field before each stage of time integration:

```@example normal-motion-term
vfield = MeshField(x -> 0.0, grid)
term = NormalMotionTerm(vfield, (v, ϕ, t) -> (values(v) .= 0.25 + 0.1 * t))
term
```

In Stefan problems, the speed `v` may only be known near the interface
`ϕ = 0`. You can extend that interface speed to a band around the interface using the
[`extend_along_normals!`](@ref), and then pass it to `NormalMotionTerm`:

```@example normal-motion-term
ϕext = LevelSetMethods.star(grid)
v = zeros(Float64, size(grid)...)
Δ = minimum(LevelSetMethods.meshsize(grid))
frozen = abs.(values(ϕext)) .<= 1.5Δ
for I in CartesianIndices(v)
    frozen[I] || continue
    x = grid[I]
    v[I] = 0.2 + 0.1 * cos(2π * atan(x[2], x[1]))
end
extend_along_normals!(v, ϕext; frozen, nb_iters = 80)
term = NormalMotionTerm(MeshField(v, grid, nothing))
term
```

## [Curvature motion](@id curvature)

This terms models the motion of the level-set in the normal direction with a velocity that
is proportional to the mean curvature:

```math
  b \kappa |\nabla \phi|
```

where ``\kappa = \nabla \cdot (\nabla \phi / |\nabla \phi|)`` is the mean curvature. Note that the
coefficient ``b`` should be negative; a positive value of ``b`` would yield an ill-posed
evolution problem (akin to a negative diffusion coefficient).

Here is the classic example of motion by mean curavature for a spiral-like level-set:

```@example curvature-term
using LevelSetMethods, GLMakie
grid = CartesianGrid((-1,-1), (1,1), (100, 100))
# create a spiral level-set
d = 1
r0 = 0.5
θ0 = -π / 3
α = π / 100.0
R = [cos(α) -sin(α); sin(α) cos(α)]
M = R * [1/0.06^2 0; 0 1/(4π^2)] * R'
ϕ = LevelSet(grid) do (x, y)
    r = sqrt(x^2 + y^2)
    θ = atan(y, x)
    result = 1e30
    for i in 0:4
        θ1 = θ + (2i - 4) * π
        v = [r - r0; θ1 - θ0]
        result = min(result, sqrt(v' * M * v) - d)
    end
    return result
end
eq = LevelSetEquation(; terms = (CurvatureTerm((x,t) -> -0.1),), levelset = ϕ, bc = PeriodicBC())
fig = Figure(; size = (1200, 300))
for (n,t) in enumerate([0.0, 0.1, 0.2, 0.3])
    integrate!(eq, t)
    ax = Axis(fig[1,n], title = "t = $t")
    plot!(ax, eq)
end
fig
```

## [Reinitialization term](@id reinitialization)

The reinitialization term is given by

```math
  \phi_t + \text{sign}(\phi) \left( |\nabla \phi| - 1 \right) = 0
```

This term is used to ensure that the level-set function remains close to a signed distance
function, which is sometimes important for numerical stability. The idea of the evolution
equation above is to penalize the deviation of the level-set from a signed distance
function, where ``|\nabla \phi| = 1``, without changing the zero level-set. In practice a
smeared `sign` function is used; see [osher2003level; Chapter 7](@cite) for more details.

Here is an example of how to use the reinitialization term to obtain a signed distance
function from a level-set. Let us first create a level-set that is not a signed distance,
and its signed distance function:

```@example reinitialization-term
using LevelSetMethods, GLMakie
grid = CartesianGrid((-1,-1), (1,1), (100, 100))
ϕ = LevelSet(x -> x[1]^2 + x[2]^2 - 0.5^2, grid) # circle level-set, but not a signed distance function
sdf = LevelSet(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid) # signed distance function
LevelSetMethods.set_makie_theme!()
fig = Figure(; size = (800, 400))
ax = Axis(fig[1,1], title = "Signed distance function")
contour!(ax, sdf; levels = [0.25, 0, 0.5], labels = true, labelsize = 14)
ax = Axis(fig[1,2], title = "ϕ at t = 0")
contour!(ax, ϕ, levels = [0.25, 0, 0.5], labels = true, labelsize = 14)
fig
```

We will now evolve the level-set using the reinitialization term:

```@example reinitialization-term
eq = LevelSetEquation(; terms = (EikonalReinitializationTerm(),), levelset = deepcopy(ϕ), bc = PeriodicBC())
fig = Figure(; size = (1200, 300))
for (n,t) in enumerate([0.0, 0.25, 0.5, 0.75])
    integrate!(eq, t)
    ax = Axis(fig[1,n], title = "t = $t")
    contour!(ax, LevelSetMethods.current_state(eq); levels = [0.25, 0, 0.5], labels = true, labelsize = 14)
end
fig
```

Observe that as the reinitialization equation evolves, `ϕ` approaches the signed distance function `sdf` depicted in the first figure.

Alternatively, you can use a modified reinitialization term that applies the sign function to the *initial level-set function* only:

```math
  \phi_t + \text{sign}(\phi_0) \left( |\nabla \phi| - 1 \right) = 0
```

To enable this behavior, simply pass a `LevelSet` object to the `EikonalReinitializationTerm`:

```@example reinitialization-term
eq = LevelSetEquation(; terms = (EikonalReinitializationTerm(ϕ),), levelset = deepcopy(ϕ), bc = PeriodicBC())
fig = Figure(; size = (1200, 300))
for (n,t) in enumerate([0.0, 0.25, 0.5, 0.75])
    integrate!(eq, t)
    ax = Axis(fig[1,n], title = "t = $t")
    contour!(ax, LevelSetMethods.current_state(eq); levels = [0.25, 0, 0.5], labels = true, labelsize = 14)
end
fig
```

The outcome closely matches that of the previous approach.

!!! tip "Consider Newton reinitialization instead"
    `EikonalReinitializationTerm` requires many pseudo-time steps to propagate corrections
    away from the interface and can cause spurious mass loss. For most use cases,
    [`NewtonReinitializer`](@ref) is a better choice: it samples the interface, builds a
    KD-tree, and computes the exact signed distance in a single pass. See the
    [Reinitialization](@ref reinitialization-newton) section for details.
