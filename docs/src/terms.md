# [Level-set terms](@id terms)

A level-set equation is given by

```math
  \phi_t + \sum_n \texttt{term}_n = 0
```

where each ``\texttt{term}_n`` is a `LevelSetTerm` object. Here we investigate the meaning
of each term, and how they can be used to model different phenomena.

## [Advection](@id advection)

The simplest term is the advection term, which is given by

```math
  \mathbf{u} \cdot \nabla \phi
```

where ``\mathbf{u}`` is a velocity field. This term models the transport of the level-set by
an *external* velocity field. You can construct an advection term using the `AdvectionTerm`
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
    ax = Axis(fig[1,n])
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
    # ax = Axis(fig[I[1],I[2]])
    ax = Axis(fig[1,n])
    plot!(ax, eq)
end
fig
```

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
# do a full revolution
tf = π
ax = Axis(fig[1,2], title = "Upwind")
integrate!(eq_upwind, tf)
plot!(ax, eq_upwind)
ax = Axis(fig[1,3], title = "WENO5")
integrate!(eq_weno, tf)
plot!(ax, eq_weno)
fig
```
