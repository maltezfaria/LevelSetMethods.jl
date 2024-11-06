# [Boundary conditions](@id boundary-conditions)

The following boundary conditions are available:

```@example
using LevelSetMethods
using InteractiveUtils # hide
subtypes(LevelSetMethods.BoundaryCondition)
```

When constructing a level-set equation, you can pass up to $2^d$ boundary conditions, where
$d$ is the dimension of the space. The following convention is followed:

- if you pass a single boundary condition, it is applied to all $2^d$ boundaries
- if you pass a vector `bcs` of $d$ boundary conditions, the $i$-th element is applied to the
  $i$-th direction. Two options are then possible:
  - `bcs[i]` is a single boundary condition, in which case it is applied to both boundaries in
    the $i$-th direction
  - `bcs[i]` is a tuple of two boundary conditions, in which case the first element is applied to
    the lower/left boundary and the second element to the upper/right boundary

Here is how it looks in practice:

```@example boundary-conditions
using LevelSetMethods, GLMakie
grid = CartesianGrid((-1,-1), (1,1), (100, 100))
ϕ₀    = LevelSet(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)
bc   = PeriodicBC()
eq   = LevelSetEquation(; levelset = deepcopy(ϕ₀), bc, terms = AdvectionTerm((x,t) -> (1,0)))
fig = Figure(; size = (1200, 300))
for (n,t) in enumerate([0.0, 0.5, 0.75, 1.0])
    integrate!(eq, t)
    ax = Axis(fig[1,n], title = "t = $t")
    plot!(ax, eq)
end
fig
```

Changing `PeriodicBC()` to `NeumannBC()` gives allows for the level-set to "leak" out of the domain:

```@example boundary-conditions
eq   = LevelSetEquation(; levelset = deepcopy(ϕ₀), bc = NeumannBC(), terms = AdvectionTerm((x,t) -> (1,0)))
fig = Figure(; size = (1200, 300))
for (n,t) in enumerate([0.0, 0.5, 0.75, 1.0])
    integrate!(eq, t)
    ax = Axis(fig[1,n], title = "t = $t")
    plot!(ax, eq)
end
fig
```

To combine both boundary conditions you can use

```@example boundary-conditions
bc = (NeumannBC(), PeriodicBC()) # Neumann in x, periodic in y
eq   = LevelSetEquation(; levelset = deepcopy(ϕ₀), bc, terms = AdvectionTerm((x,t) -> (1,1)))
fig = Figure(; size = (1200, 300))
for (n,t) in enumerate([0.0, 0.5, 0.75, 1.0])
    integrate!(eq, t)
    ax = Axis(fig[1,n], title = "t = $t")
    plot!(ax, eq)
end
fig
```

For more details on each boundary condition, see the docstring for the corresponding type.
