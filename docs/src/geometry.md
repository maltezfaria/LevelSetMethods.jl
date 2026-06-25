```@meta
CurrentModule = LevelSetMethods
```

# [Level sets](@id geometry)

A level set is just a real-valued [`AbstractMeshField`](@ref) whose zero contour represents the
interface of interest (see [Grids and mesh fields](@ref grids)). There is no separate
"shape" type: you describe a geometry by the implicit function that vanishes on its
boundary, sample it with e.g. `MeshField(f, grid)`, and combine geometries with set operations.
This page collects the common patterns.

## Signed distance functions

The most convenient implicit function is a *signed distance function* (SDF): `ϕ(x)` returns
the distance from `x` to the interface, negative inside and positive outside. The same
one-liner builds a circle in 2D or a sphere in 3D — only the grid's dimension changes:

```@example geometry
using LevelSetMethods, LinearAlgebra
grid = CartesianGrid((-1, -1), (1, 1), (32, 32))
circle = MeshField(x -> norm(x) - 0.5, grid)   # ‖x‖ - r is a true SDF, in any dimension
```

The same function and grid can build a [`NarrowBandMeshField`](@ref) in one step, storing
values only on a band of nodes around the interface rather than the whole grid (see
[Narrow-band fields](@ref narrow-band)):

```@example geometry
circle_band = NarrowBandMeshField(x -> norm(x) - 0.5, grid; nlayers = 3)
@assert 0 < length(active_nodeindices(circle_band)) < length(grid) # hide
circle_band
```

## Non-SDF implicit functions

Any function with the correct sign works — it need not be a true distance. A box is easy to
write, but is *not* a true SDF away from its faces:

```@example geometry
box = MeshField(x -> maximum(abs.(x) .- (0.6, 0.3) ./ 2), grid)
```

Near the corners the value is not the actual distance, so `‖∇ϕ‖ ≠ 1`. If a downstream
algorithm needs an SDF (e.g. a narrow band, or curvature near corners), reinitialize it with
[`reinitialize!`](@ref):

```@example geometry
reinitialize!(box)
```

## Combining geometries with set operations

A level set denotes the domain ``Ω = \{x : ϕ(x) ≤ 0\}``. Operations on those domains are
represented by simple value combinations — union by `min`, intersection by `max`, set
difference by `max(ϕ₁, -ϕ₂)` — and are available as `∪`, `∩`, [`setdiff`](@ref) (with
in-place `union!`, `intersect!`, `setdiff!`).

A dumbbell is the union of two disks and a connecting bar:

```@example geometry
disk(c) = MeshField(x -> norm(x .- c) - 0.25, grid)
bar = MeshField(x -> maximum(abs.(x) .- (1.0, 0.2) ./ 2), grid)
dumbbell = disk((-0.5, 0.0)) ∪ disk((0.5, 0.0)) ∪ bar
```

A [Zalesak disk](@ref zalesak) is a disk with a rectangular slot cut out:

```@example geometry
slot = MeshField(x -> maximum(abs.(x .- (0.0, -0.5)) .- (0.25, 1.0) ./ 2), grid)
zalesak = setdiff(MeshField(x -> norm(x) - 0.5, grid), slot)
```

!!! note
    Combining SDFs with `min`/`max` keeps the sign correct everywhere but breaks the exact
    distance property near the seams. Reinitialize the result if you need a true SDF.

## Parametric interfaces

Interfaces described in other coordinates work just as well. A star is convenient in polar
form (this one is not an SDF):

```@example geometry
star = MeshField(grid) do x
    r, θ = norm(x), atan(x[2], x[1])
    return r - 0.75 * (1 + 0.25 * cos(5θ))
end
```

## Visualizing the result

Loading a Makie backend lets you inspect any of these level sets directly (see the [Makie
extension](@ref extension-makie)). The interior ``ϕ < 0`` is shaded and the zero contour
drawn on top:

```@example geometry
using CairoMakie
LevelSetMethods.set_makie_theme!()
fig = Figure(; size = (900, 280))
for (n, (name, ψ)) in enumerate(("dumbbell" => dumbbell, "zalesak" => zalesak, "star" => star))
    ax = Axis(fig[1, n]; title = name)
    plot!(ax, ψ)
end
fig
```
