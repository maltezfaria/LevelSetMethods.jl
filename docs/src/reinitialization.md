# [Reinitialization](@id reinitialization-newton)

Reinitialization transforms a level set function into a signed distance function without
moving the zero level set. This is useful when the level set has been distorted by
advection or other terms, since many numerical schemes assume ``|\nabla\phi| \approx 1``.

Two approaches are available:

- **[`EikonalReinitializationTerm`](@ref)**: a PDE-based approach that evolves the level
  set under the equation ``\phi_t + \text{sign}(\phi)(|\nabla\phi| - 1) = 0`` using the
  same time-stepping infrastructure as other level-set terms. See the [Reinitialization
  term](@ref reinitialization) section.
- **[`reinitialize!`](@ref)** (recommended): a geometry-based approach that samples the
  interface, builds a KD-tree, and computes the exact signed distance to the interface using
  Newton's closest-point method. It is applied between time steps and converges in a single
  pass.

## Usage

Call `reinitialize!` on a `MeshField` to reinitialize it in place:

```@example reinit
using LevelSetMethods
using GLMakie

grid = CartesianGrid((-1, -1), (1, 1), (100, 100))
sdf = MeshField(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)
ϕ = MeshField(x -> x[1]^2 + x[2]^2 - 0.5^2, grid)
LevelSetMethods.set_makie_theme!()
fig = Figure(; size = (800, 300))
ax1 = Axis(fig[1, 1]; title = "Signed Distance")
ax2 = Axis(fig[1, 2]; title = "Before Reinitialization", ylabel = "", yticklabelsvisible = false)
ax3 = Axis(fig[1, 3]; title = "After Reinitialization", ylabel = "", yticklabelsvisible = false)

contour!(ax1, sdf; levels = [0.25, 0, 0.5], labels = true, labelsize = 14)
contour!(ax2, ϕ; levels = [0.25, 0, 0.5], labels = true, labelsize = 14)

reinitialize!(ϕ)
contour!(ax3, ϕ; levels = [0.25, 0, 0.5], labels = true, labelsize = 14)
fig
```

You can verify that the reinitialized level set is a signed distance function:

```@example reinit
max_er = maximum(eachindex(grid)) do i
  abs(ϕ[i] - sdf[i])
end
println("Maximum error after reinitialization: $max_er")
```

## Reinitialization during integration

Reinitialization is not built into the equation; it is driven from a `prehook` passed to
[`integrate!`](@ref), which runs at the start of every accepted step. The simplest hook
reinitializes on every step:

```julia
integrate!(eq, tf; prehook = eq -> reinitialize!(current_state(eq); upsample = 4))
```

Because the hook is an ordinary function, you control *when* to reinitialize on any
criterion — a step counter closed over by the hook, the elapsed `current_time(eq)`, or a
measured drift of `|∇ϕ|` from one.

## `NewtonSDF`: a reusable signed distance function

`LevelSetMethods.NewtonSDF` wraps the same closest-point algorithm in a callable object,
letting you evaluate the signed distance at arbitrary points without modifying the level
set in place. This is useful when you need a signed distance function as an ingredient in
a larger computation (e.g. to measure distances or to build an extension velocity):

```@example reinit
using StaticArrays
sdf_obj = LevelSetMethods.NewtonSDF(ϕ; upsample = 8)
# query the signed distance at arbitrary points
sdf_obj(SVector(0.0, 0.0))   # distance from origin to the circle
```

The interface sample points used to build the KD-tree can be retrieved with
`LevelSetMethods.get_sample_points`:

```@example reinit
pts = LevelSetMethods.get_sample_points(sdf_obj)
println("$(length(pts)) interface sample points")
```

!!! note "Thread safety"
    `NewtonSDF` is safe to evaluate concurrently from multiple tasks: its interpolant keeps
    one scratch buffer per task, and the KD-tree and sample points are read-only during
    evaluation.
