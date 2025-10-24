# Reinitialization with Newton's Method

The `ReinitializationExt` extension provides a [`reinitialize!`](@ref) method to
transform a level set function into a signed distance function by computing the closest
point on the interface using Newton's method.

## Usage

After loading the required dependencies for this extension (`Interpolations` and
`NearestNeighbors`), simply call `reinitialize!` on a `LevelSet` or a `LevelSetEquation` to
reinitialize the level set function in-place:

```julia
using LevelSetMethods
using CairoMakie
using NearestNeighbors
using Interpolations

grid = CartesianGrid((-1, -1), (1, 1), (100, 100))
# An ellipse
sdf = LevelSet(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)
ϕ = LevelSet(x -> x[1]^2 + x[2]^2 - 0.5^2, grid)
LevelSetMethods.set_makie_theme!()
fig = Figure(; size = (800, 300))
ax1 = Axis(fig[1, 1]; title="Signed Distance")
ax2 = Axis(fig[1, 2]; title="Before Reinitialization", ylabel = "", yticklabelsvisible = false)
ax3 = Axis(fig[1, 3]; title="After Reinitialization", ylabel = "", yticklabelsvisible = false)

contour!(ax1, sdf; levels = [0.25, 0, 0.5], labels = true, labelsize = 14)
contour!(ax2, ϕ; levels = [0.25, 0, 0.5], labels = true, labelsize = 14)

# Reinitialize using Newton's method
reinitialize!(ϕ)
contour!(ax3, ϕ; levels = [0.25, 0, 0.5], labels = true, labelsize = 14)
fig
```

You can easily check that the reinitialized level set function is indeed a signed distance:

```julia
max_er = maximum(eachindex(grid)) do i
  abs(ϕ[i] - sdf[i])
end
@test max_er < 1e-10 # hide
println("Maximum error after reinitialization: $max_er")
```
