# [Interpolation](@id interpolation)

LevelSetMethods.jl provides a built-in high-performance piecewise polynomial
interpolation scheme. This allows you to construct a continuous interpolant from the
discrete data in a [`MeshField`](@ref) or [`LevelSetEquation`](@ref). This is useful for
evaluating the level-set function at arbitrary coordinates, computing analytical
gradients and Hessians, or for visualization.

## Basic Usage

Interpolation is enabled by wrapping a `MeshField` in an [`InterpolatedField`](@ref)
with the desired polynomial order. The wrapper is callable and evaluates the piecewise
polynomial interpolant:

```@example interpolation
using LevelSetMethods
a, b = (-2.0, -2.0), (2.0, 2.0)
star = MeshField(CartesianGrid(a, b, (50, 50))) do x # see the [geometry](@ref) page
    r, θ = hypot(x...), atan(x[2], x[1])
    return r - (1 + 0.25 * cos(5θ))
end
ϕ = InterpolatedField(star, 3)
```

Once constructed, `ϕ` can be evaluated at any point inside the grid:

```@example interpolation
ϕ(0.5, 0.5)
```

## Plotting

Interpolation is particularly useful for creating smooth plots. Here is an example
using `Makie`:

```@example interpolation
using GLMakie
LevelSetMethods.set_makie_theme!()
xx = yy = -2:0.01:2
# Evaluate on a fine grid for plotting
contour(xx, yy, [ϕ(x, y) for x in xx, y in yy]; levels = [0], linewidth = 2)
```

## Derivatives

The interpolant also provides analytical gradients and Hessians via the
[`LevelSetMethods.gradient`](@ref) and [`LevelSetMethods.hessian`](@ref) functions
(with fused `value_and_gradient` and `value_gradient_hessian` variants). These are
zero-allocation and differentiate the local Bernstein polynomial patch exactly.

```@example interpolation
using StaticArrays
x = SVector(0.1, 0.2)
val  = ϕ(x)
grad = LevelSetMethods.gradient(ϕ, x)
hess = LevelSetMethods.hessian(ϕ, x)
println("Value:    ", val)
println("Gradient: ", grad)
```

## Three-dimensional Level Sets

Using it on three-dimensional level sets is identical:

```@example interpolation
using LinearAlgebra, StaticArrays
grid3 = CartesianGrid((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5), (32, 32, 32))
P1, P2 = (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)
b = 1.05
f3 = (x) -> norm(x .- P1)*norm(x .- P2) - b^2
ϕ3 = InterpolatedField(MeshField(f3, grid3), 3)
println("ϕ(0.5, 0.5, 0.5)   = ", f3(SVector(0.5, 0.5, 0.5)))
println("ϕ(0.5, 0.5, 0.5) = ", ϕ3(0.5, 0.5, 0.5))
```
