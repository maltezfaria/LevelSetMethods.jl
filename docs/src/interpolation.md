# [Interpolation](@id interpolation)

LevelSetMethods.jl provides a built-in high-performance piecewise polynomial
interpolation scheme. This allows you to construct a continuous interpolant from the
discrete data in a [`MeshField`](@ref) or [`LevelSetEquation`](@ref). This is useful for
evaluating the level-set function at arbitrary coordinates, computing analytical
gradients and Hessians, or for visualization.

## Basic Usage

Interpolation is enabled by passing `interp_order` when constructing a `MeshField`.
The field itself becomes callable and evaluates the piecewise polynomial interpolant:

```@example interpolation
using LevelSetMethods
a, b = (-2.0, -2.0), (2.0, 2.0)
ϕ = LevelSetMethods.star(CartesianGrid(a, b, (50, 50)); interp_order = 3)
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
[`LevelSetMethods.gradient`](@ref) and [`LevelSetMethods.hessian`](@ref) functions.
These are zero-allocation and computed using Horner's method on the underlying
Lagrange polynomials.

```@example interpolation
using StaticArrays
x = SVector(0.1, 0.2)
I = LevelSetMethods.compute_index(ϕ, x)
p = LevelSetMethods.make_interpolant(ϕ, I)
val  = p(x)
grad = LevelSetMethods.gradient(p, x)
hess = LevelSetMethods.hessian(p, x)
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
ϕ3 = MeshField(f3, grid3; interp_order = 3)
println("ϕ(0.5, 0.5, 0.5)   = ", f3(SVector(0.5, 0.5, 0.5)))
println("ϕ(0.5, 0.5, 0.5) = ", ϕ3(0.5, 0.5, 0.5))
```
