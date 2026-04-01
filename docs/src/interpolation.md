# [Interpolation](@id interpolation)

LevelSetMethods.jl provides a built-in high-performance piecewise polynomial
interpolation scheme. This allows you to construct a continuous interpolant from the
discrete data in a [`LevelSet`](@ref) or [`LevelSetEquation`](@ref). This is useful for
evaluating the level-set function at arbitrary coordinates, computing analytical
gradients and Hessians, or for visualization.

## Basic Usage

To construct an interpolant, use the [`interpolate`](@ref) function:

```@example interpolation
using LevelSetMethods
a, b = (-2.0, -2.0), (2.0, 2.0)
ϕ   = LevelSetMethods.star(CartesianGrid(a, b, (50, 50)))
# Add boundary conditions for safe evaluation near edges
bc = ntuple(_ -> (NeumannGradientBC(), NeumannGradientBC()), 2)
ϕ = LevelSetMethods.add_boundary_conditions(ϕ, bc)

itp = interpolate(ϕ) # cubic interpolation by default (order=3)
```

The returned object is a [`PiecewisePolynomialInterpolation`](@ref LevelSetMethods.PiecewisePolynomialInterpolation), which is callable and
efficient. Once constructed, the interpolant can be used to evaluate the level-set function
anywhere inside (and even slightly outside, using boundary conditions) the grid:

```@example interpolation
itp(0.5, 0.5)
```

## Plotting

Interpolation is particularly useful for creating smooth plots. Here is an example
using `Makie`:

```@example interpolation
using GLMakie
LevelSetMethods.set_makie_theme!()
xx = yy = -2:0.01:2
# Evaluate on a fine grid for plotting
contour(xx, yy, [itp(x, y) for x in xx, y in yy]; levels = [0], linewidth = 2)
```

## Derivatives

The interpolant also provides analytical gradients and Hessians via the
[`LevelSetMethods.gradient`](@ref) and [`LevelSetMethods.hessian`](@ref) functions.
These are zero-allocation and computed using Horner's method on the underlying
Lagrange polynomials.

```@example interpolation
x = (0.1, 0.2)
val  = itp(x)
grad = LevelSetMethods.gradient(itp, x)
hess = LevelSetMethods.hessian(itp, x)
println("Value:    ", val)
println("Gradient: ", grad)
```

## Three-dimensional Level Sets

Using it on three-dimensional level sets is identical:

```@example interpolation
using LinearAlgebra, StaticArrays
grid = CartesianGrid((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5), (32, 32, 32))
P1, P2 = (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)
b = 1.05
f = (x) -> norm(x .- P1)*norm(x .- P2) - b^2
ϕ3 = LevelSet(f, grid)
bc3 = ntuple(_ -> (NeumannGradientBC(), NeumannGradientBC()), 3)
ϕ3 = LevelSetMethods.add_boundary_conditions(ϕ3, bc3)

itp3 = interpolate(ϕ3)
println("ϕ(0.5, 0.5, 0.5)   = ", f(SVector(0.5, 0.5, 0.5)))
println("itp(0.5, 0.5, 0.5) = ", itp3(0.5, 0.5, 0.5))
```
