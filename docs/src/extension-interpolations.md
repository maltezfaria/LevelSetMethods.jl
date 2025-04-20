# [Interpolations extension](@id extension-interpolations)

This extension overloads the `interpolate` function from
[`Interpolations.jl`](https://juliamath.github.io/Interpolations.jl/latest/) to provide a
way to construct a global interpolant from the discrete data in a
[`LevelSet`](@ref) or [`LevelSetEquation`](@ref). This can be useful in situations where you want
to evaluate the approximate underlying functions at points that are not on the grid.

Here is an example of how to construct such an interpolant:

```@example interpolations
using LevelSetMethods, Interpolations
LevelSetMethods.set_makie_theme!()
a, b = (-2, -2), (2, 2)
ϕ   = LevelSetMethods.star(CartesianGrid(a, b, (50, 50)))
itp = interpolate(ϕ, BSpline(Cubic())) # create the interpolant
```

Once constructed, the interpolant can be used to evaluate the level-set function anywhere
inside the grid:

```@example interpolations
itp(0.5, 0.5)
```

This can be used e.g. to plot the level-set function using `Makie`:

```@example interpolations
using GLMakie
xx = yy = -2:0.01:2
contour(xx, yy, [itp(x,y) for x in xx, y in yy]; levels = [0], linewidth = 2)
```

Trying to evaluate it outside the domain will throw an error:

```@example interpolations
try
  itp(3, 0.1)
catch e
    println("Error caught")
end
```

Using it on three-dimensional level sets is similar:

```@example interpolations
using LinearAlgebra
grid = CartesianGrid((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5), (50, 50, 50))
P1, P2 = (-1, 0, 0), (1, 0, 0)
b = 1.05
f = (x) -> norm(x .- P1)*norm(x .- P2) - b^2
ϕ = LevelSet(f, grid)
itp = interpolate(ϕ) # cubic spline by default
println("ϕ(0.5, 0.5, 0.5)   = ", f((0.5, 0.5, 0.5)))
println("itp(0.5, 0.5, 0.5) = ", itp(0.5, 0.5, 0.5))
```
