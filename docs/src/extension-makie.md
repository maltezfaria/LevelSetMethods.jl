# [Makie extension](@id extension-makie)

[Makie](https://docs.makie.org/v0.21/) can be used to visualize level set functions in both 2
and 3 dimensions. After loading one of the `Makie` backends (we recommend `GLMakie` for 3D),
you can simply call `plot` on the level set function to visualize it. For example:

```@example contour2D
using LevelSetMethods, GLMakie
grid = CartesianGrid((-2, -2), (2, 2), (100, 100))
ϕ = LevelSetMethods.star(grid)
plot(ϕ)
```

By default, only the zero level set is plotted as a contour line. For more control, simply
call the `contour` (or `contourf`) function from `Makie` directly. For example:

```@example contour2D
contour(ϕ; levels = [-0.5, 0, 0.5], labels = true)
```

Although you can manually customize the `Axis` attributes for the plot, `LevelSetMethods`
provides a `Theme` with some reasonable defaults for plotting level set functions:

```@example contour2D
theme = LevelSetMethods.makie_theme()
with_theme(theme) do
  plot(ϕ)
end
```

In `3D`, the `plot` function will plot the zero level set as an isosurface. For example:

```@example volume3D
using LevelSetMethods, GLMakie, LinearAlgebra
grid = CartesianGrid((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5), (50, 50, 50))
P1, P2 = (-1, 0, 0), (1, 0, 0)
b = 1.05
ϕ = LevelSet(grid) do x
  norm(x .- P1)*norm(x .- P2) - b^2
end
theme = LevelSetMethods.makie_theme()
with_theme(theme) do
  plot(ϕ)
end
```

Once again, you can manually customize the options by calling the `volume` function from
`Makie` directly:

```@example volume3D
with_theme(theme) do
  volume(ϕ; algorithm = :iso, isovalue = 0.5)
end
```

!!! tip "Plotting a `LevelSetEquation`"
    Calling `plot` on a [`LevelSetEquation`](@ref) defaults to plotting the `LevelSet` given by its
    [`current_state`](@ref); exactly the same as calling `plot(current_state(equation))`.
