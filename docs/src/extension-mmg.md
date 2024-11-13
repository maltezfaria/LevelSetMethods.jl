# [MMG extension](@id extension-mmg)

This extension provides functions to generate meshes of levelset functions using [MMG](https://www.mmgtools.org/).
It define two methods: `export_volume_mesh` and `export_surface_mesh`.
For both of them, it is possible to control the size of the generated mesh using the following optional parameters:

- `hgrad` control the growth ratio between two adjacent edges.
- `hmin` and `hmax` control the edge sizes to be (respectively) greater than the `hmin` parameter and lower than the `hmax` one.
- `hausd` control the maximal distance between the piecewise linear representation of the boundary and the reconstructed ideal boundary.

## Generation of 2D and 3D mesh from a level-set

For 2 and 3 dimensional Cartesian levelset, one can use the `export_volume_mesh` function to generate meshes.
This method relies on the `mmg2d_O3` and `mmg3d_O3` utilities.
Example in 2D:

```@example volume2D
using LevelSetMethods, MMG_jll
grid = CartesianGrid((-2, -2), (2, 2), (50, 50))
ϕ = LevelSetMethods.star(grid)
volume_mesh2d = LevelSetMethods.export_volume_mesh(ϕ, joinpath(@__DIR__, "volume2D.mesh"))
```

You can then use e.g. `Gmsh` to visualize the mesh:

```@example volume2D
using GLMakie # hide
GLMakie.closeall() # hide
using Gmsh
try
  gmsh.initialize()
  gmsh.option.setNumber("General.Verbosity", 0)
  gmsh.open(volume_mesh2d)
  gmsh.fltk.initialize()
  gmsh.write(joinpath(@__DIR__, "volume2d.png"))
  gmsh.fltk.finalize()
finally
  gmsh.finalize()
end

```

![Volume2D](volume2d.png)

And similarly in 3D:

```@example volume3D
using LevelSetMethods, MMG_jll
grid = CartesianGrid((-1, -1, -1), (+1, +1, +1), (20, 20, 20))
ϕ = LevelSetMethods.sphere(grid; radius = 0.5)
volume_mesh3d = LevelSetMethods.export_volume_mesh(ϕ, joinpath(@__DIR__, "volume3d.mesh"))
```

![Volume3D](volume3d.png)

## Generation of 3D surface mesh with MarchingCubes.jl

Using the `mmgs_O3` utility, the `MarchingCubes.jl` library and the `export_surface_mesh` function it is possible to obtain a mesh of the levelset contour.

```@example surface3D
using LevelSetMethods, MMG_jll, MarchingCubes
grid = CartesianGrid((-2, -1, -1), (+2, +1, +1), (40, 20, 20))
ϕ₁ = LevelSetMethods.sphere(grid; radius = 0.5, center = (-1, 0, 0))
ϕ₂ = LevelSetMethods.sphere(grid; radius = 0.5, center = (+1, 0, 0))
ϕ₃ = LevelSetMethods.rectangle(grid; center = (0, 0, 0), width = (2, 0.25, 0.25))
ϕ  = ϕ₁ ∪ ϕ₂ ∪ ϕ₃
surf_mesh3d = LevelSetMethods.export_surface_mesh(ϕ, joinpath(@__DIR__,"surface3D.mesh"); hausd = 1.2, hmax = 1.0)
```

Again, to visualize it we can use `Gmsh`:

```@example surface3D
using GLMakie # hide
GLMakie.closeall() # hide
using Gmsh
try
  gmsh.initialize()
  gmsh.option.setNumber("General.Verbosity", 0)
  gmsh.open(surf_mesh3d)
  gmsh.fltk.initialize()
  gmsh.write(joinpath(@__DIR__, "surface3d.png"))
  gmsh.fltk.finalize()
finally
  gmsh.finalize()
end

```

![Surface3D](surface3d.png)
