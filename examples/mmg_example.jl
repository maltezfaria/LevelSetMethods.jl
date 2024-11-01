using LevelSetMethods
using LinearAlgebra
using MMG_jll

nx, ny, nz = 20, 20, 20
grid2D = CartesianGrid((-1, -1), (+1, +1), (nx, ny))
grid3D = CartesianGrid((-1, -1, -1), (+1, +1, +1), (nx, ny, nz))

ϕ2D = LevelSetMethods.star(grid2D)
ϕ3D = LevelSetMethods.sphere(grid3D)

#####################################

LevelSetMethods.export_volume_mesh(ϕ2D, "testVolume2D.mesh")
LevelSetMethods.export_volume_mesh(ϕ3D, "testVolume3D.mesh")

#####################################

using MarchingCubes

LevelSetMethods.export_surface_mesh(ϕ3D, "testSurface3D.mesh"; hausd = 1.2, hmax = 0.5)
