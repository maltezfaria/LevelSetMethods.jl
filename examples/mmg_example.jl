using LevelSetMethods
using LinearAlgebra
using MMG_jll

nx, ny, nz = 20, 20, 20
x = LinRange(-1, 1, nx)
y = LinRange(-1, 1, ny)
z = LinRange(-1, 1, nz)
grid2D = CartesianGrid(x, y)
grid3D = CartesianGrid(x, y, z)

ϕ2D = LevelSet(grid2D) do (x, y)
    return -0.5^2 + x^2 + y^2
end
ϕ3D = LevelSet(grid3D) do (x, y, z)
    return -0.5^2 + x^2 + y^2 + z^2
end

#####################################

LevelSetMethods.export_volume_mesh(ϕ2D, "testVolume2D.mesh")
LevelSetMethods.export_volume_mesh(ϕ3D, "testVolume3D.mesh")

#####################################

using MarchingCubes

LevelSetMethods.export_surface_mesh(ϕ3D, "testSurface3D.mesh"; hausd = 1.2, hmax = 0.1)
