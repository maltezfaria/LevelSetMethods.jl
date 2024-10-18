using LevelSetMethods
using LinearAlgebra
using GLMakie

## 2D
nx, ny = 100, 100
x = range(-1, 1, nx)
y = range(-1, 1, ny)
grid = CartesianGrid(x, y)
bc = PeriodicBC(1)
ϕ = LevelSet(grid, bc) do (x, y)
    return 0.5^2 - x^2 - y^2
end

plot(ϕ; levels = [0])

## 3D
nx, ny, nz = 100, 100, 100
x = range(-1, 1, nx)
y = range(-1, 1, ny)
z = range(-1, 1, nz)
grid = CartesianGrid(x, y, z)
bc = PeriodicBC(1)
ϕ = LevelSet(grid, bc) do (x, y, z)
    return 0.5^2 - x^2 - y^2 - z^2
end

volume(ϕ; algorithm = :iso, isovalue = 0, alpha = 0.5)

## Following a solution over time
obs = Observable(ϕ)

volume(obs; algorithm = :iso, isovalue = 0, alpha = 0.5)

for t in 0:0.01:1
    sleep(0.01)
    ϕ = LevelSet(grid, bc) do (x, y, z)
        return 0.5^2 - (x - t)^2 - (y - t)^2 - (z - t)^2
    end
    obs[] = ϕ
end
