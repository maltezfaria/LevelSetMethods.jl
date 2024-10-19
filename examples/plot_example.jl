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

plot(ϕ; algorithm = :iso, isovalue = 0, alpha = 0.5)

## Following a solution over time
obs = Observable(ϕ)

fig = Figure()
ax1 = Axis3(fig[1, 1])
ax2 = Axis3(fig[1, 2])
ax3 = Axis3(fig[2, 1:2])

plot!(ax1, obs; algorithm = :iso, isovalue = 0, alpha = 0.5)
plot!(ax2, obs; algorithm = :iso, isovalue = 0, alpha = 0.3)
plot!(ax3, obs; algorithm = :iso, isovalue = 0, alpha = 1.0)
fig

for t in 0:0.01:1
    sleep(0.1)
    ϕ = LevelSet(grid, bc) do (x, y, z)
        return 0.5^2 - (x)^2 - (y)^2 - (z - t)^2
    end
    obs[] = ϕ
end
