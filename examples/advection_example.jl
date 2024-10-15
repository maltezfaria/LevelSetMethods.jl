using Test
using LevelSetMethods
using LinearAlgebra
using Plots

nx, ny = 100, 100
x = LinRange(-1, 1, nx)
y = LinRange(-1, 1, ny)
hx, hy = step(x), step(y)
grid = CartesianGrid(x, y)
bc = PeriodicBC(1)
ϕ = LevelSet(grid, bc) do (x, y)
    return 0.5 - x^2 - y^2
end
𝐮 = MeshField(grid) do (x, y)
    return SVector(1, 0)
end
term1 = AdvectionTerm(; velocity = 𝐮)
terms = (term1,)
b = zero(ϕ)
integrator = ForwardEuler(0.5)
eq = LevelSetEquation(; terms, integrator, state = ϕ, t = 0, buffer = b)

anim = @animate for n in 0:100
    tf = 0.02 * n
    integrate!(eq, tf)
    plot(eq)
end
gif(anim, "test.gif")
