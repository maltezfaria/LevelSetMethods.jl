using Test
using LevelSetMethods
using LinearAlgebra
using Plots

nx, ny = 100, 100
x = LinRange(-2, 2, nx)
y = LinRange(-2, 2, ny)
hx, hy = step(x), step(y)
grid = CartesianGrid(x, y)
bc = PeriodicBC(1)
ϕ = LevelSet(grid, bc) do (x, y)
    return 0.5 - (4x)^2 - y^2
end
b = MeshField(x -> -0.1, grid)
term = CurvatureTerm(b)
terms = (term,)
b = zero(ϕ)
integrator = ForwardEuler(0.5)
eq = LevelSetEquation(; terms, integrator, state = ϕ, t = 0, buffer = b)

anim = @animate for n in 0:50
    tf = 0.01 * n
    integrate!(eq, tf)
    plot(eq; linecolor = :black)
end
gif(anim, "test.gif")
