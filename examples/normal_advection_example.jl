using Test
using LevelSetMethods
using LinearAlgebra
using Plots

nx, ny = 100, 100
x      = LinRange(-1, 1, nx)
y      = LinRange(-1, 1, ny)
hx, hy = step(x), step(y)
grid   = CartesianGrid(x, y)
bc     = PeriodicBC(2)
ϕ      = LevelSet(grid, bc) do (x, y)
    return 1.0
end
add_circle!(ϕ, SVector(0.5, 0.0), 0.25)
add_circle!(ϕ, SVector(-0.5, 0.0), 0.25)
add_rectangle!(ϕ, SVector(0.0, 0.0), SVector(1.0, 0.1))
plot(ϕ)
v = MeshField(grid) do (x, y)
    return -1.0
end
term1 = NormalMotionTerm(v)
terms = (term1,)
b = zero(ϕ)
integrator = ForwardEuler(0.5)
eq = LevelSetEquation(; terms, integrator, state = ϕ, t = 0, buffer = b)

dt = 0.01
anim = @animate for n in 0:20
    tf = dt * n
    integrate!(eq, tf)
    plot(eq; linecolor = :black)
end
gif(anim, "test.gif")
