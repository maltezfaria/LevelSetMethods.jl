using LevelSetMethods
using LinearAlgebra
using GLMakie

nx, ny = 100, 100
x = range(-1, 1, nx)
y = range(-1, 1, ny)
hx, hy = step(x), step(y)
grid = CartesianGrid(x, y)
bc = PeriodicBC(3)
ϕ = LevelSet(grid, bc) do (x, y)
    return 0.5^2 - x^2 - y^2
end
𝐮 = MeshField(grid) do (x, y)
    return SVector(1, 0)
end
term1 = AdvectionTerm(; velocity = 𝐮)
terms = (term1,)
b = zero(ϕ)
integrator = ForwardEuler(0.5)
eq = LevelSetEquation(; terms, integrator, state = ϕ, t = 0, buffer = b)

##
# Load a default theme
theme = LevelSetMethods.makie_theme()

anim = with_theme(theme) do
    eq.t = 0
    obs = Observable(eq)
    fig = Figure()
    ax = Axis(fig[1, 1])
    plot!(ax, obs)
    framerate = 30
    tf = 2
    timestamps = range(0, tf, tf * framerate)
    record(fig, "advection.gif", timestamps) do t_
        integrate!(eq, t_)
        return obs[] = eq
    end
end
