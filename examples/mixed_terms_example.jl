using Test
using LevelSetMethods
using LinearAlgebra
using GLMakie

grid = CartesianGrid((-1, -1), (1, 1), (100, 100))
ϕ    = LevelSet(grid) do (x, y)
    return 1.0
end
add_circle!(ϕ, SVector(0.5, 0.0), 0.25)
add_circle!(ϕ, SVector(-0.5, 0.0), 0.25)
add_rectangle!(ϕ, SVector(0.0, 0.0), SVector(1.0, 0.1))
plot(ϕ; levels = [0])

# create terms
𝐮 = MeshField(x -> SVector(-x[2], x[1]), grid)
eq = LevelSetEquation(; terms = AdvectionTerm(𝐮), levelset = ϕ, t = 0, bc = PeriodicBC())

theme = LevelSetMethods.makie_theme()

anim = with_theme(theme) do
    eq.t = 0
    obs = Observable(eq)
    fig = Figure()
    ax = Axis(fig[1, 1])
    plot!(ax, obs)
    framerate = 30
    tf = 2π
    timestamps = range(0, tf; step = 1 / framerate)
    record(fig, joinpath(@__DIR__, "ls_intro.gif"), timestamps) do t_
        integrate!(eq, t_)
        return obs[] = eq
    end
end
