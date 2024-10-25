using LevelSetMethods
using LinearAlgebra
using GLMakie

a = (-1, -1)
b = (1, 1)
n = (100, 100)
grid = CartesianGrid(a, b, n)

d = 1
r0 = 0.5
θ0 = -π / 3
α = π / 100.0
R = [cos(α) -sin(α); sin(α) cos(α)]
M = R * [1/0.06^2 0; 0 1/(4π^2)] * R'
ϕ = LevelSet(grid) do (x, y)
    r = sqrt(x^2 + y^2)
    θ = atan(y, x)
    result = 1e30
    for i in 0:4
        θ1 = θ + (2i - 4) * π
        v = [r - r0; θ1 - θ0]
        result = min(result, sqrt(v' * M * v) - d)
    end
    return result
end

b = MeshField(x -> -1.0, grid)
term1 = CurvatureTerm(b)
terms = (term1,)
integrator = ForwardEuler(0.5)
bc = PeriodicBC()
eq = LevelSetEquation(; terms, integrator, levelset = ϕ, t = 0, bc)

theme = LevelSetMethods.makie_theme()
fig = Figure()
ax = Axis(fig[1, 1])
plot!(ax, eq)

anim = with_theme(theme) do
    eq.t = 0
    obs = Observable(eq)
    fig = Figure()
    ax = Axis(fig[1, 1])
    plot!(ax, obs)
    framerate = 120
    tf = 0.1
    timestamps = range(0, tf, framerate)
    record(fig, "curvature.gif", timestamps) do t_
        integrate!(eq, t_)
        return obs[] = eq
    end
end
