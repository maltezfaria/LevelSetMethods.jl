using Test
using LevelSetMethods
using LinearAlgebra
using Plots

nx, ny = 100, 100
x      = LinRange(-1, 1, nx)
y      = LinRange(-1, 1, ny)
hx, hy = step(x), step(y)
grid   = CartesianGrid(x, y)
ϕ      = LevelSet(grid) do (x, y)
    return 0.5 - (4 * x)^2 - y^2
end
𝐮      = MeshField(grid) do (x, y)
    return SVector(1, 0)
end
b      = MeshField(x -> -0.5, grid)
term1  = AdvectionTerm(; velocity = 𝐮)
term2  = CurvatureTerm(b)
terms  = (term1, term2)
bc     = PeriodicBC(1)
buffer = zero(ϕ)
dt     = 0.5 * (min(hx, hy))^2 # stiff
t      = 0
pgap   = 10

integrator = ForwardEuler(0.5)

anim = @animate for n in 0:200
    fill!(values(buffer), 0)
    _, t = LevelSetMethods.evolve!(buffer, integrator, ϕ, terms, bc, t)
    if n % pgap == 0
        plot(ϕ; title = "t=$t")
    end
end
gif(anim, "test.gif"; fps = 15)
