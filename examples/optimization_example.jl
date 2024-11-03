using LevelSetMethods
using LinearAlgebra

a = (-1.0, -1.0)
b = (+1.0, +1.0)
n = (50, 50)
grid = CartesianGrid(a, b, n)

ϕ = LevelSetMethods.star(grid)
# ϕ = LevelSetMethods.dumbbell(grid)

term1 = NormalMotionTerm(MeshField(X -> 0.0, grid))
term2 = CurvatureTerm(MeshField(X -> -1.0, grid))
terms = (term1, term2)

bc = NeumannGradientBC()
integrator = ForwardEuler(0.5)
eq = LevelSetEquation(; terms, integrator, levelset = ϕ, t = 0, bc)

using GLMakie

nit = 200
anim = with_theme(LevelSetMethods.makie_theme()) do
    # parameters for augmented Lagrangian method
    λ, μ = 0.0, 0.1
    c = 1.1
    V0 = 0.5
    R0 = sqrt(V0 / π)
    P0 = 2π * R0
    δ = 0.25

    eq.t = 0
    obs = Observable(eq)
    fig = Figure()
    ax = Axis(fig[1, 1])
    plot!(ax, obs)
    arc!([0; 0], R0, 0, 2π)

    record(fig, "optimization.gif", 1:nit) do it
        P = LevelSetMethods.Perimeter(ϕ)
        V = LevelSetMethods.Volume(ϕ)

        println("it = ", it, "; P = ", P, "->", P0, "; V = ", V, " vs. ", V0)

        term1 = NormalMotionTerm(MeshField(X -> -(λ + μ * (V - V0)), grid))
        eq.terms = (term1, term2)

        τ = δ * LevelSetMethods.compute_cfl(eq.terms, eq.state, eq.t)
        integrate!(eq, eq.t + τ)

        λ += μ * (V - V0)
        μ *= c
        return obs[] = eq
    end
end
