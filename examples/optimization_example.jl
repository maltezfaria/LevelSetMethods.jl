using LevelSetMethods
using LinearAlgebra
using GLMakie
# using MarchingCubes

a = (-1.0, -1.0)
b = (+1.0, +1.0)
n = (100, 100)
grid = CartesianGrid(a, b, n)

function Volume(ϕ::LevelSet)
    m = LevelSetMethods.mesh(ϕ)
    vol = prod(meshsize(m))
    N = sum(values(ϕ) .< 0)
    return N * vol
end

# s, ρ, σ = 0.3, 5.0, 1.5
# ϕ = LevelSet(grid) do (x, y)
#     norm = sqrt(x^2 + y^2)
#     return norm - s * (cos(ρ * atan(y / x)) + σ)
# end
ϕ = LevelSet(grid) do (x, y)
    return (x^2 + y^2) - 0.5^2
end

# term1 = NormalMotionTerm(MeshField(X -> X[1], grid))
# term1 = NormalMotionTerm(MeshField(X -> 1.0, grid))
term2 = CurvatureTerm(MeshField(X -> -1.0, grid))
terms = (term2,)

bc = NeumannBC()
integrator = ForwardEuler(0.5)
eq = LevelSetEquation(; terms, integrator, levelset = ϕ, t = 0, bc)

# fig = Figure()
# ax = Axis(fig[1, 1])
# plot!(ax, eq)

nit = 1

anim = with_theme(LevelSetMethods.makie_theme()) do
    # parameters for augmented Lagrangian method
    λ, μ = 0.0, 0.0
    c = 1.5
    V0 = 0.5
    Ropt = sqrt(V0 / π)
    Δ = 2.0 * min(meshsize(grid)...)

    eq.t = 0
    obs = Observable(eq)
    fig = Figure()
    ax = Axis(fig[1, 1])
    plot!(ax, obs)

    record(fig, "optimization.gif", 1:nit) do it
        # update vector field for curvature
        V = Volume(ϕ)
        println("it = ", it, "; V = ", V)
        # term2 = CurvatureTerm(MeshField(X -> -(λ + μ*(V-V0)), grid))
        # eq.terms = (term1, term2)

        κ = curvature(eq.state)
        𝛉 = 1.0 + (λ + μ*(V-V0))*κ
        norminf = maximum(abs.(𝛉))
        κ = 0.0
        𝛉 = 1.0 + (λ + μ * (V - V0)) * κ
        norminf = abs(𝛉)
        τ = 0.0002#Δ / norminf
        # println(τ)
        # println(Δ)
        # println(norminf)
        integrate!(eq, eq.t + τ)

        λ += μ * (V - V0)
        μ *= c
        return obs[] = eq
    end
end
