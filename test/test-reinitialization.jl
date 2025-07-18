using Test
import LevelSetMethods as LSM
using LinearAlgebra

@testset "Reinitialization" begin
    grid = CartesianGrid((-1, -1), (1, 1), (100, 100))

    # A level set that is not a signed distance function
    ϕ_nonsdf = LevelSet(x -> (x[1]^2 + x[2]^2) - 0.5^2, grid)
    # ϕ_nonsdf = LevelSet(x -> sqrt((x[1]^2 + x[2]^2)) - 0.5, grid)

    eq = LevelSetEquation(;
        terms = (), # No terms, just for holding the state
        levelset = deepcopy(ϕ_nonsdf),
        bc = NeumannGradientBC(),
    )

    @test norm(LSM.grad_norm(eq) .- 1, Inf) > 0.5
    @test abs(LSM.volume(ϕ_nonsdf) - π / 4) < 1e-2
    reinitialize!(eq; tol = 1e-1, max_iters = 200, verbose = false)
    @test norm(LSM.grad_norm(eq) .- 1, Inf) < 0.5
    @test abs(LSM.volume(ϕ_nonsdf) - π / 4) < 1e-2
end
