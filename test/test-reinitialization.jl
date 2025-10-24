using Test
using LinearAlgebra
using Interpolations
using NearestNeighbors
import LevelSetMethods as LSM

@testset "Newton Reinitialization" begin
    grid = LSM.CartesianGrid((-1, -1), (1, 1), (100, 100))

    # A level set that is not a signed distance function
    sdf = (x) -> sqrt(x[1]^2 + x[2]^2) - 0.5
    ϕ = LSM.LevelSet(x -> (x[1]^2 + x[2]^2) - 0.5^2, grid)
    eq = LSM.LevelSetEquation(;
        terms = (), # No terms, just for holding the state
        levelset = ϕ,
        bc = LSM.NeumannGradientBC(),
    )
    @test abs(LSM.volume(ϕ) - π / 4) < 1.0e-2

    # check that we recover a signed distance function
    LSM.reinitialize!(eq; upsample = 4, maxiters = 20, xtol = 1.0e-10, ftol = 1.0e-10)
    max_er, max_idx = findmax(eachindex(grid)) do i
        x = grid[i]
        norm(ϕ[i] - sdf(x))
    end
    @test max_er < 1.0e-8

    # check that volume is OK
    @test abs(LSM.volume(ϕ) - π / 4) < 1.0e-2
end
