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

    reinit = LSM.Reinitializer()
    # check that we recover a signed distance function
    LSM.reinitialize!(ϕ, reinit)
    max_er, max_idx = findmax(eachindex(grid)) do i
        x = grid[i]
        norm(ϕ[i] - sdf(x))
    end
    @test max_er < 1.0e-8

    # check that volume is OK
    @test abs(LSM.volume(ϕ) - π / 4) < 1.0e-2
end

function _returns_true_or_false(f::Function)
    try
        f()
        return true
    catch err
        @info "Captured exception in review check" exception = err
        return false
    end
end

@testset "SemiImplicitI2OE should accept Reinitializer in eq.reinit" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (33, 33))
    ϕ = LSM.LevelSet(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)

    eq = LSM.LevelSetEquation(;
        terms = (LSM.AdvectionTerm((x, t) -> @SVector [1.0, 0.0]),),
        levelset = ϕ,
        bc = LSM.PeriodicBC(),
        integrator = LSM.SemiImplicitI2OE(),
        reinit = LSM.Reinitializer(; reinit_freq = 2),
    )

    @test _returns_true_or_false(() -> LSM.integrate!(eq, 1e-3, 1e-3))
end

@testset "3D Newton reinitialization should run and recover SDF" begin
    grid = LSM.CartesianGrid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (31, 31, 31))
    sdf(x) = sqrt(x[1]^2 + x[2]^2 + x[3]^2) - 0.45
    ϕ = LSM.LevelSet(x -> (x[1]^2 + x[2]^2 + x[3]^2) - 0.45^2, grid)
    reinit = LSM.Reinitializer(; upsample = 4, maxiters = 20, xtol = 1e-8, ftol = 1e-8)

    @test _returns_true_or_false(() -> LSM.reinitialize!(ϕ, reinit))

    max_er = maximum(eachindex(grid)) do i
        x = grid[i]
        abs(ϕ[i] - sdf(x))
    end
    @test max_er < 5e-3
end
