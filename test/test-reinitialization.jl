using Test
using LinearAlgebra
using StaticArrays
import LevelSetMethods as LSM

@testset "NewtonReinitializer 2D circle" begin
    grid = LSM.CartesianGrid((-1, -1), (1, 1), (100, 100))
    exact_sdf(x) = sqrt(x[1]^2 + x[2]^2) - 0.5
    ϕ = LSM.LevelSet(x -> (x[1]^2 + x[2]^2) - 0.5^2, grid)

    @test abs(LSM.volume(ϕ) - π / 4) < 1.0e-2

    reinit = LSM.NewtonReinitializer()
    LSM.reinitialize!(ϕ, reinit)

    max_er = maximum(eachindex(grid)) do i
        abs(ϕ[i] - exact_sdf(grid[i]))
    end
    @test max_er < 1.0e-8

    # volume is preserved after reinitialization
    @test abs(LSM.volume(ϕ) - π / 4) < 1.0e-2
end

@testset "NewtonReinitializer 3D sphere" begin
    grid = LSM.CartesianGrid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (31, 31, 31))
    exact_sdf(x) = sqrt(x[1]^2 + x[2]^2 + x[3]^2) - 0.45
    ϕ = LSM.LevelSet(x -> (x[1]^2 + x[2]^2 + x[3]^2) - 0.45^2, grid)

    reinit = LSM.NewtonReinitializer(; upsample = 4)
    LSM.reinitialize!(ϕ, reinit)

    max_er = maximum(eachindex(grid)) do i
        abs(ϕ[i] - exact_sdf(grid[i]))
    end
    @test max_er < 5.0e-3
end

@testset "NewtonReinitializer in LevelSetEquation" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (33, 33))
    ϕ = LSM.LevelSet(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)

    eq = LSM.LevelSetEquation(;
        terms = (LSM.AdvectionTerm((x, t) -> @SVector [1.0, 0.0]),),
        levelset = ϕ,
        bc = LSM.PeriodicBC(),
        reinit = 2,
    )
    @test LSM.integrate!(eq, 1.0e-3, 1.0e-3) isa LSM.LevelSetEquation
end

@testset "NewtonReinitializer with SemiImplicitI2OE" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (33, 33))
    ϕ = LSM.LevelSet(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)

    eq = LSM.LevelSetEquation(;
        terms = (LSM.AdvectionTerm((x, t) -> @SVector [1.0, 0.0]),),
        levelset = ϕ,
        bc = LSM.PeriodicBC(),
        integrator = LSM.SemiImplicitI2OE(),
        reinit = 2,
    )
    @test LSM.integrate!(eq, 1.0e-3, 1.0e-3) isa LSM.LevelSetEquation
end
