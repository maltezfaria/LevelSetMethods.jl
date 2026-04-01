using Test
using LinearAlgebra
using StaticArrays
import LevelSetMethods as LSM
using LevelSetMethods: NewtonSDF, update!, get_sample_points, interpolate

function check_allocs(f, x)
    f(x) # warmup
    return @allocated f(x)
end

@testset "NewtonSDF" begin
    @testset "2D circle" begin
        grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (50, 50))
        r = 0.5
        exact_sdf(x) = norm(x) - r
        ϕ = LSM.LevelSet(exact_sdf, grid)
        sdf = NewtonSDF(ϕ; upsample = 4)

        @test sdf isa LSM.AbstractSignedDistanceFunction

        # spot checks: inside, on, and outside the interface
        @test sdf(SVector(0.0, 0.0)) ≈ -r atol = 2.0e-5
        @test sdf(SVector(r, 0.0)) ≈ 0.0 atol = 2.0e-5
        @test sdf(SVector(1.0, 0.0)) ≈ 1 - r atol = 2.0e-5

        # global accuracy over sampled grid points
        indices = CartesianIndices(grid)
        max_err = maximum(1:10:length(indices)) do k
            abs(sdf(grid[indices[k]]) - exact_sdf(grid[indices[k]]))
        end
        @test max_err < 1.0e-5

        @test check_allocs(sdf, SVector(0.25, 0.25)) == 0
    end

    @testset "3D sphere" begin
        grid = LSM.CartesianGrid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (25, 25, 25))
        r = 0.45
        exact_sdf(x) = norm(x) - r
        ϕ = LSM.LevelSet(exact_sdf, grid)
        sdf = NewtonSDF(ϕ; upsample = 3)

        @test sdf(SVector(r, 0.0, 0.0)) ≈ 0.0 atol = 1.0e-4
        @test sdf(SVector(0.0, 0.0, 0.0)) ≈ -r atol = 1.0e-4

        indices = CartesianIndices(grid)
        max_err = maximum(1:20:length(indices)) do k
            abs(sdf(grid[indices[k]]) - exact_sdf(grid[indices[k]]))
        end
        @test max_err < 5.0e-3

        @test check_allocs(sdf, SVector(0.25, 0.25, 0.25)) == 0
    end

    @testset "from PiecewisePolynomialInterpolant" begin
        grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (30, 30))
        ϕ = LSM.LevelSet(x -> norm(x) - 0.5, grid)
        itp = interpolate(ϕ, 3)
        sdf = NewtonSDF(itp; upsample = 4)
        @test sdf(SVector(0.5, 0.0)) ≈ 0.0 atol = 2.0e-5
    end

    @testset "get_sample_points" begin
        grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (20, 20))
        ϕ = LSM.LevelSet(x -> norm(x) - 0.5, grid)
        sdf = NewtonSDF(ϕ; upsample = 3)
        pts = get_sample_points(sdf)
        @test length(pts) > 0
        @test maximum(p -> abs(sdf.itp(p)), pts) < 1.0e-6
    end

    @testset "update!" begin
        grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (40, 40))
        r1, r2 = 0.5, 0.3
        sdf = NewtonSDF(LSM.LevelSet(x -> norm(x) - r1, grid); upsample = 4)
        @test sdf(SVector(r1, 0.0)) ≈ 0.0 atol = 2.0e-4

        update!(sdf, LSM.LevelSet(x -> norm(x) - r2, grid))
        @test sdf(SVector(r2, 0.0)) ≈ 0.0 atol = 2.0e-4
        @test sdf(SVector(r1, 0.0)) ≈ r1 - r2 atol = 1.0e-4
    end

    @testset "deepcopy" begin
        grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (30, 30))
        sdf = NewtonSDF(LSM.LevelSet(x -> norm(x) - 0.5, grid); upsample = 4)
        sdf2 = deepcopy(sdf)
        @test sdf2(SVector(0.5, 0.0)) ≈ 0.0 atol = 2.0e-5
        @test sdf2(SVector(0.0, 0.0)) ≈ -0.5 atol = 2.0e-5
    end
end

@testset "NewtonReinitializer" begin
    @testset "2D circle" begin
        grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (100, 100))
        exact_sdf(x) = sqrt(x[1]^2 + x[2]^2) - 0.5
        ϕ = LSM.LevelSet(x -> (x[1]^2 + x[2]^2) - 0.25, grid)

        @test abs(LSM.volume(ϕ) - π / 4) < 1.0e-2

        LSM.reinitialize!(ϕ, LSM.NewtonReinitializer())

        max_err = maximum(eachindex(grid)) do i
            abs(ϕ[i] - exact_sdf(grid[i]))
        end
        @test max_err < 1.0e-8
        @test abs(LSM.volume(ϕ) - π / 4) < 1.0e-2  # volume preserved
    end

    @testset "3D sphere" begin
        grid = LSM.CartesianGrid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (31, 31, 31))
        exact_sdf(x) = sqrt(x[1]^2 + x[2]^2 + x[3]^2) - 0.45
        ϕ = LSM.LevelSet(x -> (x[1]^2 + x[2]^2 + x[3]^2) - 0.45^2, grid)

        LSM.reinitialize!(ϕ, LSM.NewtonReinitializer(; upsample = 4))

        max_err = maximum(eachindex(grid)) do i
            abs(ϕ[i] - exact_sdf(grid[i]))
        end
        @test max_err < 5.0e-3
    end
end
