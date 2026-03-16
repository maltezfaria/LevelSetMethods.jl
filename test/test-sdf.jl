using LevelSetMethods
using StaticArrays
using LinearAlgebra
using Test

import LevelSetMethods: NewtonSDF, AbstractSignedDistanceFunction, update!, get_sample_points

function check_allocs(f, x)
    f(x) # warmup
    return @allocated f(x)
end

@testset "NewtonSDF 2D circle" begin
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (50, 50))
    r = 0.5
    exact_sdf(x) = norm(x) - r
    ϕ = LevelSet(exact_sdf, grid)
    sdf = NewtonSDF(ϕ; upsample = 4)

    @test sdf isa AbstractSignedDistanceFunction

    indices = CartesianIndices(grid)
    max_err = maximum(1:10:length(indices)) do k
        x = grid[indices[k]]
        abs(sdf(x) - exact_sdf(x))
    end
    @test max_err < 1.0e-5

    @test sdf(SVector(0.0, 0.0)) ≈ -r     atol = 2.0e-5
    @test sdf(SVector(r, 0.0)) ≈ 0.0   atol = 2.0e-5
    @test sdf(SVector(1.0, 0.0)) ≈ 1 - r atol = 2.0e-5

    @test check_allocs(sdf, SVector(0.25, 0.25)) == 0
end

@testset "NewtonSDF from PiecewisePolynomialInterpolation" begin
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (30, 30))
    ϕ = LevelSet(x -> norm(x) - 0.5, grid)
    itp = interpolate(ϕ, 3)
    sdf = NewtonSDF(itp; upsample = 4)
    @test sdf isa AbstractSignedDistanceFunction
    @test sdf(SVector(0.5, 0.0)) ≈ 0.0 atol = 2.0e-5
end

@testset "get_sample_points" begin
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (20, 20))
    ϕ = LevelSet(x -> norm(x) - 0.5, grid)
    sdf = NewtonSDF(ϕ; upsample = 4)
    pts = get_sample_points(sdf)
    @test pts isa Vector
    @test length(pts) > 0
    max_res = maximum(p -> abs(sdf.itp(p)), pts)
    @test max_res < 1.0e-6
end

@testset "update! rebuilds from new level set" begin
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (40, 40))
    r1, r2 = 0.5, 0.3
    ϕ = LevelSet(x -> norm(x) - r1, grid)
    sdf = NewtonSDF(ϕ; upsample = 4)
    @test sdf(SVector(r1, 0.0)) ≈ 0.0 atol = 2.0e-4

    ϕ2 = LevelSet(x -> norm(x) - r2, grid)
    update!(sdf, ϕ2)
    @test sdf(SVector(r2, 0.0)) ≈ 0.0       atol = 2.0e-4
    @test sdf(SVector(r1, 0.0)) ≈ r1 - r2   atol = 1.0e-4
end

@testset "deepcopy produces independent copy" begin
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (30, 30))
    ϕ = LevelSet(x -> norm(x) - 0.5, grid)
    sdf = NewtonSDF(ϕ; upsample = 4)
    sdf2 = deepcopy(sdf)
    @test sdf2(SVector(0.5, 0.0)) ≈ 0.0  atol = 2.0e-5
    @test sdf2(SVector(0.0, 0.0)) ≈ -0.5  atol = 2.0e-5
end

@testset "NewtonSDF 3D sphere" begin
    grid = CartesianGrid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (25, 25, 25))
    r = 0.45
    exact_sdf(x) = norm(x) - r
    ϕ = LevelSet(exact_sdf, grid)
    sdf = NewtonSDF(ϕ; upsample = 3)

    @test sdf(SVector(r, 0.0, 0.0)) ≈ 0.0 atol = 1.0e-4
    @test sdf(SVector(0.0, 0.0, 0.0)) ≈ -r   atol = 1.0e-4

    indices = CartesianIndices(grid)
    max_err = maximum(1:20:length(indices)) do k
        x = grid[indices[k]]
        abs(sdf(x) - exact_sdf(x))
    end
    @test max_err < 5.0e-3

    @test check_allocs(sdf, SVector(0.25, 0.25, 0.25)) == 0
end
