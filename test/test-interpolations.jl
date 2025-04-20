using Test
using LevelSetMethods
using LinearAlgebra
using StaticArrays
using Interpolations

@testset "Interpolations extension" begin
    @testset "2D" begin
        a, b = (-2, -2), (2, 2)
        grid = CartesianGrid(a, b, (100, 100))
        f = x -> x[1]^2 + x[2]^2 - 1
        ϕ = LevelSet(f, grid) # a disk
        itp = interpolate(ϕ)
        # test value at a few points
        @test itp(0.1, 0.2) ≈ f((0.1, 0.2))
        @test itp(-0.4, 0.5) ≈ f((-0.4, 0.5))
        @test itp(0.5, -0.4) ≈ f((0.5, -0.4))
    end

    @testset "3D" begin
        a, b = (-2, -2, -2), (2, 2, 2)
        grid = CartesianGrid(a, b, (100, 100, 100))
        f = x -> x[1]^2 + x[2]^2 + x[3]^2 - 1
        ϕ = LevelSet(f, grid) # a sphere
        itp = interpolate(ϕ)
        # test value at a few points
        @test itp(0.1, 0.2, 0.3) ≈ f((0.1, 0.2, 0.3))
        @test itp(-0.4, 0.5, -0.6) ≈ f((-0.4, 0.5, -0.6))
        @test itp(0.5, -0.4, 0.6) ≈ f((0.5, -0.4, 0.6))
    end
end
