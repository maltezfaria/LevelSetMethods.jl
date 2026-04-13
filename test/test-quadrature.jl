using LevelSetMethods
using ImplicitIntegration
using Test
using StaticArrays

@testset "ImplicitIntegration Quadrature" begin
    @testset "2D circle" begin
        grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (21, 21))
        R = 0.5
        ϕ = MeshField(x -> x[1]^2 + x[2]^2 - R^2, grid; interp_order = 3)

        quads = quadrature(ϕ; order = 4, surface = false)
        total_area = sum(integrate(x -> 1.0, q) for (_, q) in quads)
        @test total_area ≈ π * R^2 atol = 1.0e-4

        quads_s = quadrature(ϕ; order = 4, surface = true)
        total_perimeter = sum(integrate(x -> 1.0, q) for (_, q) in quads_s)
        @test total_perimeter ≈ 2π * R atol = 1.0e-3
    end

    @testset "3D sphere" begin
        grid = CartesianGrid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (11, 11, 11))
        R = 0.5
        ϕ = MeshField(x -> x[1]^2 + x[2]^2 + x[3]^2 - R^2, grid; interp_order = 3)

        quads = quadrature(ϕ; order = 2, surface = false)
        total_volume = sum(integrate(x -> 1.0, q) for (_, q) in quads)
        @test total_volume ≈ 4π / 3 * R^3 atol = 1.0e-3

        quads_s = quadrature(ϕ; order = 2, surface = true)
        total_surface = sum(integrate(x -> 1.0, q) for (_, q) in quads_s)
        @test total_surface ≈ 4π * R^2 atol = 1.0e-2
    end
end
