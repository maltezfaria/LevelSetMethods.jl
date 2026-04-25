using LevelSetMethods
using ImplicitIntegration
using Test

_total(quads, f = x -> 1.0) = sum(integrate(f, q) for (_, q) in quads)

@testset "ImplicitIntegration Quadrature" begin
    @testset "2D circle" begin
        R = 0.5
        grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (21, 21))
        ϕ = MeshField(x -> x[1]^2 + x[2]^2 - R^2, grid; interp_order = 3)
        @test _total(quadrature(ϕ; order = 4, surface = false)) ≈ π * R^2 atol = 1.0e-4
        @test _total(quadrature(ϕ; order = 4, surface = true)) ≈ 2π * R atol = 1.0e-3
    end

    @testset "2D ellipse" begin
        a, b = 0.6, 0.3
        grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (41, 41))
        ϕ = MeshField(x -> (x[1] / a)^2 + (x[2] / b)^2 - 1.0, grid; interp_order = 3)
        h = ((a - b) / (a + b))^2
        peri_approx = π * (a + b) * (1 + 3h / (10 + sqrt(4 - 3h)))
        @test _total(quadrature(ϕ; order = 4, surface = false)) ≈ π * a * b rtol = 1.0e-3
        @test _total(quadrature(ϕ; order = 4, surface = true)) ≈ peri_approx rtol = 1.0e-3
    end

    @testset "3D sphere" begin
        R = 0.5
        grid = CartesianGrid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (11, 11, 11))
        ϕ = MeshField(x -> x[1]^2 + x[2]^2 + x[3]^2 - R^2, grid; interp_order = 3)
        @test _total(quadrature(ϕ; order = 2, surface = false)) ≈ 4π / 3 * R^3 atol = 1.0e-3
        @test _total(quadrature(ϕ; order = 2, surface = true)) ≈ 4π * R^2 atol = 1.0e-2
    end

    @testset "3D ellipsoid" begin
        a, b, c = 0.61, 0.37, 0.29
        grid = CartesianGrid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (21, 21, 21))
        ϕ = MeshField(grid; interp_order = 3) do x
            (x[1] / a)^2 + (x[2] / b)^2 + (x[3] / c)^2 - 1.0
        end
        @test _total(quadrature(ϕ; order = 3, surface = false)) ≈ (4 / 3) * π * a * b * c rtol = 1.0e-3
    end

    @testset "Narrow band" begin
        R = 0.5
        grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (41, 41))
        ϕ_full = MeshField(x -> x[1]^2 + x[2]^2 - R^2, grid; interp_order = 3)
        ϕ_nb = NarrowBandMeshField(ϕ_full, 0.3; reinitialize = false)
        @test _total(quadrature(ϕ_full; order = 4, surface = false)) ≈
            _total(quadrature(ϕ_nb; order = 4, surface = false)) rtol = 1.0e-10
        @test _total(quadrature(ϕ_full; order = 4, surface = true)) ≈
            _total(quadrature(ϕ_nb; order = 4, surface = true)) rtol = 1.0e-10
    end
end
