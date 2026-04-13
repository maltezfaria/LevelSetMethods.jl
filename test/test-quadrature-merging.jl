using LevelSetMethods
using ImplicitIntegration
using Test
using StaticArrays

@testset "ImplicitIntegration Quadrature Merging" begin
    @testset "2D circle (Full Grid)" begin
        grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (21, 21))
        R = 0.5
        ϕ = MeshField(x -> (x[1] - 0.049)^2 + (x[2] - 0.049)^2 - R^2, grid; interp_order = 3)

        quads_base = quadrature(ϕ; order = 4, surface = false)
        area_base = sum(integrate(x -> 1.0, q) for (_, q) in quads_base)

        quads_merged = quadrature(ϕ; order = 4, surface = false, min_mass_fraction = 0.2)
        area_merged = sum(integrate(x -> 1.0, q) for (_, q) in quads_merged)

        @test area_merged ≈ area_base atol = 1.0e-4
        @test length(quads_merged) < length(quads_base)

        M = maximum(sum(q.weights) for (_, q) in quads_base)
        masses = [sum(q.weights) for (_, q) in quads_merged]
        @test minimum(masses) >= 0.2 * M
    end

    @testset "2D circle (Narrow Band)" begin
        grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (21, 21))
        R = 0.5
        ϕ_full = MeshField(x -> (x[1] - 0.049)^2 + (x[2] - 0.049)^2 - R^2, grid; interp_order = 3)
        ϕ_nb = NarrowBandMeshField(ϕ_full, 0.2; reinitialize = false)

        quads_base = quadrature(ϕ_nb; order = 4, surface = false)
        area_base = sum(integrate(x -> 1.0, q) for (_, q) in quads_base)

        quads_merged = quadrature(ϕ_nb; order = 4, surface = false, min_mass_fraction = 0.2)
        area_merged = sum(integrate(x -> 1.0, q) for (_, q) in quads_merged)

        @test area_merged ≈ area_base atol = 1.0e-4
        @test length(quads_merged) < length(quads_base)

        M = maximum(sum(q.weights) for (_, q) in quads_base)
        masses = [sum(q.weights) for (_, q) in quads_merged]
        @test minimum(masses) >= 0.2 * M
    end

    @testset "3D sphere (Full Grid)" begin
        grid = CartesianGrid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (11, 11, 11))
        R = 0.5
        ϕ = MeshField(x -> (x[1] - 0.049)^2 + (x[2] - 0.049)^2 + (x[3] - 0.049)^2 - R^2, grid; interp_order = 3)

        quads_base = quadrature(ϕ; order = 2, surface = false)
        vol_base = sum(integrate(x -> 1.0, q) for (_, q) in quads_base)

        quads_merged = quadrature(ϕ; order = 2, surface = false, min_mass_fraction = 0.2)
        vol_merged = sum(integrate(x -> 1.0, q) for (_, q) in quads_merged)

        @test vol_merged ≈ vol_base atol = 2.0e-3
        @test length(quads_merged) < length(quads_base)

        M = maximum(sum(q.weights) for (_, q) in quads_base)
        masses = [sum(q.weights) for (_, q) in quads_merged]
        @test minimum(masses) >= 0.2 * M
    end

    @testset "3D sphere (Narrow Band)" begin
        grid = CartesianGrid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (11, 11, 11))
        R = 0.5
        ϕ_full = MeshField(x -> (x[1] - 0.049)^2 + (x[2] - 0.049)^2 + (x[3] - 0.049)^2 - R^2, grid; interp_order = 3)
        ϕ_nb = NarrowBandMeshField(ϕ_full, 0.4; reinitialize = false)

        quads_base = quadrature(ϕ_nb; order = 2, surface = false)
        vol_base = sum(integrate(x -> 1.0, q) for (_, q) in quads_base)

        quads_merged = quadrature(ϕ_nb; order = 2, surface = false, min_mass_fraction = 0.2)
        vol_merged = sum(integrate(x -> 1.0, q) for (_, q) in quads_merged)

        @test vol_merged ≈ vol_base atol = 2.0e-3
        @test length(quads_merged) < length(quads_base)

        M = maximum(sum(q.weights) for (_, q) in quads_base)
        masses = [sum(q.weights) for (_, q) in quads_merged]
        @test minimum(masses) >= 0.2 * M
    end
end
