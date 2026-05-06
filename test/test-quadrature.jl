using LevelSetMethods
using ImplicitIntegration
using StaticArrays
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

    @testset "h-convergence (2D circle, odd interp_order)" begin
        # Use the exact SDF ‖x‖-R as input (non-polynomial) so the interpolant
        # approximation error actually governs the quadrature convergence.
        # The polynomial x₁²+x₂²-R² is exactly representable at any order and
        # would hide the h-dependence entirely.
        #
        # Odd interp_order k is tested: the even-order LS stencil is not a true
        # interpolant and gives degraded perimeter rates (only O(h^k) instead of
        # O(h^(k+1))) so it is excluded here.
        #
        # quad_order = k+1: the quadrature rule must be at least as accurate as the
        # interpolant to avoid being the bottleneck (quad ≤ k caps the perimeter rate).
        _orders(errs, Ns) = [log(errs[i] / errs[i + 1]) / log(Ns[i + 1] / Ns[i]) for i in 1:(length(Ns) - 1)]
        R = 0.5
        Ns = [10, 20, 40, 80]
        for k in [3, 5]
            q = k + 1
            @testset "interp_order=$k, quad_order=$q → O(h^$(k + 1))" begin
                area_errs = map(Ns) do N
                    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (N, N))
                    ϕ = MeshField(x -> norm(x) - R, grid; interp_order = k)
                    abs(_total(quadrature(ϕ; order = q, surface = false)) - π * R^2)
                end
                perim_errs = map(Ns) do N
                    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (N, N))
                    ϕ = MeshField(x -> norm(x) - R, grid; interp_order = k)
                    abs(_total(quadrature(ϕ; order = q, surface = true)) - 2π * R)
                end
                @test all(≥(k + 0.5), _orders(area_errs, Ns))
                @test all(≥(k + 0.5), _orders(perim_errs, Ns))
            end
        end
    end

    @testset "Narrow band" begin
        R = 0.5
        grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (41, 41))
        ϕ_full = MeshField(x -> x[1]^2 + x[2]^2 - R^2, grid; interp_order = 3)
        ϕ_nb = NarrowBandMeshField(ϕ_full, 0.3; reinitialize = false)
        # surface=false (volume integral) is not supported on NarrowBandMeshField
        @test_throws ErrorException quadrature(ϕ_nb; order = 4, surface = false)
        # surface=true (surface integral) works: only interface cells are needed
        @test _total(quadrature(ϕ_full; order = 4, surface = true)) ≈
            _total(quadrature(ϕ_nb; order = 4, surface = true)) rtol = 1.0e-10
    end
end
