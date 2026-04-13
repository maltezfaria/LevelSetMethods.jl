using LevelSetMethods
using ImplicitIntegration
using Test
using StaticArrays

# Helper: total integral over a quadrature result vector
_total(quads, f = x -> 1.0) = sum(integrate(f, q) for (_, q) in quads)

@testset "Quadrature Stress Tests" begin
    @testset "Ellipse 2D (anisotropic)" begin
        a, b = 0.6, 0.3
        grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (41, 41))
        ϕ = MeshField(x -> (x[1] / a)^2 + (x[2] / b)^2 - 1.0, grid; interp_order = 3)

        area = _total(quadrature(ϕ; order = 4, surface = false))
        @test area ≈ π * a * b rtol = 1.0e-3

        # Ramanujan's first approximation to the ellipse perimeter
        h = ((a - b) / (a + b))^2
        peri_approx = π * (a + b) * (1 + 3h / (10 + sqrt(4 - 3h)))
        peri = _total(quadrature(ϕ; order = 4, surface = true))
        @test peri ≈ peri_approx rtol = 1.0e-3
    end

    @testset "Ellipsoid 3D (anisotropic)" begin
        a, b, c = 0.6, 0.4, 0.3
        grid = CartesianGrid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (21, 21, 21))
        ϕ = MeshField(grid; interp_order = 3) do x
            (x[1] / a)^2 + (x[2] / b)^2 + (x[3] / c)^2 - 1.0
        end

        vol = _total(quadrature(ϕ; order = 3, surface = false))
        @test vol ≈ (4 / 3) * π * a * b * c rtol = 5.0e-3
    end

    @testset "Torus 3D (non-convex topology)" begin
        # ϕ = (sqrt(x²+y²) - R)² + z² - r²
        # V = 2π²Rr²,  S = 4π²Rr
        R, r = 0.6, 0.2
        grid = CartesianGrid((-1.0, -1.0, -0.5), (1.0, 1.0, 0.5), (31, 31, 17))
        ϕ = MeshField(grid; interp_order = 3) do x
            (sqrt(x[1]^2 + x[2]^2) - R)^2 + x[3]^2 - r^2
        end

        vol = _total(quadrature(ϕ; order = 3, surface = false))
        @test vol ≈ 2π^2 * R * r^2 rtol = 1.0e-2

        surf = _total(quadrature(ϕ; order = 3, surface = true))
        @test surf ≈ 4π^2 * R * r rtol = 1.0e-2
    end

    @testset "h-refinement convergence (2D circle, area)" begin
        # Off-grid center so the interface never aligns with grid nodes —
        # otherwise ImplicitIntegration hits its degenerate-cell fallback and
        # the convergence rate stalls.
        R = 0.5
        cx, cy = 0.04321, -0.07654
        exact = π * R^2
        ns = (11, 21, 41, 81)
        errs = Float64[]
        for n in ns
            grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (n, n))
            ϕ = MeshField(x -> (x[1] - cx)^2 + (x[2] - cy)^2 - R^2, grid; interp_order = 3)
            area = _total(quadrature(ϕ; order = 4, surface = false))
            push!(errs, abs(area - exact))
        end
        # Order=4 quadrature on a degree-3 Bernstein interpolant: each halving
        # of h should reduce the error by at least ~4× in the asymptotic regime.
        @test errs[2] < errs[1] / 4
        @test errs[3] < errs[2] / 4
        @test errs[4] < errs[3] / 4
        @test errs[end] < 1.0e-7
    end

    @testset "h-refinement convergence (2D circle, perimeter)" begin
        R = 0.5
        cx, cy = 0.04321, -0.07654
        exact = 2π * R
        ns = (11, 21, 41, 81)
        errs = Float64[]
        for n in ns
            grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (n, n))
            ϕ = MeshField(x -> (x[1] - cx)^2 + (x[2] - cy)^2 - R^2, grid; interp_order = 3)
            peri = _total(quadrature(ϕ; order = 4, surface = true))
            push!(errs, abs(peri - exact))
        end
        @test errs[2] < errs[1] / 4
        @test errs[3] < errs[2] / 4
        @test errs[4] < errs[3] / 4
        @test errs[end] < 1.0e-7
    end

    @testset "Off-grid placement (no node hits the interface)" begin
        # Center is irrational-ish so the interface never aligns with a grid node.
        R = 0.5
        cx, cy = 0.04321, -0.07654
        grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (41, 41))
        ϕ = MeshField(x -> (x[1] - cx)^2 + (x[2] - cy)^2 - R^2, grid; interp_order = 3)

        area = _total(quadrature(ϕ; order = 4, surface = false))
        @test area ≈ π * R^2 rtol = 1.0e-3

        peri = _total(quadrature(ϕ; order = 4, surface = true))
        @test peri ≈ 2π * R rtol = 1.0e-3
    end

    @testset "Narrow band agrees with full grid" begin
        # The narrow-band quadrature should match the full-grid result whenever the
        # interface is well inside the band.
        R = 0.5
        grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (41, 41))
        ϕ_full = MeshField(x -> x[1]^2 + x[2]^2 - R^2, grid; interp_order = 3)
        ϕ_nb = NarrowBandMeshField(ϕ_full, 0.3; reinitialize = false)

        area_full = _total(quadrature(ϕ_full; order = 4, surface = false))
        area_nb = _total(quadrature(ϕ_nb; order = 4, surface = false))
        @test area_full ≈ area_nb rtol = 1.0e-10

        peri_full = _total(quadrature(ϕ_full; order = 4, surface = true))
        peri_nb = _total(quadrature(ϕ_nb; order = 4, surface = true))
        @test peri_full ≈ peri_nb rtol = 1.0e-10
    end

end
