using LevelSetMethods
using StaticArrays
using LinearAlgebra
using Test

@testset "Interpolation Module" begin

    @testset "Standalone BernsteinPolynomial" begin
        @testset "2D Evaluation" begin
            lc, hc = SVector(0.0, 0.0), SVector(1.0, 1.0)
            c = zeros(SMatrix{3, 3})
            c = setindex(c, 1.0, 1, 1)
            p = LevelSetMethods.BernsteinPolynomial(c, lc, hc)
            x = SVector(0.1, 0.2)
            @test p(x) ≈ (1 - x[1])^2 * (1 - x[2])^2
        end

        @testset "3D Evaluation" begin
            lc, hc = SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0)
            c = zeros(SArray{Tuple{2, 2, 2}, Float64, 3, 8})
            c = setindex(c, 1.0, 2, 2, 2)
            p = LevelSetMethods.BernsteinPolynomial(c, lc, hc)
            x = SVector(0.1, 0.2, 0.3)
            @test p(x) ≈ 0.1 * 0.2 * 0.3
        end
    end

    # Helper function to get clean allocation counts
    function check_allocs(itp, x)
        itp(x) # Warmup
        return @allocated itp(x)
    end

    @testset "Mesh Interpolation (2D)" begin
        grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (21, 21))
        f(x) = x[1]^2 + 2 * x[2]^2 - 0.5
        grad_f(x) = SVector(2 * x[1], 4 * x[2])
        hess_f(x) = SMatrix{2, 2}(2.0, 0.0, 0.0, 4.0)

        ϕ = MeshField(f, grid; interp_order = 3)
        x_test = SVector(0.15, -0.25)

        @test ϕ(x_test) ≈ f(x_test) atol = 1.0e-12
        I = LevelSetMethods.compute_index(ϕ, x_test)
        p = LevelSetMethods.make_interpolant(ϕ, I)
        @test LevelSetMethods.gradient(p, x_test) ≈ grad_f(x_test) atol = 1.0e-12

        @test check_allocs(ϕ, x_test) == 0
    end

    @testset "Least Squares Approximation (K=2)" begin
        grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (21, 21))
        f(x) = x[1]^2 + 2 * x[2]^2 - 0.5
        ϕ = MeshField(f, grid; interp_order = 2)
        x_test = SVector(0.15, -0.25)
        @test ϕ(x_test) ≈ f(x_test) atol = 1.0e-12
        I = LevelSetMethods.compute_index(ϕ, x_test)
        p = LevelSetMethods.make_interpolant(ϕ, I)
        @test LevelSetMethods.gradient(p, x_test) ≈ SVector(2 * 0.15, 4 * (-0.25)) atol = 1.0e-12
        @test check_allocs(ϕ, x_test) == 0
    end

    @testset "Mesh Interpolation (3D)" begin
        grid = CartesianGrid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (11, 11, 11))
        f(x) = x[1]^2 + x[2]^2 + x[3]^2 - 0.5
        grad_f(x) = SVector(2 * x[1], 2 * x[2], 2 * x[3])

        ϕ = MeshField(f, grid; interp_order = 3)
        x_test = SVector(0.1, -0.2, 0.3)

        @test ϕ(x_test) ≈ f(x_test) atol = 1.0e-12
        I = LevelSetMethods.compute_index(ϕ, x_test)
        p = LevelSetMethods.make_interpolant(ϕ, I)
        @test LevelSetMethods.gradient(p, x_test) ≈ grad_f(x_test) atol = 1.0e-12

        @test check_allocs(ϕ, x_test) == 0
    end

    @testset "Convex Hull & proven_empty" begin
        grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (20, 20))
        f(x) = x[1]
        ϕ = MeshField(f, grid; interp_order = 3)

        I_inside = CartesianIndex(1, 1)
        m, M = LevelSetMethods.cell_extrema(ϕ, I_inside)
        @test M < 0
        @test LevelSetMethods.proven_empty(ϕ, I_inside; surface = true) # no surface
        @test !LevelSetMethods.proven_empty(ϕ, I_inside; surface = false) # fully inside

        I_outside = CartesianIndex(20, 20)
        m, M = LevelSetMethods.cell_extrema(ϕ, I_outside)
        @test m > 0
        @test LevelSetMethods.proven_empty(ϕ, I_outside; surface = true)
        @test LevelSetMethods.proven_empty(ϕ, I_outside; surface = false)

        # cell crossing the interface
        I_interface = CartesianIndex(10, 4)
        m, M = LevelSetMethods.cell_extrema(ϕ, I_interface)
        @test m < 0 && M > 0
        @test !LevelSetMethods.proven_empty(ϕ, I_interface; surface = true)
        @test !LevelSetMethods.proven_empty(ϕ, I_interface; surface = false)
    end
end
