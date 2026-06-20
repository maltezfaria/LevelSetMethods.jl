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

        ϕ = InterpolatedField(MeshField(f, grid), 3)
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
        ϕ = InterpolatedField(MeshField(f, grid), 2)
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

        ϕ = InterpolatedField(MeshField(f, grid), 3)
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
        ϕ = InterpolatedField(MeshField(f, grid), 3)

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

    @testset "Continuous Unified Interface (gradient, hessian, fused)" begin
        grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (21, 21))
        f(x) = x[1]^2 + 2 * x[2]^2 - 0.5
        grad_f(x) = SVector(2 * x[1], 4 * x[2])
        hess_f(x) = SMatrix{2, 2}(2.0, 0.0, 0.0, 4.0)

        ϕ = InterpolatedField(MeshField(f, grid), 3)
        @test ϕ isa InterpolatedField

        # Test explicit wrapper construction
        ϕ_disc = MeshField(f, grid)
        @test !(ϕ_disc isa InterpolatedField)
        ϕ_wrap = InterpolatedField(ϕ_disc, 3)
        @test ϕ_wrap isa InterpolatedField

        x_test = SVector(0.15, -0.25)

        # 1. Test local_interpolant
        p = LevelSetMethods.local_interpolant(ϕ, x_test)
        @test p isa LevelSetMethods.BernsteinPolynomial

        # 2. Test continuous gradient
        @test LevelSetMethods.gradient(ϕ, x_test) ≈ grad_f(x_test) atol = 1.0e-12
        @test LevelSetMethods.gradient(ϕ, (0.15, -0.25)) ≈ grad_f(x_test) atol = 1.0e-12

        # 3. Test continuous Hessian
        @test LevelSetMethods.hessian(ϕ, x_test) ≈ hess_f(x_test) atol = 1.0e-12
        @test LevelSetMethods.hessian(ϕ, (0.15, -0.25)) ≈ hess_f(x_test) atol = 1.0e-12

        # 4. Test value_and_gradient
        val, grad = LevelSetMethods.value_and_gradient(ϕ, x_test)
        @test val ≈ f(x_test) atol = 1.0e-12
        @test grad ≈ grad_f(x_test) atol = 1.0e-12

        # 5. Test value_gradient_hessian
        val, grad, hess = LevelSetMethods.value_gradient_hessian(ϕ, x_test)
        @test val ≈ f(x_test) atol = 1.0e-12
        @test grad ≈ grad_f(x_test) atol = 1.0e-12
        @test hess ≈ hess_f(x_test) atol = 1.0e-12

        # 6. Test zero allocations. Warm up each call (fills the per-task scratch and the
        # cell cache) before measuring, then check steady-state evaluation never allocates.
        I_test = LevelSetMethods.compute_index(ϕ, x_test)
        ϕ(x_test)
        LevelSetMethods.make_interpolant(ϕ, I_test)
        LevelSetMethods.local_interpolant(ϕ, x_test)
        LevelSetMethods.proven_empty(ϕ, I_test)
        LevelSetMethods.gradient(ϕ, x_test)
        LevelSetMethods.hessian(ϕ, x_test)
        LevelSetMethods.value_and_gradient(ϕ, x_test)
        LevelSetMethods.value_gradient_hessian(ϕ, x_test)

        @test (@allocated ϕ(x_test)) == 0
        @test (@allocated LevelSetMethods.make_interpolant(ϕ, I_test)) == 0
        @test (@allocated LevelSetMethods.local_interpolant(ϕ, x_test)) == 0
        @test (@allocated LevelSetMethods.proven_empty(ϕ, I_test)) == 0
        @test (@allocated LevelSetMethods.gradient(ϕ, x_test)) == 0
        @test (@allocated LevelSetMethods.hessian(ϕ, x_test)) == 0
        @test (@allocated LevelSetMethods.value_and_gradient(ϕ, x_test)) == 0
        @test (@allocated LevelSetMethods.value_gradient_hessian(ϕ, x_test)) == 0
    end

    @testset "Cache invalidation on mutation" begin
        # Mutating the field must invalidate the cached cell coefficients of every task
        # (generation counter), so a subsequent evaluation sees the new values.
        grid = CartesianGrid((0.0,), (1.0,), (11,))
        ϕ = InterpolatedField(MeshField(x -> 0.0, grid), 1)
        x = SVector(0.05)
        @test ϕ(x) ≈ 0.0 atol = 1.0e-14
        ϕ[1] = 1.0   # node at x = 0, inside the cached cell
        @test ϕ(x) ≈ 0.5 atol = 1.0e-14   # linear interpolant between ϕ[1]=1 and ϕ[2]=0

        # copy! must invalidate as well
        copy!(ϕ, MeshField(x -> 2.0, grid))
        @test ϕ(x) ≈ 2.0 atol = 1.0e-14
    end

    @testset "Concurrent evaluation" begin
        # Each task gets its own scratch buffer, so concurrent evaluation must match
        # serial evaluation exactly (also meaningful with a single thread).
        grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (21, 21))
        f(x) = x[1]^2 + 2 * x[2]^2 - 0.5
        ϕ = InterpolatedField(MeshField(f, grid), 3)
        pts = [SVector(-1 + 2 * rand(), -1 + 2 * rand()) for _ in 1:1000]
        serial = [ϕ(p) for p in pts]
        threaded = similar(serial)
        Threads.@threads for i in eachindex(pts)
            threaded[i] = ϕ(pts[i])
        end
        @test threaded == serial
    end
end

@testset "Interpolation h-convergence" begin
    # Consecutive convergence orders from L∞ errors on grids of sizes Ns.
    _orders(errs, Ns) = [log(errs[i] / errs[i + 1]) / log(Ns[i + 1] / Ns[i]) for i in 1:(length(Ns) - 1)]

    # Smooth non-polynomial test function on [-1,1]².
    f(x) = sin(π * x[1]) * cos(π * x[2])

    # 20×20 evaluation grid in [-0.95, 0.95]². Using length=20 gives irrational-like
    # spacing (1.9/19) that avoids coinciding with grid nodes for any N in the sweep.
    # A dense set is needed so the sup-norm estimate is stable: with too few points the
    # worst-case location can shift between resolutions and corrupt the order estimate.
    test_pts = [
        SVector(x, y) for x in range(-0.95, 0.95; length = 20)
            for y in range(-0.95, 0.95; length = 20)
    ]

    # Pass bc = ExtrapolationBC(k) so ghost-node extrapolation matches the interpolant
    # order. The default ExtrapolationBC{2} caps accuracy at O(h³) for k ≥ 3 whenever
    # the stencil touches the boundary.
    Ns = [20, 40, 80, 160]

    for k in 1:4
        @testset "interp_order = $k → O(h^$(k + 1))" begin
            errors = map(Ns) do N
                grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (N, N))
                ϕ = InterpolatedField(MeshField(f, grid; bc = ExtrapolationBC(k)), k)
                maximum(pt -> abs(ϕ(pt) - f(pt)), test_pts)
            end
            orders = _orders(errors, Ns)
            @test all(≥(k + 0.5), orders)
        end
    end
end
