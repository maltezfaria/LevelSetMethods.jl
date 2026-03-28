using Test
using LinearAlgebra
using StaticArrays
using LevelSetMethods
import LevelSetMethods as LSM

@testset "Construction and accessors" begin
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (10, 5))
    f = x -> x[1]^2 + x[2]^2 - 0.5
    ϕ = MeshField(f, grid)
    @test LSM.mesh(ϕ) === grid
    @test LSM.domain(ϕ) isa LSM.FullDomain
    @test !LSM.has_boundary_conditions(ϕ)
    @test ndims(ϕ) == 2
    @test size(values(ϕ)) == (10, 5)
    @test ϕ[3, 2] ≈ f(grid[3, 2])
    @test LSM.meshsize(ϕ) == LSM.meshsize(grid)
end

@testset "add_boundary_conditions" begin
    grid = CartesianGrid((0.0,), (1.0,), (5,))
    ϕ = MeshField(x -> x[1], grid)
    @test !LSM.has_boundary_conditions(ϕ)
    ϕ_bc = LSM.add_boundary_conditions(ϕ, ((NeumannBC(), NeumannBC()),))
    @test LSM.has_boundary_conditions(ϕ_bc)
    @test values(ϕ_bc) === values(ϕ)   # underlying data is aliased
end

@testset "copy!" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (5, 5))
    ϕ = LevelSet(x -> x[1] + x[2], grid)
    ψ = LevelSet(x -> 0.0, grid)
    copy!(ψ, ϕ)
    @test values(ψ) == values(ϕ)
    @test values(ψ) !== values(ϕ)   # copied, not aliased
end

@testset "update_bcs!" begin
    grid = CartesianGrid((0.0,), (1.0,), (5,))
    bc = DirichletBC((x, t) -> t)
    ϕ = MeshField(x -> 0.0, grid, ((bc, NeumannBC()),))
    LSM.update_bcs!(ϕ, 3.0)
    @test bc.t == 3.0
end

@testset "Type aliases" begin
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (5, 5))
    ϕ = LevelSet(x -> norm(x) - 0.5, grid)
    @test ϕ isa LevelSet{2, Float64}
    @test ϕ isa MeshField
end

@testset "Periodic BC getindex" begin
    a = (0, 0)
    b = (1, 1)
    n = (10, 5)
    grid = CartesianGrid(a, b, n)
    vals = rand(n...)
    bcs = ((PeriodicBC(), PeriodicBC()), (PeriodicBC(), PeriodicBC()))
    mf = MeshField(vals, grid, bcs)
    @test mf[1, 1] == vals[1, 1]
    @test mf[1, 0] == vals[1, 4]   # wraps around in dim 2
    @test mf[11, 5] == mf[2, 5]    # wraps around in dim 1
end

@testset "Extrapolation BC getindex" begin
    # 1D: ExtrapolationBC{P} reproduces degree-P polynomials exactly,
    # on a grid not aligned with [0,1], for multiple ghost layers.
    a, b, n = -0.3, 1.7, 10
    grid = CartesianGrid((a,), (b,), (n,))
    h = LSM.meshsize(grid)[1]
    for P in 0:5
        bcs = ((ExtrapolationBC(P), ExtrapolationBC(P)),)
        for k in 0:P
            f = x -> x[1]^k
            ϕ_bc = LSM.add_boundary_conditions(MeshField(f, grid), bcs)
            # node i is at x = a + (i-1)*h, so ghost 1-j is at x = a - j*h
            # and ghost n+j is at x = b + j*h
            for j in 1:(P + 1)
                @test ϕ_bc[1 - j] ≈ f(a - j * h)   atol = 1.0e-10
                @test ϕ_bc[n + j] ≈ f(b + j * h)    atol = 1.0e-10
            end
        end
    end

    # 2D: dimension-by-dimension extrapolation reproduces separable polynomials
    # x^j * y^k exactly for j, k ≤ P, including at corner ghost points.
    grid2 = CartesianGrid((-0.3, 0.5), (1.7, 2.1), (8, 6))
    h1, h2 = LSM.meshsize(grid2)
    a1, a2 = -0.3, 0.5
    b1, b2 = 1.7, 2.1
    n1, n2 = 8, 6
    for P in 1:3
        bcs2 = ((ExtrapolationBC(P), ExtrapolationBC(P)), (ExtrapolationBC(P), ExtrapolationBC(P)))
        for j in 0:P, k in 0:P
            f = x -> x[1]^j * x[2]^k
            ϕ_bc = LSM.add_boundary_conditions(MeshField(f, grid2), bcs2)
            # interior ghost in dim 1, in-bounds in dim 2
            @test ϕ_bc[0, 3] ≈ f((a1 - h1, grid2[1, 3][2]))     atol = 1.0e-10
            @test ϕ_bc[n1 + 1, 3] ≈ f((b1 + h1, grid2[1, 3][2]))  atol = 1.0e-10
            # corner ghost (out-of-bounds in both dims)
            @test ϕ_bc[0, 0] ≈ f((a1 - h1, a2 - h2))             atol = 1.0e-10
            @test ϕ_bc[n1 + 1, n2 + 1] ≈ f((b1 + h1, b2 + h2))       atol = 1.0e-10
        end
    end
end

@testset "Dirichlet BC getindex" begin
    grid = CartesianGrid((0.0,), (1.0,), (5,))
    h = LSM.meshsize(grid)[1]
    bc = DirichletBC((x, t) -> x[1] + t)
    bcs = ((bc, NeumannBC()),)
    ϕ = MeshField(x -> 0.0, grid, bcs)
    @test ϕ[0] ≈ -h + 0.0      # t = 0 initially
    LSM.update_bc!(bc, 2.0)
    @test ϕ[0] ≈ -h + 2.0      # t updated to 2.0
end
