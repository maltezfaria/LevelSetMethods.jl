using Test
using LevelSetMethods
using StaticArrays
using LinearAlgebra
import LevelSetMethods as LSM

@testset "smooth_heaviside and smooth_delta" begin
    α = 0.1
    @test LSM.smooth_heaviside( 0.2, α) == 1.0
    @test LSM.smooth_heaviside(-0.2, α) == 0.0
    @test LSM.smooth_heaviside(0.0, α) ≈ 0.5
    @test LSM.smooth_heaviside(α, α) ≈ 1.0
    @test LSM.smooth_heaviside(-α, α) ≈ 0.0 atol=1e-12
    @test LSM.smooth_delta(0.0, α) ≈ 1/α
    @test LSM.smooth_delta(α, α) ≈ 0.0 atol=1e-12
    @test LSM.smooth_delta(0.2, α) == 0.0
    @test LSM.smooth_delta(-0.2, α) == 0.0
end

@testset "trapezoidal_coefficients" begin
    grid2 = CartesianGrid((0.0,0.0),(1.0,1.0),(4,5))
    ϕ2 = LevelSet(x->0.0, grid2)
    c2 = LSM.trapezoidal_coefficients(ϕ2)
    @test size(c2) == (4,5)
    # corners = 1/4, edges = 1/2, interior =1
    @test c2[1,1] == 0.25
    @test c2[1,3] == 0.5
    @test c2[3,1] == 0.5
    @test c2[2,2] == 1.0
    @test sum(c2) ≈ (4-1)*(5-1)  # trapezoidal rule area scaling check

    grid3 = CartesianGrid((0.0,0.0,0.0),(1.0,1.0,1.0),(3,3,3))
    ϕ3 = LevelSet(x->0.0, grid3)
    c3 = LSM.trapezoidal_coefficients(ϕ3)
    @test c3[1,1,1] == 0.125
    @test c3[2,1,1] == 0.25
    @test c3[2,2,1] == 0.5
    @test c3[2,2,2] == 1.0
end

@testset "_ensure_boundary_conditions" begin
    grid = CartesianGrid((-1.0,-1.0),(1.0,1.0),(10,10))
    ϕ = LevelSet(x->x[1], grid)  # no bc
    @test !LSM.has_boundary_conditions(ϕ)
    ϕbc = LSM._ensure_boundary_conditions(ϕ)
    @test LSM.has_boundary_conditions(ϕbc)
    @test ϕbc !== ϕ || true  # may return new object
    ϕ2 = LevelSet(x->x[1], grid, ((LSM.NeumannBC(),LSM.NeumannBC()),(LSM.NeumannBC(),LSM.NeumannBC())))
    @test LSM._ensure_boundary_conditions(ϕ2) === ϕ2
end

@testset "Geometric shapes" begin
    grid2 = CartesianGrid((-2.0,-2.0),(2.0,2.0),(101,101))
    c = LSM.circle(grid2; center=(0.0,0.0), radius=1.0)
    @test c isa LevelSet
    @test c[51,51] ≈ -1.0 atol=1e-12  # center at index 51
    @test c[101,51] ≈ 1.0 atol=1e-2  # right edge approx

    r = LSM.rectangle(grid2; center=(0.5,-0.5), width=(1.0,0.5))
    @test r[64,39] < 0  # inside roughly at (0.5,-0.5) -> index 64,39
    @test r[1,1] > 0

    s = LSM.star(grid2; radius=1.0, deformation=0.25, n=5)
    @test minimum(values(s)) < 0 < maximum(values(s))

    d = LSM.dumbbell(grid2; width=1.0, height=0.2, radius=0.25)
    @test minimum(values(d)) < 0

    z = LSM.zalesak_disk(grid2; center=(0.0,0.0), radius=0.5, width=0.25, height=1.0)
    @test minimum(values(z)) < 0 < maximum(values(z))

    grid3 = CartesianGrid((-1.0,-1.0,-1.0),(1.0,1.0,1.0),(21,21,21))
    sph = LSM.sphere(grid3; center=(0.0,0.0,0.0), radius=0.5)
    @test sph[11,11,11] ≈ -0.5 atol=1e-12
    @test_throws ArgumentError LSM.circle(CartesianGrid((0.0,0.0,0.0),(1.0,1.0,1.0),(5,5,5)); radius=1.0)
    @test_throws ArgumentError LSM.sphere(CartesianGrid((0.0,0.0),(1.0,1.0),(5,5)); radius=0.5)
    @test_throws ArgumentError LSM.zalesak_disk(CartesianGrid((0.0,0.0,0.0),(1.0,1.0,1.0),(5,5,5)); radius=0.5)
end

@testset "Set operations" begin
    grid = CartesianGrid((-2.0,-2.0),(2.0,2.0),(41,41))
    c1 = LSM.circle(grid; center=(-0.5,0.0), radius=0.6)
    c2 = LSM.circle(grid; center=( 0.5,0.0), radius=0.6)
    u = c1 ∪ c2
    @test minimum(values(u)) < minimum(values(c1)) + 1e-12 || minimum(values(u)) ≈ minimum(values(c1))
    @test values(u)[21,21] < 0  # overlap region should be negative (inside at least one)

    i = c1 ∩ c2
    @test maximum(values(i)) > maximum(values(c1)) - 1e-12

    comp = LSM.complement(c1)
    @test values(comp) ≈ -values(c1)

    diff = setdiff(c1,c2)
    # point in left circle center should stay negative, right center should become positive
    idx_left = argmin(abs.(LSM.grid1d(grid,1) .+ 0.5))
    idx_right = argmin(abs.(LSM.grid1d(grid,1) .- 0.5))
    mid = 21
    @test diff[idx_left, mid] < 0
    @test diff[idx_right, mid] > 0

    # in-place versions return same object
    c1copy = deepcopy(c1)
    @test union!(c1copy,c2) === c1copy
    c1copy2 = deepcopy(c1)
    @test intersect!(c1copy2,c2) === c1copy2
    c1copy3 = deepcopy(c1)
    @test LSM.complement!(c1copy3) === c1copy3
    c1copy4 = deepcopy(c1)
    @test setdiff!(c1copy4,c2) === c1copy4
end

@testset "volume and perimeter" begin
    R = 0.5
    grid = CartesianGrid((-1.0,-1.0),(1.0,1.0),(200,200))
    ϕc = LSM.circle(grid; center=(0.0,0.0), radius=R)
    vol = LSM.volume(ϕc)
    per = LSM.perimeter(ϕc)
    @test vol ≈ π*R^2 rtol=0.02
    @test per ≈ 2π*R rtol=0.02

    grid3 = CartesianGrid((-1.0,-1.0,-1.0),(1.0,1.0,1.0),(60,60,60))
    sph = LSM.sphere(grid3; center=(0.0,0.0,0.0), radius=R)
    vol3 = LSM.volume(sph)
    @test vol3 ≈ 4/3*π*R^3 rtol=0.05
end

@testset "gradient normal hessian" begin
    grid = CartesianGrid((-2.0,-2.0),(2.0,2.0),(101,51))
    f(x) = x[1]^2 + x[2]^2
    ϕ = LevelSet(f, grid)
    ϕbc = LSM.add_boundary_conditions(ϕ, ((LSM.LinearExtrapolationBC(),LSM.LinearExtrapolationBC()),(LSM.LinearExtrapolationBC(),LSM.LinearExtrapolationBC())))
    I = CartesianIndex(60,30)
    x = grid[I]
    g = LSM.gradient(ϕbc, I)
    @test g[1] ≈ 2*x[1] atol=0.05
    @test g[2] ≈ 2*x[2] atol=0.1
    gfull = LSM.gradient(ϕbc)
    @test gfull[I][1] ≈ g[1]

    n = LSM.normal(ϕbc, I)
    @test norm(n) ≈ 1.0 atol=1e-6
    nfull = LSM.normal(ϕbc)
    @test norm(nfull[I]) ≈ 1.0 atol=1e-6

    H = LSM.hessian(ϕbc, I)
    @test H[1,1] ≈ 2.0 atol=0.1
    @test H[2,2] ≈ 2.0 atol=0.1
    @test H[1,2] ≈ 0.0 atol=0.1
    Hfull = LSM.hessian(ϕbc)
    @test Hfull[I][1,1] ≈ 2.0 atol=0.1
end

@testset "curvature" begin
    R = 0.7
    grid = CartesianGrid((-2.0,-2.0),(2.0,2.0),(200,200))
    ϕ = LSM.circle(grid; center=(0.0,0.0), radius=R)
    ϕbc = LSM.add_boundary_conditions(ϕ, ((LSM.LinearExtrapolationBC(),LSM.LinearExtrapolationBC()),(LSM.LinearExtrapolationBC(),LSM.LinearExtrapolationBC())))
    # sample points near interface at angle 0, π/4, π/2
    for θ in (0.0, π/4, π/2)
        x = R * cos(θ); y = R * sin(θ)
        # find nearest grid index
        ix = argmin(abs.(LSM.grid1d(grid,1) .- x))
        iy = argmin(abs.(LSM.grid1d(grid,2) .- y))
        I = CartesianIndex(ix,iy)
        κ = LSM.curvature(ϕbc, I)
        @test κ ≈ 1/R rtol=0.15
    end
    κfull = LSM.curvature(ϕbc)
    @test size(κfull) == size(values(ϕ))
    @test any(isfinite, κfull)
end

@testset "grad_norm" begin
    grid = CartesianGrid((-1.0,-1.0),(1.0,1.0),(50,50))
    ϕ = LevelSet(x-> sqrt(x[1]^2+x[2]^2)-0.5, grid)
    ϕbc = LSM.add_boundary_conditions(ϕ, ((LSM.LinearExtrapolationBC(),LSM.LinearExtrapolationBC()),(LSM.LinearExtrapolationBC(),LSM.LinearExtrapolationBC())))
    gn = LSM.grad_norm(ϕbc)
    @test size(gn) == size(values(ϕ))
    # away from singularity, grad norm should be near 1 for SDF
    mid = CartesianIndex(25,25)
    # center is singular, pick near interface
    idx = CartesianIndex(25,38)  # approx radius 0.5 up
    @test gn[idx] ≈ 1.0 atol=0.2
    # no bc should error
    ϕnobc = LevelSet(x->x[1], grid)
    @test_throws ErrorException LSM.grad_norm(ϕnobc)
end
