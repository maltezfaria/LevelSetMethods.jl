using Test
using LinearAlgebra
using StaticArrays
using LevelSetMethods
import LevelSetMethods as LSM
using LevelSetMethods: D⁺, D⁻, D⁰, D2⁰, D2, weno5⁻, weno5⁺

@testset "Construction" begin
    grid = LSM.CartesianGrid((-2.0, -2.0), (2.0, 2.0), (100, 100))
    ϕ = LSM.LevelSet(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)
    halfwidth = 0.3
    nb = NarrowBandLevelSet(ϕ, halfwidth; reinitialize = false)

    @test 0 < length(LSM.active_indices(nb)) < length(grid)
    @test all(I -> abs(nb[I]) < LSM.halfwidth(nb), LSM.active_indices(nb))
    @test all(I -> nb[I] ≈ ϕ[I], LSM.active_indices(nb))

    # Check that active points satisfy requirements
    active_idxs = LSM.active_indices(nb)
    @test all(I -> abs(nb[I]) < halfwidth, active_idxs)
    inactive_idxs = setdiff(CartesianIndices(grid), active_idxs)
    @test all(I -> abs(ϕ[I]) >= halfwidth, inactive_idxs)

    # Automatic halfwidth via nlayers
    nb2 = NarrowBandLevelSet(ϕ; nlayers = 8)
    Δx = minimum(LSM.meshsize(grid))
    @test LSM.halfwidth(nb2) ≈ 8 * Δx
    @test length(LSM.active_indices(nb2)) > 0

end

@testset "Extrapolation outside of narrow band" begin
    grid = LSM.CartesianGrid((-2.0, -2.0), (2.0, 2.0), (100, 100))
    # Intentionally not an sdf to check exact extrapolation behavior in polynomial
    # case. Requires sufficiently large extrap_order
    f = x -> x[1]^4 + x[2]^4 - 0.5
    nb = NarrowBandLevelSet(f, grid, 0.3; extrap_order = 4)
    active_idxs = LSM.active_indices(nb)
    # compute extrema in each dimension of active idxs
    Imin = map(d -> minimum(I[d] for I in active_idxs), (1, 2)) |> CartesianIndex
    Imax = map(d -> maximum(I[d] for I in active_idxs), (1, 2)) |> CartesianIndex
    ok = true
    # Check that we can extrapolate correctly, even along diagonals
    k = 5
    Ip = CartesianIndices(ntuple(d -> Imax[d]:(Imax[d] + k), 2))
    for I in Ip
        @test nb[I] ≈ f(grid[I])
    end
    Im = CartesianIndices(ntuple(d -> (Imin[d] - k):Imin[d], 2))
    for I in Im
        @test nb[I] ≈ f(grid[I])
    end
end


@testset "Derivatives match full grid" begin
    grid = LSM.CartesianGrid((-2.0, -2.0), (2.0, 2.0), (100, 100))
    ϕ = LSM.LevelSet(x -> x[1]^2 + x[2]^2 - 1, grid)
    bc = LSM._normalize_bc(LSM.ExtrapolationBC{2}(), 2)
    ϕ_bc = LSM.add_boundary_conditions(ϕ, bc)
    nb = NarrowBandLevelSet(ϕ_bc, 0.5; reinitialize = false)

    best_I = argmin(I -> abs(nb[I]), LSM.active_indices(nb))

    @testset "First order" begin
        for op in (D⁺, D⁻, D⁰)
            for dim in 1:2
                @test op(nb, best_I, dim) ≈ op(ϕ_bc, best_I, dim)
            end
        end
    end

    @testset "Second order" begin
        for dim in 1:2
            @test D2⁰(nb, best_I, dim) ≈ D2⁰(ϕ_bc, best_I, dim)
        end
        for dims in ((1, 2), (2, 1))
            @test D2(nb, best_I, dims) ≈ D2(ϕ_bc, best_I, dims)
        end
    end

    @testset "WENO5" begin
        for dim in 1:2
            @test weno5⁻(nb, best_I, dim) ≈ weno5⁻(ϕ_bc, best_I, dim)
            @test weno5⁺(nb, best_I, dim) ≈ weno5⁺(ϕ_bc, best_I, dim)
        end
    end
end

@testset "Interpolation" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (50, 50))
    f(x) = x[1]^2 + 2 * x[2]^2 - 0.5
    ϕ = LSM.LevelSet(f, grid)
    bc = LSM._normalize_bc(LSM.ExtrapolationBC{2}(), 2)
    ϕ_bc = LSM.add_boundary_conditions(ϕ, bc)
    nb = NarrowBandLevelSet(ϕ_bc, 0.4; reinitialize = false)

    itp_nb = interpolate(nb, 3)
    itp_full = interpolate(ϕ_bc, 3)

    for x in [SVector(0.5, 0.0), SVector(0.0, 0.5), SVector(0.3, 0.3)]
        @test itp_nb(x) ≈ itp_full(x)
    end
end

@testset "NewtonSDF from narrow band" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (50, 50))
    r = 0.5
    exact_sdf(x) = norm(x) - r
    ϕ = LSM.LevelSet(exact_sdf, grid)
    bc = LSM._normalize_bc(LSM.ExtrapolationBC{2}(), 2)
    ϕ_bc = LSM.add_boundary_conditions(ϕ, bc)
    nb = NarrowBandLevelSet(ϕ_bc, 0.3)

    sdf = LSM.NewtonSDF(nb; upsample = 4)
    @test sdf(SVector(r, 0.0)) ≈ 0.0 atol = 2.0e-5
    @test sdf(SVector(0.0, 0.0)) ≈ -r atol = 2.0e-5

    sdf_full = LSM.NewtonSDF(ϕ_bc; upsample = 4)
    for x in [SVector(0.5, 0.0), SVector(0.3, 0.0), SVector(0.6, 0.0)]
        @test sdf(x) ≈ sdf_full(x) atol = 1.0e-5
    end
end

@testset "Band rebuild with interface motion" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (50, 50))
    r = 0.5
    exact_sdf(x) = norm(x) - r
    ϕ = LSM.LevelSet(exact_sdf, grid)
    bc = LSM._normalize_bc(LSM.ExtrapolationBC{2}(), 2)
    ϕ_bc = LSM.add_boundary_conditions(ϕ, bc)
    nb = NarrowBandLevelSet(ϕ_bc, 0.3)
    n_before = length(LSM.active_indices(nb))

    sdf = LSM.NewtonSDF(nb; upsample = 4)
    LSM.rebuild_band!(nb, sdf)

    @test length(LSM.active_indices(nb)) > 0
    max_err = maximum(LSM.active_indices(nb)) do I
        x = grid[I]
        abs(nb[I] - exact_sdf(x))
    end
    @test max_err < 1.0e-5
    @test abs(length(LSM.active_indices(nb)) - n_before) < 0.1 * n_before
end

@testset "Copy behavior" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (30, 30))
    ϕ = LSM.LevelSet(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)
    bc = LSM._normalize_bc(LSM.LinearExtrapolationBC(), 2)
    ϕ_bc = LSM.add_boundary_conditions(ϕ, bc)

    nb1 = NarrowBandLevelSet(ϕ_bc, 0.2; reinitialize = false)
    nb2 = NarrowBandLevelSet(ϕ_bc, 0.4; reinitialize = false)

    copy!(nb2, nb1)
    @test LSM.active_indices(nb1) == LSM.active_indices(nb2)
    @test all(I -> nb1[I] ≈ nb2[I], LSM.active_indices(nb1))
end

@testset "eachindex iteration" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (30, 30))
    ϕ = LSM.LevelSet(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)
    bc = LSM._normalize_bc(LSM.LinearExtrapolationBC(), 2)
    ϕ_bc = LSM.add_boundary_conditions(ϕ, bc)
    nb = NarrowBandLevelSet(ϕ_bc, 0.3; reinitialize = false)

    @test collect(eachindex(nb)) == collect(LSM.active_indices(nb))
end

@testset "Extrapolation at band boundary" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (50, 50))
    ϕ = LSM.LevelSet(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)
    bc = LSM._normalize_bc(LSM.LinearExtrapolationBC(), 2)
    ϕ_bc = LSM.add_boundary_conditions(ϕ, bc)
    nb = NarrowBandLevelSet(ϕ_bc, 0.3)

    # Find an in-grid node outside the band
    non_band = findfirst(
        I -> !haskey(values(nb), I) && all(s -> I[s] in axes(nb)[s], 1:2),
        CartesianIndices(grid)
    )
    @test non_band !== nothing

    # Should extrapolate without error
    val = nb[non_band]
    @test isfinite(val)
end

@testset "Corner extrapolation (multi-dimensional)" begin
    # Tests tensor-product extrapolation: a point outside band in both dimensions
    grid = LSM.CartesianGrid((-2.0, -2.0), (2.0, 2.0), (40, 40))
    f(x) = x[1]^2 + x[2]^2 - 1.0
    ϕ = LSM.LevelSet(f, grid)
    bc = LSM._normalize_bc(LSM.ExtrapolationBC{2}(), 2)
    ϕ_bc = LSM.add_boundary_conditions(ϕ, bc)
    nb = NarrowBandLevelSet(ϕ_bc, 0.5; reinitialize = false)
    vals = values(nb)

    # Check all out-of-band points against full grid
    max_err = 0.0
    for I in CartesianIndices(grid)
        haskey(vals, I) && continue
        all(d -> I[d] in axes(nb)[d], 1:2) || continue
        max_err = max(max_err, abs(nb[I] - ϕ_bc[I]))
    end
    @test max_err < 1.0e-10
end

@testset "Periodic BC" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (50, 50))
    ϕ = LSM.LevelSet(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)
    bc = LSM._normalize_bc(LSM.PeriodicBC(), 2)
    ϕ_bc = LSM.add_boundary_conditions(ϕ, bc)
    nb = NarrowBandLevelSet(ϕ_bc, 0.3)

    I = CartesianIndex(0, 25)
    @test isfinite(nb[I])
end

@testset "3D polynomial extrapolation exactness" begin
    grid = LSM.CartesianGrid((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0), (20, 20, 20))
    f(x) = x[1]^2 + x[2]^2 + x[3]^2 - 1.0
    ϕ = LSM.LevelSet(f, grid)
    bc = LSM._normalize_bc(LSM.ExtrapolationBC{2}(), 3)
    ϕ_bc = LSM.add_boundary_conditions(ϕ, bc)
    nb = NarrowBandLevelSet(ϕ_bc, 0.5; reinitialize = false)
    vals = values(nb)

    max_err = 0.0
    for I in CartesianIndices(grid)
        haskey(vals, I) && continue
        all(d -> I[d] in axes(nb)[d], 1:3) || continue
        max_err = max(max_err, abs(nb[I] - ϕ_bc[I]))
    end
    @test max_err < 1.0e-10
end

@testset "Auto-reinitialization of non-SDF input" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (100, 100))
    d = 1; r0 = 0.5; θ0 = -π / 3; α = π / 100.0
    R = [cos(α) -sin(α); sin(α) cos(α)]; M = R * [1 / 0.06^2 0; 0 1 / (4π^2)] * R'
    ϕ = LSM.LevelSet(grid) do (x, y)
        r = sqrt(x^2 + y^2); θ = atan(y, x); res = 1.0e30
        for i in 0:4
            θ1 = θ + (2i - 4) * π; v = [r - r0; θ1 - θ0]
            res = min(res, sqrt(v' * M * v) - d)
        end
        res
    end
    bc = LSM._normalize_bc(LSM.ExtrapolationBC{2}(), 2)
    ϕ_bc = LSM.add_boundary_conditions(ϕ, bc)

    nb = NarrowBandLevelSet(ϕ_bc; nlayers = 6)
    @test length(LSM.active_indices(nb)) > 3000
end

@testset "3D narrow band" begin
    grid = LSM.CartesianGrid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (25, 25, 25))
    ϕ = LSM.LevelSet(x -> norm(x) - 0.45, grid)
    bc = LSM._normalize_bc(LSM.ExtrapolationBC{2}(), 3)
    ϕ_bc = LSM.add_boundary_conditions(ϕ, bc)
    nb = NarrowBandLevelSet(ϕ_bc, 0.3; reinitialize = false)

    @test ndims(nb) == 3
    @test length(LSM.active_indices(nb)) > 0
    @test length(LSM.active_indices(nb)) < length(grid)

    best_I = argmin(I -> abs(nb[I]), LSM.active_indices(nb))
    for dim in 1:3
        @test D⁰(nb, best_I, dim) ≈ D⁰(ϕ_bc, best_I, dim)
    end

    sdf = LSM.NewtonSDF(nb; upsample = 3)
    @test sdf(SVector(0.45, 0.0, 0.0)) ≈ 0.0 atol = 1.0e-3
end
