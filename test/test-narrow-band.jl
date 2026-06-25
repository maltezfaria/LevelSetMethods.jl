using Test
using LinearAlgebra
using StaticArrays
using LevelSetMethods
import LevelSetMethods as LSM
using LevelSetMethods: D⁺, D⁻, D⁰, D2⁰, D2, weno5⁻, weno5⁺

@testset "Construction" begin
    grid = LSM.CartesianGrid((-2.0, -2.0), (2.0, 2.0), (100, 100))
    ϕ = LSM.MeshField(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)
    nlayers = 5
    nb = NarrowBandMeshField(ϕ; nlayers)

    @test 0 < length(LSM.active_nodeindices(nb)) < length(grid)
    # values are inherited unchanged from ϕ on active nodes (no reinitialization)
    @test all(I -> nb[I] ≈ ϕ[I], LSM.active_nodeindices(nb))

    # The band is topological: every active node lies within `nlayers` axis steps of a cut
    # cell, so its physical distance to the interface is at most `nlayers` cell diagonals.
    h = minimum(LSM.meshsize(grid))
    @test all(I -> abs(nb[I]) <= nlayers * sqrt(2) * h + h, LSM.active_nodeindices(nb))

    # A wider band contains strictly more nodes.
    nb_wide = NarrowBandMeshField(ϕ; nlayers = 10)
    @test length(LSM.active_nodeindices(nb_wide)) > length(LSM.active_nodeindices(nb))

    # The function constructor matches nesting MeshField then restricting.
    nb_f = NarrowBandMeshField(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid; nlayers)
    @test LSM.active_nodeindices(nb_f) == LSM.active_nodeindices(nb)
    @test all(I -> nb_f[I] == nb[I], LSM.active_nodeindices(nb_f))
end

@testset "Extrapolation outside of narrow band" begin
    # The band extrapolant is a first-order Taylor expansion from the nearest band node, so it
    # is exact for affine functions. Indexing past the grid (Imax+k) needs boundary conditions;
    # LinearExtrapolationBC is also affine-exact, so the composition stays exact.
    grid = LSM.CartesianGrid((-2.0, -2.0), (2.0, 2.0), (100, 100))
    f = x -> 3x[1] - 2x[2] + 1.0
    bc = LSM._normalize_bc(LSM.LinearExtrapolationBC(), 2)
    ϕ_bc = LSM._add_boundary_conditions(LSM.MeshField(f, grid), bc)
    nb = NarrowBandMeshField(ϕ_bc; nlayers = 4)
    active_idxs = LSM.active_nodeindices(nb)
    Imin = map(d -> minimum(I[d] for I in active_idxs), (1, 2)) |> CartesianIndex
    Imax = map(d -> maximum(I[d] for I in active_idxs), (1, 2)) |> CartesianIndex
    k = 5
    for I in CartesianIndices(ntuple(d -> Imax[d]:(Imax[d] + k), 2))
        x = LevelSetMethods._getnode(grid, I) # extrapolate outside bounds
        @test nb[I] ≈ f(x)
    end
    for I in CartesianIndices(ntuple(d -> (Imin[d] - k):Imin[d], 2))
        x = LevelSetMethods._getnode(grid, I) # extrapolate outside bounds
        @test nb[I] ≈ f(x)
    end
end


@testset "Derivatives match full grid" begin
    grid = LSM.CartesianGrid((-2.0, -2.0), (2.0, 2.0), (100, 100))
    ϕ = LSM.MeshField(x -> x[1]^2 + x[2]^2 - 1, grid)
    bc = LSM._normalize_bc(LSM.ExtrapolationBC{2}(), 2)
    ϕ_bc = LSM._add_boundary_conditions(ϕ, bc)
    nb = NarrowBandMeshField(ϕ_bc; nlayers = 5)

    best_I = argmin(I -> abs(nb[I]), LSM.active_nodeindices(nb))

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
    # Full-grid field with interpolation
    ϕ_full = LSM.InterpolatedField(LSM.MeshField(f, grid; bc = LSM.ExtrapolationBC{2}()), 3)
    # Narrow-band field: build from ϕ_full (which already has BCs and interp_order).
    # A wide band is used so the interpolation stencils of the query points stay in-band.
    nb = NarrowBandMeshField(ϕ_full; nlayers = 8)

    for x in [SVector(0.5, 0.0), SVector(0.0, 0.5), SVector(0.3, 0.3)]
        @test nb(x) ≈ ϕ_full(x)
    end
end

@testset "NewtonSDF from narrow band" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (50, 50))
    r = 0.5
    exact_sdf(x) = norm(x) - r
    ϕ = LSM.MeshField(exact_sdf, grid)
    bc = LSM._normalize_bc(LSM.ExtrapolationBC{2}(), 2)
    ϕ_bc = LSM._add_boundary_conditions(ϕ, bc)
    nb = NarrowBandMeshField(ϕ_bc; nlayers = 5)

    sdf = LSM.NewtonSDF(nb; upsample = 4)
    @test sdf(SVector(r, 0.0)) ≈ 0.0 atol = 2.0e-5
    @test sdf(SVector(0.0, 0.0)) ≈ -r atol = 2.0e-5

    sdf_full = LSM.NewtonSDF(ϕ_bc; upsample = 4)
    for x in [SVector(0.5, 0.0), SVector(0.3, 0.0), SVector(0.6, 0.0)]
        @test sdf(x) ≈ sdf_full(x) atol = 1.0e-5
    end
end

@testset "NewtonSDF sign far outside band" begin
    # Regression test: a NewtonSDF built from a narrow band must return the correct
    # SIGN when evaluated far outside the band (e.g. domain corners). The
    # closest-point distance is always correct, but the previous default sign,
    # `sign(meshfield(x))`, relied on extrapolating the band interpolant and flipped
    # sign in the corners. The sign is now derived from the closest-point normal
    # `∇φ(cp)`, which is robust everywhere.
    grid = LSM.CartesianGrid((-2.0, -2.0), (2.0, 2.0), (80, 80))
    r = 0.5
    exact_sdf(x) = norm(x) - r
    ϕ = LSM.MeshField(exact_sdf, grid)
    bc = LSM._normalize_bc(LSM.ExtrapolationBC{2}(), 2)
    ϕ_bc = LSM._add_boundary_conditions(ϕ, bc)
    nb = NarrowBandMeshField(ϕ_bc; nlayers = 5)    # band: a few layers around |x| = 0.5
    sdf = LSM.NewtonSDF(nb; upsample = 4)

    # Deep-corner points, ~2 away from the interface and well outside the band.
    for x in (SVector(1.8, 1.8), SVector(-1.8, 1.8), SVector(1.8, -1.8), SVector(-1.8, -1.8))
        @test sdf(x) > 0                            # outside ⇒ positive
        @test sdf(x) ≈ exact_sdf(x) atol = 1.0e-3
    end

    # A deep interior point, also outside the band but on the negative side.
    @test sdf(SVector(0.0, 0.0)) < 0
    @test sdf(SVector(0.0, 0.0)) ≈ -r atol = 1.0e-3
end

@testset "Band update is stable and value-preserving" begin
    # On a static, exact signed distance function, `update_band!` reproduces the same band
    # and leaves the stored values untouched (it never reinitializes).
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (50, 50))
    r = 0.5
    exact_sdf(x) = norm(x) - r
    ϕ = LSM.MeshField(exact_sdf, grid)
    bc = LSM._normalize_bc(LSM.ExtrapolationBC{2}(), 2)
    ϕ_bc = LSM._add_boundary_conditions(ϕ, bc)
    nb = NarrowBandMeshField(ϕ_bc; nlayers = 5)
    set_before = Set(LSM.active_nodeindices(nb))
    vals_before = Dict(I => nb[I] for I in set_before)

    LSM.update_band!(nb; nlayers = 5)

    @test length(LSM.active_nodeindices(nb)) > 0
    # idempotent on a static interface
    @test Set(LSM.active_nodeindices(nb)) == set_before
    # values preserved (no reinitialization)
    @test all(I -> nb[I] == vals_before[I], LSM.active_nodeindices(nb))
    # and they still equal the exact SDF (the field passed in already was one)
    max_err = maximum(LSM.active_nodeindices(nb)) do I
        abs(nb[I] - exact_sdf(getnode(grid, I)))
    end
    @test max_err < 1.0e-5
end

@testset "Copy behavior" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (30, 30))
    ϕ = LSM.MeshField(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)
    bc = LSM._normalize_bc(LSM.LinearExtrapolationBC(), 2)
    ϕ_bc = LSM._add_boundary_conditions(ϕ, bc)

    nb1 = NarrowBandMeshField(ϕ_bc; nlayers = 2)
    nb2 = NarrowBandMeshField(ϕ_bc; nlayers = 4)

    copy!(nb2, nb1)
    @test LSM.active_nodeindices(nb1) == LSM.active_nodeindices(nb2)
    @test all(I -> nb1[I] ≈ nb2[I], LSM.active_nodeindices(nb1))
end

@testset "active vs full index sets" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (30, 30))
    ϕ = LSM.MeshField(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)
    bc = LSM._normalize_bc(LSM.LinearExtrapolationBC(), 2)
    ϕ_bc = LSM._add_boundary_conditions(ϕ, bc)
    nb = NarrowBandMeshField(ϕ_bc; nlayers = 3)

    # nodeindices/cellindices delegate to the mesh (all of them); the active sets are the band
    @test LSM.nodeindices(nb) == LSM.nodeindices(grid)
    @test LSM.cellindices(nb) == LSM.cellindices(grid)
    @test Set(LSM.active_nodeindices(nb)) ⊆ Set(LSM.nodeindices(nb))
    @test length(LSM.active_nodeindices(nb)) < length(LSM.nodeindices(nb))
    @test Set(LSM.active_cellindices(nb)) ⊆ Set(LSM.cellindices(nb))
end

@testset "Volume and perimeter" begin
    # Band-only volume/perimeter must reproduce the full-grid measures, computed from the band
    # alone. Cases span compact, multi-component, boundary-clipped and domain-spanning shapes.
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (100, 100))
    cases = (
        x -> norm(x) - 0.5,                                                   # compact circle
        x -> min(
            norm(x .- SVector(0.4, 0.0)) - 0.25,
            norm(x .- SVector(-0.4, 0.0)) - 0.2
        ),                        # two discs
        x -> norm(x .- SVector(0.7, 0.0)) - 0.6,                             # clipped by boundary
        x -> x[2],                                                            # slab spanning domain
    )
    for f in cases
        ϕ = LSM.MeshField(f, grid)
        nb = NarrowBandMeshField(ϕ; nlayers = 3)
        @test LSM.volume(nb) ≈ LSM.volume(ϕ)
    end

    # perimeter is supported inside the band, so it matches the full-grid sum (and 2πr)
    ϕ = LSM.MeshField(x -> norm(x) - 0.5, grid)
    nb = NarrowBandMeshField(ϕ; nlayers = 3)
    @test LSM.perimeter(nb) ≈ LSM.perimeter(ϕ)
    @test LSM.perimeter(nb) ≈ 2π * 0.5 rtol = 1.0e-2

    # 3D sphere volume, where the band-free-line resolution must scale
    g3 = LSM.CartesianGrid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (40, 40, 40))
    ϕ3 = LSM.MeshField(x -> norm(x) - 0.5, g3)
    nb3 = NarrowBandMeshField(ϕ3; nlayers = 3)
    @test LSM.volume(nb3) ≈ LSM.volume(ϕ3)

    # an empty band (no interface captured) measures zero
    empty_nb = NarrowBandMeshField(Dict{CartesianIndex{2}, Float64}(), grid)
    @test LSM.volume(empty_nb) == 0
end

@testset "Extrapolation at band boundary" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (50, 50))
    ϕ = LSM.MeshField(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)
    bc = LSM._normalize_bc(LSM.LinearExtrapolationBC(), 2)
    ϕ_bc = LSM._add_boundary_conditions(ϕ, bc)
    nb = NarrowBandMeshField(ϕ_bc; nlayers = 3)

    # A halo node (in-grid, just outside the band) extrapolates finitely
    offs = CartesianIndices((-1:1, -1:1))
    halo = findfirst(nodeindices(grid)) do I
        !haskey(values(nb), I) && all(s -> I[s] in axes(nb)[s], 1:2) &&
            any(o -> haskey(values(nb), I + o), offs)
    end
    @test halo !== nothing
    @test isfinite(nb[halo])

    # a node far from the band has no data: it throws rather than fabricating a value
    @test_throws ArgumentError nb[CartesianIndex(1, 1)]
end

@testset "Corner extrapolation (multi-dimensional)" begin
    # For nodes outside the band in multiple dimensions simultaneously the
    # tensor-product linear extrapolation must preserve sign. Accuracy is not
    # guaranteed for non-linear functions, but sign correctness is (no spurious zeros).
    grid = LSM.CartesianGrid((-2.0, -2.0), (2.0, 2.0), (40, 40))
    f(x) = norm(x) - 0.5      # circle SDF; a thin band → both ± neighbors fall outside
    ϕ = LSM.MeshField(f, grid)
    bc = LSM._normalize_bc(LSM.ExtrapolationBC{2}(), 2)
    ϕ_bc = LSM._add_boundary_conditions(ϕ, bc)
    nb = NarrowBandMeshField(ϕ_bc; nlayers = 2)
    vals = values(nb)

    N = ndims(nb)
    for I in LSM.active_nodeindices(nb)
        for d in 1:N
            Im = CartesianIndex(ntuple(i -> i == d ? I[i] - 1 : I[i], N))
            Ip = CartesianIndex(ntuple(i -> i == d ? I[i] + 1 : I[i], N))
            @test sign(nb[Im]) == sign(ϕ_bc[Im])
            @test sign(nb[Ip]) == sign(ϕ_bc[Ip])
        end
    end
end

@testset "Periodic BC unsupported" begin
    # the band is not periodic-aware, so PeriodicBC is rejected at construction
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (50, 50))
    ϕ = LSM.MeshField(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)
    ϕ_bc = LSM._add_boundary_conditions(ϕ, LSM._normalize_bc(LSM.PeriodicBC(), 2))
    @test_throws ArgumentError NarrowBandMeshField(ϕ_bc; nlayers = 3)
end

@testset "3D extrapolation: sign preservation" begin
    # Same guarantee in 3D: linear extrapolation from a well-conditioned SDF
    # preserves sign for all in-grid out-of-band nodes.
    grid = LSM.CartesianGrid((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0), (20, 20, 20))
    f(x) = norm(x) - 0.5      # sphere SDF; a thin band → both ± neighbors fall outside
    ϕ = LSM.MeshField(f, grid)
    bc = LSM._normalize_bc(LSM.ExtrapolationBC{2}(), 3)
    ϕ_bc = LSM._add_boundary_conditions(ϕ, bc)
    nb = NarrowBandMeshField(ϕ_bc; nlayers = 2)
    vals = values(nb)

    N = ndims(nb)
    for I in LSM.active_nodeindices(nb)
        for d in 1:N
            Im = CartesianIndex(ntuple(i -> i == d ? I[i] - 1 : I[i], N))
            Ip = CartesianIndex(ntuple(i -> i == d ? I[i] + 1 : I[i], N))
            @test sign(nb[Im]) == sign(ϕ_bc[Im])
            @test sign(nb[Ip]) == sign(ϕ_bc[Ip])
        end
    end
end

@testset "Band construction from non-SDF input" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (100, 100))
    d = 1; r0 = 0.5; θ0 = -π / 3; α = π / 100.0
    R = [cos(α) -sin(α); sin(α) cos(α)]; M = R * [1 / 0.06^2 0; 0 1 / (4π^2)] * R'
    ϕ = LSM.MeshField(grid) do (x, y)
        r = sqrt(x^2 + y^2); θ = atan(y, x); res = 1.0e30
        for i in 0:4
            θ1 = θ + (2i - 4) * π; v = [r - r0; θ1 - θ0]
            res = min(res, sqrt(v' * M * v) - d)
        end
        res
    end
    bc = LSM._normalize_bc(LSM.ExtrapolationBC{2}(), 2)
    ϕ_bc = LSM._add_boundary_conditions(ϕ, bc)

    nb = NarrowBandMeshField(ϕ_bc; nlayers = 6)
    @test length(LSM.active_nodeindices(nb)) > 3000
end

@testset "3D narrow band" begin
    grid = LSM.CartesianGrid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (25, 25, 25))
    ϕ = LSM.MeshField(x -> norm(x) - 0.45, grid)
    bc = LSM._normalize_bc(LSM.ExtrapolationBC{2}(), 3)
    ϕ_bc = LSM._add_boundary_conditions(ϕ, bc)
    nb = NarrowBandMeshField(ϕ_bc; nlayers = 3)

    @test ndims(nb) == 3
    @test length(LSM.active_nodeindices(nb)) > 0
    @test length(LSM.active_nodeindices(nb)) < length(grid)

    best_I = argmin(I -> abs(nb[I]), LSM.active_nodeindices(nb))
    for dim in 1:3
        @test D⁰(nb, best_I, dim) ≈ D⁰(ϕ_bc, best_I, dim)
    end

    sdf = LSM.NewtonSDF(nb; upsample = 3)
    @test sdf(SVector(0.45, 0.0, 0.0)) ≈ 0.0 atol = 1.0e-3
end

@testset "NarrowBand h-convergence — curve accuracy (2D advection, RK3)" begin
    # Advect a circle at constant velocity and check that the zero level-set (curve)
    # converges to its exact position at the expected rate.
    #
    # Error measure: max |ϕ_nb[I] - sdf_exact(xI, tf)| restricted to nodes within 3h
    # of the true interface.  After each step the band is reinitialized, so ϕ_nb ≈ SDF
    # of the computed zero set; the error at near-interface nodes is the displacement of
    # the computed curve from the true curve.
    #
    # With RK3 at default CFL=0.5, global temporal error O(Δt³) = O(h³) dominates;
    # the reinitialization-accumulation error is also O(h³) (O(h⁴) per call × O(1/h)
    # calls), so both sources give the same rate.  Expected convergence: ≥ 2.5.
    _orders(errs, Ns) = [log(errs[i] / errs[i + 1]) / log(Ns[i + 1] / Ns[i]) for i in 1:(length(Ns) - 1)]

    r = 0.5
    u = SVector(1.0, 0.0)
    tf = 0.5
    Ns = [30, 60, 120]

    errors = map(Ns) do N
        grid = CartesianGrid((-2.0, -2.0), (2.0, 2.0), (N, N))
        h = minimum(LSM.meshsize(grid))
        ϕ_full = MeshField(x -> norm(x) - r, grid)
        ϕ_nb = NarrowBandMeshField(ϕ_full; nlayers = 5)
        eq = LevelSetEquation(;
            terms = (AdvectionTerm((x, t) -> u),),
            ic = ϕ_nb,
            bc = ExtrapolationBC(2),
            integrator = RK3(),
        )
        integrate!(eq, tf; posthook = eq -> reinitialize!(current_state(eq)))
        nb = current_state(eq)
        maximum(LSM.active_nodeindices(nb)) do I
            x = getnode(grid, I)
            # keep only nodes close to the true interface
            norm(x - u * tf) - r |> abs < 3h || return 0.0
            abs(nb[I] - (norm(x - u * tf) - r))
        end
    end
    @test all(≥(2.5), _orders(errors, Ns))
end
