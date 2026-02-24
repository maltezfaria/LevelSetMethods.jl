using Test
using LevelSetMethods

@testset "Normal Motion update hook" begin
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (21, 21))
    ϕ = LevelSet(grid) do (x, y)
        sqrt(x^2 + y^2) - 0.5
    end
    bcs = ((PeriodicBC(), PeriodicBC()), (PeriodicBC(), PeriodicBC()))
    ϕ = LevelSetMethods.add_boundary_conditions(ϕ, bcs)

    v = MeshField(zeros(Float64, size(grid)...), grid, nothing)
    term = NormalMotionTerm(v, (speed, ϕ, t) -> (fill!(values(speed), 2t); nothing))
    LevelSetMethods.update_term!(term, ϕ, 0.3)

    @test all(values(v) .== 0.6)
end

@testset "Extend Along Normals" begin
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (81, 61))
    ϕ = LevelSet(grid) do (x, y)
        x
    end

    F = zeros(Float64, size(grid)...)
    frozen = falses(size(F))
    Δ = minimum(LevelSetMethods.meshsize(grid))
    for I in CartesianIndices(F)
        if abs(ϕ[I]) <= Δ
            y = grid[I][2]
            F[I] = sin(π * y)
            frozen[I] = true
        end
    end
    F_ref_interface = copy(F)

    extend_along_normals!(F, ϕ; nb_iters = 150, frozen, cfl = 0.45)

    ygrid = LevelSetMethods.grid1d(grid, 2)
    F_ref = repeat(reshape(sin.(π .* ygrid), 1, :), size(F, 1), 1)
    @test maximum(abs.(F .- F_ref)) < 0.08
    @test F[frozen] == F_ref_interface[frozen]
end

@testset "Circle periodic extension" begin
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (121, 121))
    R = 0.55
    ϕ = LevelSet(grid) do (x, y)
        sqrt(x^2 + y^2) - R
    end
    bcs = ((PeriodicBC(), PeriodicBC()), (PeriodicBC(), PeriodicBC()))
    ϕ = LevelSetMethods.add_boundary_conditions(ϕ, bcs)

    v = zeros(Float64, size(grid)...)
    Δ = minimum(LevelSetMethods.meshsize(grid))
    seed_band = 1.1
    frozen = abs.(values(ϕ)) .<= seed_band * Δ
    for I in CartesianIndices(v)
        frozen[I] || continue
        x, y = grid[I]
        r = sqrt(x^2 + y^2)
        v[I] = y / max(r, eps(Float64))
    end
    v_seed = copy(v)

    extend_along_normals!(v, ϕ; nb_iters = 100, frozen, cfl = 0.45)
    @test v[frozen] == v_seed[frozen]

    vf = MeshField(v, grid, bcs)
    extend_band = 5.0
    sum_abs_n_dot_grad = 0.0
    nb_samples = 0
    for I in eachindex(ϕ)
        if abs(ϕ[I]) <= extend_band * Δ && !frozen[I]
            n = LevelSetMethods.normal(ϕ, I)
            any(isnan, n) && continue
            vx = LevelSetMethods.D⁰(vf, I, 1)
            vy = LevelSetMethods.D⁰(vf, I, 2)
            sum_abs_n_dot_grad += abs(n[1] * vx + n[2] * vy)
            nb_samples += 1
        end
    end
    @test nb_samples > 100
    @test sum_abs_n_dot_grad / nb_samples < 0.12
end

@testset "MeshField and argument checks" begin
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (41, 41))
    ϕ = LevelSet(grid) do (x, y)
        x + y
    end
    vals = zeros(Float64, size(grid)...)
    F = MeshField(vals, grid, nothing)
    out = extend_along_normals!(F, ϕ; nb_iters = 5)
    @test out === F

    @test_throws ArgumentError extend_along_normals!(zeros(Int, size(grid)...), ϕ)
    @test_throws ArgumentError extend_along_normals!(zeros(Float64, 2, 2), ϕ)
    @test_throws ArgumentError extend_along_normals!(
        zeros(Float64, size(grid)...),
        ϕ;
        frozen = falses(size(grid)[1] - 1, size(grid)[2]),
    )
end

function _run_curvature_extension_cycle!(
        ϕ;
        nsteps,
        dt_motion,
        dt_reinit,
        ext_iters,
        seed_band = 1.5,
    )
    grid = LevelSetMethods.mesh(ϕ)
    speed_vals = zeros(Float64, size(grid)...)
    speed = MeshField(speed_vals, grid, nothing)

    update_speed = function (v, ϕstate, t)
        vals = values(v)
        fill!(vals, 0.0)
        Δ = minimum(LevelSetMethods.meshsize(ϕstate))
        frozen = falses(size(vals))
        for I in eachindex(ϕstate)
            if abs(ϕstate[I]) <= seed_band * Δ
                frozen[I] = true
                vals[I] = -LevelSetMethods.curvature(ϕstate, I)
            end
        end
        extend_along_normals!(vals, ϕstate; frozen, cfl = 0.3, nb_iters = ext_iters)
        return nothing
    end

    eq_motion = LevelSetEquation(;
        terms = (NormalMotionTerm(speed, update_speed),),
        levelset = ϕ,
        bc = PeriodicBC(),
        integrator = ForwardEuler(cfl = 0.35),
    )
    eq_reinit = LevelSetEquation(;
        terms = (ReinitializationTerm(),),
        levelset = ϕ,
        bc = PeriodicBC(),
        integrator = ForwardEuler(cfl = 0.45),
    )

    for _ in 1:nsteps
        integrate!(eq_motion, current_time(eq_motion) + dt_motion, dt_motion)
        integrate!(eq_reinit, current_time(eq_reinit) + dt_reinit, dt_reinit)
    end
    return LevelSetMethods.current_state(eq_motion)
end

function _interface_radius_stats(ϕ; band = 1.5)
    grid = LevelSetMethods.mesh(ϕ)
    Δ = minimum(LevelSetMethods.meshsize(ϕ))
    radii = Float64[]
    for I in eachindex(ϕ)
        if abs(ϕ[I]) <= band * Δ
            x = grid[I]
            push!(radii, sqrt(sum(abs2, x)))
        end
    end
    mean_radius = sum(radii) / length(radii)
    std_radius = sqrt(sum((r - mean_radius)^2 for r in radii) / length(radii))

    return mean_radius, std_radius, length(radii)
end

@testset "Classical circular reconstruction (2D)" begin
    grid = CartesianGrid((-0.5, -0.5), (0.5, 0.5), (128, 128))
    R0 = 0.45 # diameter = 9/10
    ϕ = LevelSet(x -> sqrt(x[1]^2 + x[2]^2) - R0, grid)
    Δ = minimum(LevelSetMethods.meshsize(grid))

    ϕf = _run_curvature_extension_cycle!(
        ϕ;
        nsteps = 3,
        dt_motion = 1.2e-3,
        dt_reinit = 0.2 * Δ,
        ext_iters = 30,
    )

    rmean, rstd, npts = _interface_radius_stats(ϕf; band = 1.5)
    @test npts > 300
    @test rmean < R0
    @test rstd / rmean < 0.05
end

@testset "Classical spherical reconstruction (3D)" begin
    grid = CartesianGrid((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), (48, 48, 48))
    R0 = 0.45
    ϕ = LevelSet(x -> sqrt(x[1]^2 + x[2]^2 + x[3]^2) - R0, grid)
    Δ = minimum(LevelSetMethods.meshsize(grid))

    ϕf = _run_curvature_extension_cycle!(
        ϕ;
        nsteps = 2,
        dt_motion = 7.0e-4,
        dt_reinit = 0.15 * Δ,
        ext_iters = 22,
    )

    rmean, rstd, npts = _interface_radius_stats(ϕf; band = 1.5)
    @test npts > 2000
    @test rmean < R0
    @test rstd / rmean < 0.09
end

function _closest_grid_index(grid, x)
    Δ = LevelSetMethods.meshsize(grid)
    lc = grid.lc
    n = size(grid)
    idx = ntuple(length(n)) do d
        i = round(Int, (x[d] - lc[d]) / Δ[d]) + 1
        return clamp(i, 1, n[d])
    end
    return CartesianIndex(idx)
end

@testset "Crystal normal extension signs" begin
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (161, 161))
    R = 0.6
    deformation = 0.45
    nfacets = 6

    ϕ = LevelSetMethods.star(grid; radius = R, deformation = deformation, n = nfacets)
    bcs = ((PeriodicBC(), PeriodicBC()), (PeriodicBC(), PeriodicBC()))
    ϕb = LevelSetMethods.add_boundary_conditions(ϕ, bcs)

    Δ = minimum(LevelSetMethods.meshsize(grid))
    v = zeros(Float64, size(grid)...)
    frozen = abs.(values(ϕb)) .<= 1.5 * Δ
    for I in eachindex(ϕb)
        frozen[I] || continue
        v[I] = -LevelSetMethods.curvature(ϕb, I)
    end
    extend_along_normals!(v, ϕb; frozen, cfl = 0.3, nb_iters = 45)

    tips = Float64[]
    kinks = Float64[]
    for k in 0:(Int(nfacets) - 1)
        θtip = 2π * k / nfacets
        rtip = R * (1 + deformation * cos(nfacets * θtip))
        Itip = _closest_grid_index(grid, (rtip * cos(θtip), rtip * sin(θtip)))
        push!(tips, v[Itip])

        θkink = (2k + 1) * π / nfacets
        rkink = R * (1 + deformation * cos(nfacets * θkink))
        Ikink = _closest_grid_index(grid, (rkink * cos(θkink), rkink * sin(θkink)))
        push!(kinks, v[Ikink])
    end

    mean_tips = sum(tips) / length(tips)
    mean_kinks = sum(kinks) / length(kinks)
    @test mean_tips < 0
    @test mean_kinks > 0

    # One short step with the extended velocity should reduce shape anisotropy.
    _radius_cv(ϕstate) = begin
        rs = Float64[]
        g = LevelSetMethods.mesh(ϕstate)
        δ = minimum(LevelSetMethods.meshsize(ϕstate))
        for I in eachindex(ϕstate)
            if abs(ϕstate[I]) <= 1.5 * δ
                x = g[I]
                push!(rs, sqrt(x[1]^2 + x[2]^2))
            end
        end
        mean_rs = sum(rs) / length(rs)
        std_rs = sqrt(sum((r - mean_rs)^2 for r in rs) / length(rs))
        return std_rs / mean_rs
    end
    cv0 = _radius_cv(ϕb)
    term = NormalMotionTerm(MeshField(v, grid, nothing))
    eq = LevelSetEquation(;
        terms = (term,),
        levelset = ϕ,
        bc = PeriodicBC(),
        integrator = ForwardEuler(cfl = 0.3),
    )
    integrate!(eq, 2.5e-3, 2.5e-3)
    cv1 = _radius_cv(LevelSetMethods.current_state(eq))
    @test cv1 < cv0
end
