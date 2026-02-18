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

    extend_along_normals!(F, ϕ; nb_iters = 140, frozen, cfl = 0.45)

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

    extend_along_normals!(v, ϕ; nb_iters = 180, frozen, cfl = 0.45)
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
