using Test
using LinearAlgebra
using StaticArrays
using LevelSetMethods
import LevelSetMethods as LSM

@testset "AdvectionTerm CFL" begin
    grid = LSM.CartesianGrid((-1.0,), (1.0,), (100,))
    ϕ = LSM.MeshField(x -> x[1], grid)
    Δx = LSM.meshsize(ϕ, 1)
    term = LSM.AdvectionTerm((x, t) -> SVector(2.0))
    @test LSM.compute_cfl((term,), ϕ, 0.0) ≈ Δx / 2.0
end

@testset "CurvatureTerm CFL" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (50, 50))
    ϕ = LSM.MeshField(x -> norm(x) - 0.5, grid)
    Δx = minimum(LSM.meshsize(ϕ))
    b = 0.5
    term = LSM.CurvatureTerm((x, t) -> b)
    @test LSM.compute_cfl((term,), ϕ, 0.0) ≈ Δx^2 / (2b)
end

@testset "NormalMotionTerm CFL" begin
    grid = LSM.CartesianGrid((-1.0,), (1.0,), (100,))
    ϕ = LSM.MeshField(x -> x[1], grid)
    Δx = LSM.meshsize(ϕ, 1)
    v = 3.0
    term = LSM.NormalMotionTerm((x, t) -> v)
    @test LSM.compute_cfl((term,), ϕ, 0.0) ≈ Δx / v
end

@testset "EikonalReinitializationTerm — drives scaled SDF toward unit gradient" begin
    # ϕ = 2*(x - 0.3): correct zero set but |∇ϕ| = 2 ≠ 1.
    # After pseudo-time marching it should converge to x - 0.3.
    grid = LSM.CartesianGrid((-1.0,), (1.0,), (101,))
    ϕ = LSM.MeshField(x -> 2 * (x[1] - 0.3), grid)
    eq = LSM.LevelSetEquation(;
        terms = (LSM.EikonalReinitializationTerm(ϕ),),
        ic = deepcopy(ϕ),
        bc = LSM.LinearExtrapolationBC(),
    )
    integrate!(eq, 2.0)
    ϕ_out = LSM.current_state(eq)
    ϕ_exact = LSM.MeshField(x -> x[1] - 0.3, grid)
    err = maximum(nodeindices(LSM.mesh(ϕ_out))) do I
        abs(ϕ_out[I]) > 0.5 && return 0.0
        abs(ϕ_out[I] - ϕ_exact[I])
    end
    @test err < 0.05
end
