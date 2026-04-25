using Test
using LinearAlgebra
using StaticArrays
using LevelSetMethods
import LevelSetMethods as LSM

# Consecutive convergence orders from a sequence of errors on grids Ns.
function _convergence_orders(errors, Ns)
    return [log(errors[i] / errors[i + 1]) / log(Ns[i + 1] / Ns[i]) for i in 1:(length(Ns) - 1)]
end

# Max error between a MeshField and a full-grid field, restricted to
# active nodes with |ϕ| < halfwidth/2 (near the interface where accuracy matters).
function _nb_full_error(nb_s, full_s)
    γ = LSM.halfwidth(nb_s)
    return maximum(LSM.nodeindices(nb_s)) do I
        abs(nb_s[I]) < γ / 2 || return 0.0
        abs(nb_s[I] - full_s[I])
    end
end

@testset "AdvectionTerm WENO5 — convergence order (1D periodic, RK3)" begin
    # 1D advection of sin(πx) with u=1 on a periodic domain.  Exact solution: sin(π(x - t)).
    # WENO5 is 5th-order in space; RK3 temporal error O(Δt³) = O((cfl·Δx)³) dominates
    # at default cfl=0.5, so we use cfl=1e-2 to expose the spatial rate.
    u, tf = 1.0, 0.5
    ϕ_exact = (x, t) -> sin(π * (x[1] - u * t))
    Ns = [20, 40, 80]
    errors = map(Ns) do N
        grid = LSM.CartesianGrid((-1.0,), (1.0,), (N,))
        ϕ = LSM.MeshField(x -> ϕ_exact(x, 0.0), grid)
        eq = LSM.LevelSetEquation(;
            terms = (LSM.AdvectionTerm((x, t) -> SVector(u)),),
            ic = ϕ, bc = PeriodicBC(), integrator = RK3(; cfl = 1.0e-2),
        )
        integrate!(eq, tf)
        ϕ_out = current_state(eq)
        maximum(I -> abs(ϕ_out[I] - ϕ_exact(getnode(grid, I), tf)), nodeindices(LSM.mesh(ϕ_out)))
    end
    @test all(≥(4.5), _convergence_orders(errors, Ns))
end

@testset "NormalMotionTerm — convergence order (2D expanding circle, RK3)" begin
    # ϕ₀ = ‖x‖ - r₀.  The PDE ϕₜ + v|∇ϕ| = 0 with radial symmetry reduces to
    # f_t + v·f_r = 0, giving the exact pointwise solution ϕ(x,t) = ‖x‖ - r₀ - v·t.
    # Error is measured in a band around the interface to avoid the high-curvature
    # region near r = 0.
    r0, v, tf = 0.5, 0.5, 0.2
    ϕ_exact = x -> norm(x) - r0 - v * tf
    Ns = [30, 60, 120]
    errors = map(Ns) do N
        grid = LSM.CartesianGrid((-2.0, -2.0), (2.0, 2.0), (N, N))
        ϕ = LSM.MeshField(x -> norm(x) - r0, grid)
        eq = LSM.LevelSetEquation(;
            terms = (LSM.NormalMotionTerm((x, t) -> v),),
            ic = ϕ, bc = ExtrapolationBC(2), integrator = RK3(),
        )
        integrate!(eq, tf)
        ϕ_out = current_state(eq)
        maximum(nodeindices(LSM.mesh(ϕ_out))) do I
            x = getnode(grid, I)
            (0.5 ≤ norm(x) ≤ 1.5) || return 0.0
            abs(ϕ_out[I] - ϕ_exact(x))
        end
    end
    @test all(≥(1.5), _convergence_orders(errors, Ns))
end

@testset "CurvatureTerm — convergence order (2D circle, RK3)" begin
    # ϕ₀ = ‖x‖ - r₀.  The 2D curvature PDE ϕₜ + b κ|∇ϕ| = 0 with κ = 1/r has
    # the exact pointwise solution ϕ(x,t) = √(‖x‖² − 2bt) − r₀, obtained from
    # the characteristics dr/dt = b/r.  Zero set: ‖x‖ = √(r₀² + 2bt).
    # Curvature discretization is 2nd order; RK3 (3rd order) keeps spatial error dominant.
    # Error is measured in a band around the interface to avoid the large-curvature region
    # near r = 0 (κ = 1/r → ∞) and the boundary.
    r0, b, tf = 0.7, -0.1, 0.2
    ϕ_exact = x -> sqrt(norm(x)^2 - 2b * tf) - r0
    Ns = [30, 60, 120]
    errors = map(Ns) do N
        grid = LSM.CartesianGrid((-2.0, -2.0), (2.0, 2.0), (N, N))
        ϕ = LSM.MeshField(x -> norm(x) - r0, grid)
        eq = LSM.LevelSetEquation(;
            terms = (LSM.CurvatureTerm((x, t) -> b),),
            ic = ϕ, bc = ExtrapolationBC(2), integrator = RK3(),
        )
        integrate!(eq, tf)
        ϕ_out = current_state(eq)
        maximum(nodeindices(LSM.mesh(ϕ_out))) do I
            x = getnode(grid, I)
            (0.5 ≤ norm(x) ≤ 1.5) || return 0.0
            abs(ϕ_out[I] - ϕ_exact(x))
        end
    end
    @test all(≥(1.5), _convergence_orders(errors, Ns))
end

@testset "Reinitialization inside LevelSetEquation" begin
    @testset "with RK2" begin
        grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (33, 33))
        ϕ = LSM.MeshField(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)
        eq = LSM.LevelSetEquation(;
            terms = (LSM.AdvectionTerm((x, t) -> @SVector [1.0, 0.0]),),
            ic = ϕ, bc = LSM.PeriodicBC(), reinit = 2,
        )
        LSM.integrate!(eq, 0.2)
        @test eq isa LSM.LevelSetEquation
    end

    @testset "with SemiImplicitI2OE" begin
        grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (33, 33))
        ϕ = LSM.MeshField(x -> sqrt(x[1]^2 + x[2]^2) - 0.5, grid)
        eq = LSM.LevelSetEquation(;
            terms = (LSM.AdvectionTerm((x, t) -> @SVector [1.0, 0.0]),),
            ic = ϕ, bc = LSM.PeriodicBC(), integrator = LSM.SemiImplicitI2OE(), reinit = 2,
        )
        @test LSM.integrate!(eq, 1.0e-3, 1.0e-3) isa LSM.LevelSetEquation
    end
end

@testset "NarrowBand integrate! — advection matches full grid" begin
    grid = LSM.CartesianGrid((-2.0, -2.0), (2.0, 2.0), (60, 60))
    ϕ = LSM.MeshField(x -> norm(x) - 0.5, grid)
    𝐮 = (x, t) -> SVector(1.0, 0.0)
    bc = ExtrapolationBC(2)
    eq_nb = LevelSetEquation(; terms = (AdvectionTerm(𝐮),), ic = NarrowBandMeshField(ϕ; nlayers = 5), bc, reinit = 1)
    eq_full = LevelSetEquation(; terms = (AdvectionTerm(𝐮),), ic = deepcopy(ϕ), bc, reinit = 1)
    integrate!(eq_full, 0.1); integrate!(eq_nb, 0.1)
    @test _nb_full_error(current_state(eq_nb), current_state(eq_full)) < 1.0e-3
end

@testset "NarrowBand integrate! — advection with reinitialization" begin
    grid = LSM.CartesianGrid((-2.0, -2.0), (2.0, 2.0), (60, 60))
    ϕ = LSM.MeshField(x -> norm(x) - 0.5, grid)
    eq_nb = LevelSetEquation(;
        terms = (AdvectionTerm((x, t) -> SVector(1.0, 0.0)),),
        ic = NarrowBandMeshField(ϕ), bc = ExtrapolationBC(2), reinit = NewtonReinitializer(),
    )
    integrate!(eq_nb, 0.1)
    nb_s = current_state(eq_nb)
    γ = LSM.halfwidth(nb_s)
    exact_sdf = x -> norm(x - SVector(0.1, 0.0)) - 0.5
    @test length(LSM.nodeindices(nb_s)) > 0
    @test maximum(LSM.nodeindices(nb_s)) do I
        abs(nb_s[I]) < γ / 2 || return 0.0
        abs(nb_s[I] - exact_sdf(getnode(grid, I)))
    end < 0.01
end

@testset "NarrowBand integrate! — spiral curvature flow matches full grid" begin
    # Spiral with multiple closely-spaced arms; stresses the band-rebuild logic
    # because inter-arm gaps can be narrower than the halfwidth.
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (50, 50))
    r0, θ0, α = 0.5, -π / 3, π / 100
    R = [cos(α) -sin(α); sin(α) cos(α)]
    M = R * [1 / 0.06^2 0; 0 1 / (4π^2)] * R'
    ϕ = LSM.MeshField(grid) do (x, y)
        r, θ = sqrt(x^2 + y^2), atan(y, x)
        minimum(0:4) do i
            v = [r - r0; θ + (2i - 4) * π - θ0]
            sqrt(v' * M * v) - 1
        end
    end
    b, reinit, bc = (x, t) -> -0.1, NewtonReinitializer(), ExtrapolationBC(2)
    eq_nb = LevelSetEquation(; ic = NarrowBandMeshField(ϕ), bc, terms = (CurvatureTerm(b),), reinit)
    eq_full = LevelSetEquation(; ic = deepcopy(ϕ), bc, terms = (CurvatureTerm(b),), reinit)
    integrate!(eq_full, 0.1); integrate!(eq_nb, 0.1)
    @test _nb_full_error(current_state(eq_nb), current_state(eq_full)) < 0.05
end

@testset "NarrowBand integrate! — full rotation" begin
    grid = LSM.CartesianGrid((-2.0, -2.0), (2.0, 2.0), (40, 40))
    ϕ = LSM.MeshField(x -> norm(x - SVector(0.8, 0.0)) - 0.5, grid)
    𝐮, bc = (x, t) -> SVector(-x[2], x[1]), ExtrapolationBC(2)
    eq_nb = LevelSetEquation(; terms = (AdvectionTerm(𝐮),), ic = NarrowBandMeshField(ϕ), bc, reinit = NewtonReinitializer())
    eq_full = LevelSetEquation(; terms = (AdvectionTerm(𝐮),), ic = deepcopy(ϕ), bc)
    integrate!(eq_full, 2π); integrate!(eq_nb, 2π)
    nb_s = current_state(eq_nb)
    @test length(LSM.nodeindices(nb_s)) > 0
    @test _nb_full_error(nb_s, current_state(eq_full)) < 0.01
end

@testset "NarrowBand integrate! — star rotation" begin
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (40, 40))
    ϕ₀ = MeshField(grid) do (x, y)
        r, θ = sqrt(x^2 + y^2), atan(y, x) - π / 2
        r - (0.5 + 0.1 * cos(5θ))
    end
    𝐮, bc = (x, t) -> SVector(-x[2], x[1]), ExtrapolationBC(2)
    eq_nb = LevelSetEquation(; terms = (AdvectionTerm(𝐮),), ic = NarrowBandMeshField(ϕ₀), bc, reinit = NewtonReinitializer())
    eq_full = LevelSetEquation(; terms = (AdvectionTerm(𝐮),), ic = deepcopy(ϕ₀), bc)
    integrate!(eq_full, pi); integrate!(eq_nb, pi)
    nb_s = current_state(eq_nb)
    @test length(LSM.nodeindices(nb_s)) > 0
    @test _nb_full_error(nb_s, current_state(eq_full)) < 0.05
end
