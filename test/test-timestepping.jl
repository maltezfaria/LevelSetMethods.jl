using Test
using StaticArrays
using LevelSetMethods
import LevelSetMethods as LSM

# Run 1D periodic advection of sin(πx) with constant unit velocity and return the L∞ error
# at the final time. WENO5 (5th order spatial) is used so that temporal error dominates.
function _advection_error_1d(integrator, N; u = 1.0, tf = 0.5)
    grid = LSM.CartesianGrid((-1.0,), (1.0,), (N,))
    ϕ = LSM.MeshField(x -> sin(π * x[1]), grid)
    eq = LSM.LevelSetEquation(;
        terms = (LSM.AdvectionTerm((x, t) -> SVector(u)),),
        ic = ϕ,
        bc = LSM.PeriodicBC(),
        integrator = integrator,
    )
    integrate!(eq, tf)
    ϕ_out = LSM.current_state(eq)
    return maximum(nodeindices(LSM.mesh(ϕ_out))) do I
        abs(ϕ_out[I] - sin(π * (getnode(grid, I)[1] - u * tf)))
    end
end

@testset "ForwardEuler — 1D advection accuracy" begin
    @test _advection_error_1d(ForwardEuler(), 200) < 0.05
end

@testset "RK2 — 1D advection accuracy" begin
    @test _advection_error_1d(RK2(), 200) < 1.0e-3
end

@testset "RK3 — 1D advection accuracy" begin
    @test _advection_error_1d(RK3(), 200) < 1.0e-5
end

@testset "Convergence order — 1D periodic advection" begin
    Ns = [50, 100, 200, 400]
    cases = [(ForwardEuler(), 1), (RK2(), 2), (RK3(), 3)]
    for (integrator, expected_order) in cases
        errors = [_advection_error_1d(integrator, N) for N in Ns]
        for i in 1:(length(Ns) - 1)
            order = log(errors[i] / errors[i + 1]) / log(Ns[i + 1] / Ns[i])
            @test order ≥ expected_order - 0.5
        end
    end
end
