"""
    abstract type TimeIntegrator end

Abstract type for time integrators. See `subtypes(TimeIntegrator)` for a list of available
time integrators.
"""
abstract type TimeIntegrator end

@kwdef struct ForwardEuler <: TimeIntegrator
    cfl::Float64 = 0.5
end
cfl(fe::ForwardEuler) = fe.cfl

"""
    struct RK2

Second order total variation dimishing Runge-Kutta scheme, also known as Heun's
predictor-corrector method.
"""
@kwdef struct RK2 <: TimeIntegrator
    cfl::Float64 = 0.5
end
cfl(rk2::RK2) = rk2.cfl

"""
    struct RK3

Third order total variation dimishing Runge-Kutta scheme.
"""
@kwdef struct RK3 <: TimeIntegrator
    cfl::Float64 = 0.5
end

cfl(rk3::RK3) = rk3.cfl
