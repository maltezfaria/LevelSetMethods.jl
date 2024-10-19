"""
    abstract type TimeIntegrator end

Abstract type for time integrators. See `subtypes(TimeIntegrator)` for a list of available
time integrators.
"""
abstract type TimeIntegrator end

Base.@kwdef struct ForwardEuler <: TimeIntegrator
    cfl::Float64 = 0.5
end
cfl(fe::ForwardEuler) = fe.cfl

Base.@kwdef struct RK2 <: TimeIntegrator
    cfl::Float64 = 0.5
end
cfl(rk2::RK2) = rk2.cfl

Base.@kwdef struct RKLM2 <: TimeIntegrator
    cfl::Float64 = 0.5
end
cfl(rk2::RKLM2) = rk2.cfl

# function evolve!(ϕ,integ::RK2,terms,bc,t,Δtmax=Inf)
#     α = integ.cfl
#     buffer1,buffer2 = integ.buffers[1],integ.buffers[2]
#     fill!(values(buffer1),0)
#     fill!(values(buffer2),0)
#     #
#     buffer1, Δtˢ = compute_terms!(buffer1,terms,ϕ,bc)
#     Δt = min(Δtmax,α*Δtˢ)
#     axpy!(-Δt/2,buffer1.vals,ϕ.vals) # ϕ = ϕ - dt/2*buffer1
#     #
#     @. buffer1.vals = ϕ.vals + Δt * buffer1.vals
#     buffer2, _   = compute_terms!(buffer2,terms,buffer1,bc)
#     #
#     axpy!(-Δt/2,buffer2.vals,ϕ.vals) # ϕ = ϕ - dt/2*buffer2
#     return ϕ,t+Δt
# end
