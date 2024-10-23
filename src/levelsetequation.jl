mutable struct LevelSetEquation
    terms::Tuple
    integrator::TimeIntegrator
    state::MeshField
    t::Float64
    buffers
end

"""
    LevelSetEquation(; terms, state, boundary_conditions, t = 0, integrator = ForwardEuler())

Create a of a level-set equation of the form `ϕₜ + sum(terms) = 0`, where each `t ∈ terms`
is a [`LevelSetTerm`](@ref) and `ϕ` is a [`LevelSet`](@ref) object with initial value given
by `state`.

Calling `integrate!(ls, tf)` will evolve the level-set equation up to time `tf`, modifying
the `current_state(eq)` and `current_time(eq)` of the object `eq` in the process.

Boundary conditions can be specified in the following ways. If a single `BoundaryCondition`
is passed, it will be applied to all boundaries of the domain. To apply different boundary
conditions, pass a vector of the form `[(xleft,xright), (yleft,yright), ...]` where each
element is a `BoundaryCondition`, and there are as many elements as dimensions in the
domain.
"""
function LevelSetEquation(; terms, integrator, levelset, t, bc)
    N = dimension(levelset)
    bc = if bc isa BoundaryCondition
        ntuple(_ -> (bc, bc), N)
    end
    _check_valid_bc(bc, N) || throw(ArgumentError("Invalid format of boundary conditions"))
    # append boundary conditions to the state
    state = add_boundary_conditions(levelset, bc)
    # create buffers for the time-integrator
    nb = number_of_buffers(integrator)
    buffers = ntuple(_ -> deepcopy(state), nb)
    return LevelSetEquation(terms, integrator, state, t, buffers)
end

function _check_valid_bc(bc, N)
    return true # FIXME: check that bc respects the format
end

function Base.show(io::IO, eq::LevelSetEquation)
    print(io, "Level-set equation given by\n")
    print(io, "\n \t ϕₜ + ")
    terms = eq.terms
    for term in terms[1:end-1]
        print(io, term)
        print(io, " + ")
    end
    print(io, terms[end])
    print(io, " = 0")
    print(io, "\n\n Current time $(eq.t)")
    return io
end

# getters
current_state(ls::LevelSetEquation) = ls.state
current_time(ls::LevelSetEquation) = ls.t[]
buffers(ls::LevelSetEquation) = ls.buffers
time_integrator(ls) = ls.integrator
terms(ls) = ls.terms
boundary_conditions(ls) = boundary_conditions(ls.state)

"""
    integrate!(ls::LevelSetEquation,tf,Δt=Inf)

Integrate the [`LevelSetEquation`](@ref) `ls` up to time `tf`,
mutating the `levelset` and `current_time` of the object `ls` in the
process.

An optional parameter `Δt` can be passed to specify a maximum time-step
allowed for the integration. Note that the internal time-steps taken to evolve
the level-set up to `tf` may be smaller than `Δt` due to stability reasons
related to the `terms` and `integrator` employed.
"""
function integrate!(ls::LevelSetEquation, tf, Δt = Inf)
    tc = current_time(ls)
    msg = "final time $(tf) must be larger than the initial time $(tc):
           the level-set equation cannot be solved back in time"
    @assert tf >= tc msg
    # append boundary conditions for integration
    ϕ          = current_state(ls)
    buf        = buffers(ls)
    integrator = time_integrator(ls)
    # dynamic dispatch. Should not be a problem provided enough computation is
    # done inside of the function below
    out = _integrate!(ϕ, buf, integrator, ls.terms, tc, tf, Δt)
    ls.t = tf
    # a copy may be needed if the last buffer is not the state
    out === ϕ || copy!(values(ϕ), values(out))
    return ls
end

number_of_buffers(fe::ForwardEuler) = 1

@noinline function _integrate!(ϕ, buffers, integrator::ForwardEuler, terms, tc, tf, Δt)
    buffer = buffers[1]
    α      = cfl(integrator)
    Δt_cfl = α * compute_cfl(terms, ϕ)
    Δt     = min(Δt, Δt_cfl)
    while tc <= tf - eps(tc)
        Δt = min(Δt, tf - tc) # if needed, take a smaller time-step to exactly land on tf
        for I in eachindex(ϕ)
            buffer[I] = _compute_terms(terms, ϕ, I)
            buffer[I] = ϕ[I] - Δt * buffer[I] # muladd?
        end
        ϕ, buffer = buffer, ϕ # swap the roles, no copies
        tc += Δt
        @debug tc, Δt
    end
    # @assert tc ≈ tf
    return ϕ
end

number_of_buffers(fe::RK2) = 2

function _integrate!(ϕ::LevelSet, buffers, integrator::RK2, terms, tc, tf, Δt)
    α = cfl(integrator)
    buffer1, buffer2 = buffers[1], buffers[2]
    Δt_cfl = α * compute_cfl(terms, ϕ)
    Δt = min(Δt, Δt_cfl)
    while tc <= tf - eps(tc)
        Δt = min(Δt, tf - tc) # if needed, take a smaller time-step to exactly land on tf
        for I in eachindex(ϕ)
            tmp = _compute_terms(terms, ϕ, I)
            buffer1[I] = ϕ[I] - Δt * tmp # muladd?
            buffer2[I] = ϕ[I] - 0.5 * Δt * tmp # muladd?
        end
        for I in eachindex(ϕ)
            tmp = _compute_terms(terms, buffer1, I)
            buffer2[I] -= 0.5 * Δt * tmp
        end
        ϕ, buffer1, buffer2 = buffer2, ϕ, buffer1 # swap the roles, no copies
        tc += Δt
        @debug tc, Δt
    end
    # @assert tc ≈ tf
    return ϕ
end

number_of_buffers(fe::RKLM2) = 1

function _integrate!(ϕ::LevelSet, buffers, integrator::RKLM2, terms, tc, tf, Δt)
    buffer = buffers[1]
    α      = cfl(integrator)
    Δt_cfl = α * compute_cfl(terms, ϕ)
    Δt     = min(Δt, Δt_cfl)
    while tc <= tf - eps(tc)
        Δt = min(Δt, tf - tc) # if needed, take a smaller time-step to exactly land on tf
        for I in eachindex(ϕ)
            tmp = _compute_terms(terms, ϕ, I)
            buffer[I] = tmp
        end
        for I in eachindex(ϕ)
            ϕ[I] = ϕ[I] - Δt * buffer[I] # muladd?
        end
        applybc!(ϕ)
        for I in eachindex(ϕ)
            tmp = _compute_terms(terms, ϕ, I)
            buffer[I] = ϕ[I] - 0.5 * Δt * tmp + 0.5 * Δt * buffer[I]
        end
        ϕ, buffer = buffer, ϕ # swap the roles, no copies
        tc += Δt
        @debug tc, Δt
    end
    # @assert tc ≈ tf
    return ϕ
end

function _compute_terms(terms, ϕ, I)
    sum(terms) do term
        return _compute_term(term, ϕ, I)
    end
end
