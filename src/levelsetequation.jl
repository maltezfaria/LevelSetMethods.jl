mutable struct LevelSetEquation
    terms::Tuple
    integrator::TimeIntegrator
    state::MeshField
    t::Float64
    buffers
end

"""
    LevelSetEquation(; terms, levelset, boundary_conditions, t = 0, integrator = RK3())

Create a of a level-set equation of the form `ϕₜ + sum(terms) = 0`, where each `t ∈ terms`
is a [`LevelSetTerm`](@ref) and `levelset` is the initial [`LevelSet`](@ref).

Calling [`integrate!(ls, tf)`](@ref) will evolve the level-set equation up to time `tf`,
modifying the `current_state(eq)` and `current_time(eq)` of the object `eq` in the process
(and therefore the original `levelset`).

Boundary conditions can be specified in two ways. If a single `BoundaryCondition` is
provided, it will be applied uniformly to all boundaries of the domain. To apply different
boundary conditions to each boundary, pass a tuple of the form `(bc_x, bc_y, ...)` with as
many elements as dimensions in the domain. If `bc_x` is a `BoundaryCondition`, it will be
applied to both boundaries in the `x` direction. If `bc_x` is a tuple of two
`BoundaryCondition`s, the first will be applied to the left boundary and the second to the
right boundary. The same logic applies to the other dimensions.

The optional parameter `t` specifies the initial time of the simulation, and `integrator` is
the [`TimeIntegrator`](@ref) used to evolve the level-set equation.

```jldoctest; output = true
using LevelSetMethods, StaticArrays
grid = CartesianGrid((-1, -1), (1, 1), (50, 50))    # define the grid
ϕ = LevelSet(x -> x[1]^2 + x[2]^2 - 0.5^2, grid)    # initial shape
𝐮 = MeshField(x -> SVector(1, 0), grid)             # advection velocity
terms = (AdvectionTerm(𝐮),)            # advection and curvature terms
bc = PeriodicBC()                                   # periodic boundary conditions
eq = LevelSetEquation(; terms, levelset = ϕ, bc)    # level-set equation

# output

Level-set equation given by

 	 ϕₜ + 𝐮 ⋅ ∇ ϕ = 0

Current time 0.0

```
"""
function LevelSetEquation(; terms, integrator = RK3(), levelset, t = 0, bc)
    N = dimension(levelset)
    terms = _normalize_terms(terms, N)
    bc = _normalize_bc(bc, N)
    # append boundary conditions to the state
    state = add_boundary_conditions(levelset, bc)
    # create buffers for the time-integrator
    nb = number_of_buffers(integrator)
    buffers = ntuple(_ -> deepcopy(state), nb)
    return LevelSetEquation(terms, integrator, state, t, buffers)
end

function _normalize_terms(terms, dim)
    if isa(terms, LevelSetTerm) # single term
        return (terms,)
    else
        N = length(terms)
        return ntuple(N) do i
            if isa(terms[i], LevelSetTerm)
                return terms[i]
            else
                throw(ArgumentError("invalid term $(terms[i]) on entry $i"))
            end
        end
    end
end

function _normalize_bc(bc, dim)
    if isa(bc, BoundaryCondition)
        return ntuple(_ -> (bc, bc), dim)
    else
        length(bc) == dim || throw(ArgumentError("invalid number of boundary conditions"))
        return ntuple(dim) do i
            if isa(bc[i], BoundaryCondition)
                return (bc[i], bc[i])
            else
                length(bc[i]) == 2 && all(isa(bc[i][n], BoundaryCondition) for n in 1:2) ||
                    throw(ArgumentError("invalid boundary condition for dimension $i"))
                # check that periodic boundary conditions are not mixed with others
                if any(isa(bc[i][n], PeriodicBC) for n in 1:2)
                    all(isa(bc[i][n], PeriodicBC) for n in 1:2) || throw(
                        ArgumentError(
                            "periodic boundary conditions cannot be mixed with others in dimension $i",
                        ),
                    )
                end
                return (bc[i][1], bc[i][2])
            end
        end
    end
end

function Base.show(io::IO, eq::LevelSetEquation)
    print(io, "Level-set equation given by\n")
    print(io, "\n \t ϕₜ")
    terms = eq.terms
    for term in terms
        print(io, " + ")
        print(io, term)
    end
    print(io, " = 0")
    print(io, "\n\nCurrent time $(eq.t)")
    return io
end

# getters
current_state(ls::LevelSetEquation) = ls.state
current_time(ls::LevelSetEquation) = ls.t[]
buffers(ls::LevelSetEquation) = ls.buffers
time_integrator(ls) = ls.integrator
terms(ls) = ls.terms
boundary_conditions(ls) = boundary_conditions(ls.state)
mesh(ls::LevelSetEquation) = mesh(ls.state)

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
    ϕ = current_state(ls)
    buf = buffers(ls)
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
    α = cfl(integrator)
    Δt_cfl = α * compute_cfl(terms, ϕ, tc)
    Δt = min(Δt, Δt_cfl)
    while tc <= tf - eps(tc)
        Δt = min(Δt, tf - tc) # if needed, take a smaller time-step to exactly land on tf
        for I in eachindex(ϕ)
            buffer[I] = _compute_terms(terms, ϕ, I, tc)
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
    # Heun's method
    α = cfl(integrator)
    buffer1, buffer2 = buffers[1], buffers[2]
    Δt_cfl = α * compute_cfl(terms, ϕ, tc)
    Δt = min(Δt, Δt_cfl)
    while tc <= tf - eps(tc)
        Δt = min(Δt, tf - tc) # if needed, take a smaller time-step to exactly land on tf
        for I in eachindex(ϕ)
            tmp = _compute_terms(terms, ϕ, I, tc)
            buffer1[I] = ϕ[I] - Δt * tmp
            buffer2[I] = ϕ[I] - 0.5 * Δt * tmp
        end
        for I in eachindex(ϕ)
            tmp = _compute_terms(terms, buffer1, I, tc + Δt)
            buffer2[I] -= 0.5 * Δt * tmp
        end
        ϕ, buffer1, buffer2 = buffer2, ϕ, buffer1 # swap the roles, no copies
        tc += Δt
        @debug tc, Δt
    end
    # @assert tc ≈ tf
    return ϕ
end

number_of_buffers(fe::RK3) = 2

function _integrate!(ϕ::LevelSet, buffers, integrator::RK3, terms, tc, tf, Δt)
    buffer1, buffer2 = buffers
    α = cfl(integrator)
    Δt_cfl = α * compute_cfl(terms, ϕ, tc)
    Δt = min(Δt, Δt_cfl)
    while tc <= tf - eps(tc)
        Δt = min(Δt, tf - tc) # if needed, take a smaller time-step to exactly land on tf
        for I in eachindex(ϕ)
            tmp = _compute_terms(terms, ϕ, I, tc)
            buffer1[I] = ϕ[I] - Δt * tmp # ϕ(t + Δt)
        end
        for I in eachindex(ϕ)
            tmp = _compute_terms(terms, buffer1, I, tc + Δt)
            buffer2[I] = buffer1[I] - Δt * tmp # ϕ(t + 2Δt)
            buffer2[I] = 1 / 4 * buffer2[I] + 3 / 4 * ϕ[I] # ϕ(t + Δt/2) = 3/4 ϕ(t) + 1/4 ϕ(t + 2Δt)
        end
        for I in eachindex(ϕ)
            tmp = _compute_terms(terms, buffer2, I, tc + 3 / 2 * Δt)
            buffer1[I] = buffer2[I] - Δt * tmp # ϕ(t + 3/2 Δt)
            ϕ[I] = 1 / 3 * ϕ[I] + 2 / 3 * buffer1[I] # ϕ(t + Δt) = 1/2 ϕ(t) + 2/3 ϕ(t + 3/2 Δt)
        end
        tc += Δt
        @debug tc, Δt
    end
    # @assert tc ≈ tf
    return ϕ
end

function _compute_terms(terms, ϕ, I, t)
    return sum(terms) do term
        return _compute_term(term, ϕ, I, t)
    end
end

"""
    grad_norm(ϕ::LevelSet[, I])

Compute the norm of the gradient of ϕ at index `I`, i.e. `|∇ϕ|`, or for all grid points
if `I` is not provided.
"""
function grad_norm(ϕ::LevelSet)
    msg = """level-set must have boundary conditions to compute gradient. See
    `add_boundary_conditions`."""
    has_boundary_conditions(ϕ) || error(msg)
    idxs = eachindex(ϕ)
    return map(i -> _compute_∇_norm(sign(ϕ[i]), ϕ, i), idxs)
end
function grad_norm(eq::LevelSetEquation)
    return grad_norm(current_state(eq))
end
