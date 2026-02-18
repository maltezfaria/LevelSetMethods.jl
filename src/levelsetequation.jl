mutable struct LevelSetEquation
    terms::Tuple
    integrator::TimeIntegrator
    state::MeshField
    t::Float64
    reinit_freq::Union{Nothing, Int}
    buffers
end

"""
    LevelSetEquation(; terms, levelset, boundary_conditions, t = 0, integrator = RK2(),
    reinit_freq = nothing)

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

Reinitialization of the level-set function can be performed every
`reinit_freq` time steps. By default, no reinitialization is performed. Using
this feature requires the `ReinitializationExt` to be loaded.

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
function LevelSetEquation(;
        terms,
        integrator = RK2(),
        levelset,
        t = 0,
        bc,
        reinit_freq = nothing,
    )
    N = dimension(levelset)
    terms = _normalize_terms(terms, N)
    bc = _normalize_bc(bc, N)
    # append boundary conditions to the state
    state = add_boundary_conditions(levelset, bc)
    # create buffers for the time-integrator
    nb = number_of_buffers(integrator)
    buffers = ntuple(_ -> deepcopy(state), nb)
    return LevelSetEquation(terms, integrator, state, t, reinit_freq, buffers)
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
reinit_freq(ls::LevelSetEquation) = ls.reinit_freq

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
    rfreq = reinit_freq(ls)
    # dynamic dispatch. Should not be a problem provided enough computation is
    # done inside of the function below
    out = _integrate!(ϕ, buf, integrator, ls.terms, rfreq, tc, tf, Δt)
    ls.t = tf
    # a copy may be needed if the last buffer is not the state
    out === ϕ || copy!(values(ϕ), values(out))
    return ls
end

number_of_buffers(fe::ForwardEuler) = 1

@noinline function _integrate!(ϕ, buffers, integrator::ForwardEuler, terms, rfreq, tc, tf, Δt_max)
    buffer = buffers[1]
    α = cfl(integrator)
    nsteps = 0
    while tc <= tf - eps(tc)
        !isnothing(rfreq) && mod(nsteps, rfreq) == 0 && reinitialize!(ϕ)
        # update terms and compute an appropriate time-step
        _update_terms!(terms, ϕ, tc)
        Δt_cfl = α * compute_cfl(terms, ϕ, tc)
        Δt = min(Δt_max, Δt_cfl, tf - tc)
        for I in eachindex(ϕ)
            buffer[I] = _compute_terms(terms, ϕ, I, tc)
            buffer[I] = ϕ[I] - Δt * buffer[I] # muladd?
        end
        ϕ, buffer = buffer, ϕ # swap the roles, no copies
        tc += Δt
        nsteps += 1
        @debug tc, Δt
    end
    # @assert tc ≈ tf
    return ϕ
end

number_of_buffers(fe::RK2) = 2

function _integrate!(ϕ::LevelSet, buffers, integrator::RK2, terms, rfreq, tc, tf, Δt_max)
    # Heun's method
    α = cfl(integrator)
    buffer1, buffer2 = buffers[1], buffers[2]
    nsteps = 0
    while tc <= tf - eps(tc)
        # handle reinitialization
        !isnothing(rfreq) && mod(nsteps, rfreq) == 0 && reinitialize!(ϕ)

        # update terms and compute an appropriate time-step
        _update_terms!(terms, ϕ, tc)
        Δt_cfl = α * compute_cfl(terms, ϕ, tc)
        Δt = min(Δt_max, Δt_cfl, tf - tc)

        for I in eachindex(ϕ)
            tmp = _compute_terms(terms, ϕ, I, tc)
            buffer1[I] = ϕ[I] - Δt * tmp
            buffer2[I] = ϕ[I] - 0.5 * Δt * tmp
        end
        _update_terms!(terms, buffer1, tc + Δt)
        for I in eachindex(ϕ)
            tmp = _compute_terms(terms, buffer1, I, tc + Δt)
            buffer2[I] -= 0.5 * Δt * tmp
        end
        ϕ, buffer1, buffer2 = buffer2, ϕ, buffer1 # swap the roles, no copies
        tc += Δt
        nsteps += 1
        @debug tc, Δt
    end
    # @assert tc ≈ tf
    return ϕ
end

number_of_buffers(fe::RK3) = 2

function _integrate!(ϕ::LevelSet, buffers, integrator::RK3, terms, rfreq, tc, tf, Δt_max)
    buffer1, buffer2 = buffers
    α = cfl(integrator)
    nsteps = 0
    while tc <= tf - eps(tc)
        !isnothing(rfreq) && mod(nsteps, rfreq) == 0 && reinitialize!(ϕ)
        # update terms and compute an appropriate time-step
        _update_terms!(terms, ϕ, tc)
        Δt_cfl = α * compute_cfl(terms, ϕ, tc)
        Δt = min(Δt_max, Δt_cfl, tf - tc)
        for I in eachindex(ϕ)
            tmp = _compute_terms(terms, ϕ, I, tc)
            buffer1[I] = ϕ[I] - Δt * tmp
        end
        _update_terms!(terms, buffer1, tc + Δt)
        for I in eachindex(ϕ)
            tmp = _compute_terms(terms, buffer1, I, tc + Δt)
            buffer2[I] = buffer1[I] - Δt * tmp
            buffer2[I] = 1 / 4 * buffer2[I] + 3 / 4 * ϕ[I]
        end
        _update_terms!(terms, buffer2, tc + 1 / 2 * Δt)
        for I in eachindex(ϕ)
            tmp = _compute_terms(terms, buffer2, I, tc + 1 / 2 * Δt)
            buffer1[I] = buffer2[I] - Δt * tmp
            ϕ[I] = 1 / 3 * ϕ[I] + 2 / 3 * buffer1[I]
        end
        tc += Δt
        nsteps += 1
        @debug tc, Δt
    end
    # @assert tc ≈ tf
    return ϕ
end

number_of_buffers(::SemiImplicitI2OE) = 1

function _integrate!(ϕ::LevelSet, buffers, integrator::SemiImplicitI2OE, terms, rfreq, tc, tf, Δt_max)
    _validate_i2oe_setup(ϕ, terms)
    term = only(terms)
    vals = values(ϕ)
    old_vals = values(buffers[1])
    T = float(eltype(vals))
    N = dimension(ϕ)
    velocity_components = ntuple(_ -> zeros(T, size(vals)), N)

    α = cfl(integrator)
    nsteps = 0
    while tc <= tf - eps(tc)
        !isnothing(rfreq) && mod(nsteps, rfreq) == 0 && reinitialize!(ϕ)
        _update_terms!(terms, ϕ, tc)
        Δt_cfl = α * compute_cfl(terms, ϕ, tc)
        Δt = min(Δt_max, Δt_cfl, tf - tc)

        copy!(old_vals, vals)
        _fill_advection_velocity_components!(velocity_components, term, ϕ, tc)
        _i2oe_global_step!(vals, old_vals, velocity_components, ϕ, Δt)

        tc += Δt
        nsteps += 1
        @debug tc, Δt
    end
    return ϕ
end

function _validate_i2oe_setup(ϕ::LevelSet, terms)
    length(terms) == 1 && first(terms) isa AdvectionTerm || throw(
        ArgumentError("SemiImplicitI2OE requires exactly one AdvectionTerm"),
    )
    all(size(values(ϕ)) .>= 3) || throw(
        ArgumentError("SemiImplicitI2OE requires at least 3 grid nodes along each dimension"),
    )
    return nothing
end

function _fill_advection_velocity_components!(out, term::AdvectionTerm{V}, ϕ, t) where {V}
    vel = velocity(term)
    N = dimension(ϕ)
    if V <: MeshField
        mesh(vel) == mesh(ϕ) ||
            throw(ArgumentError("advection velocity field must be defined on the same mesh"))
        for I in eachindex(ϕ)
            vI = vel[I]
            for dim in 1:N
                out[dim][I] = _velocity_component(vI, dim, N)
            end
        end
    elseif V <: Function
        g = mesh(ϕ)
        for I in eachindex(ϕ)
            vI = vel(g[I], t)
            for dim in 1:N
                out[dim][I] = _velocity_component(vI, dim, N)
            end
        end
    else
        error("velocity field type $V not supported")
    end
    return out
end

function _i2oe_global_step!(vals, old_vals, velocity_components, ϕ, Δt)
    # Coupled I2OE update: solve one global sparse system built from all neighbors.
    T = eltype(vals)
    Δ = meshsize(ϕ)
    N = dimension(ϕ)
    mₚ = prod(Δ)
    fac = T(Δt / (2 * mₚ))
    bcs = boundary_conditions(ϕ)
    grid = mesh(ϕ)
    LI = LinearIndices(vals)
    nb_nodes = length(vals)
    rows = Int[]
    cols = Int[]
    coeffs = T[]
    rhs = zeros(T, nb_nodes)
    sizehint!(rows, nb_nodes * (2N + 1))
    sizehint!(cols, nb_nodes * (2N + 1))
    sizehint!(coeffs, nb_nodes * (2N + 1))

    for I in eachindex(ϕ)
        row = LI[I]
        uold_p = old_vals[I]
        diag = one(T)
        rhsp = uold_p
        for dim in 1:N
            area = _i2oe_face_measure(Δ, dim)
            rel_m = _i2oe_neighbor_relation(I, dim, -1, vals, bcs, grid)
            rel_p = _i2oe_neighbor_relation(I, dim, +1, vals, bcs, grid)

            vface_m = _i2oe_face_velocity(velocity_components[dim], I, rel_m, dim)
            vface_p = _i2oe_face_velocity(velocity_components[dim], I, rel_p, dim)

            a_m = area * vface_m
            a_p = -area * vface_p
            diag, rhsp = _i2oe_add_side_contrib!(
                rows,
                cols,
                coeffs,
                LI,
                old_vals,
                row,
                rel_m,
                a_m,
                fac,
                diag,
                rhsp,
                uold_p,
            )
            diag, rhsp = _i2oe_add_side_contrib!(
                rows,
                cols,
                coeffs,
                LI,
                old_vals,
                row,
                rel_p,
                a_p,
                fac,
                diag,
                rhsp,
                uold_p,
            )
        end
        push!(rows, row)
        push!(cols, row)
        push!(coeffs, diag)
        rhs[row] = rhsp
    end

    A = sparse(rows, cols, coeffs, nb_nodes, nb_nodes)
    copy!(vals, reshape(A \ rhs, size(vals)))
    return vals
end

function _i2oe_add_side_contrib!(
        rows,
        cols,
        coeffs,
        LI,
        old_vals,
        row,
        rel,
        a,
        fac,
        diag,
        rhsp,
        uold_p,
    )
    ain = max(a, zero(a))
    aout = min(a, zero(a))

    α, β, idx, γ = rel
    if ain != 0
        diag += fac * ain * (1 - α)
        if β != 0
            push!(rows, row)
            push!(cols, LI[idx])
            push!(coeffs, -fac * ain * β)
        end
        rhsp += fac * ain * γ
    end

    if aout != 0
        uold_q = _i2oe_neighbor_value(old_vals, uold_p, rel)
        rhsp -= fac * aout * (uold_p - uold_q)
    end
    return diag, rhsp
end

function _i2oe_neighbor_value(old_vals, uold_p, rel)
    α, β, idx, γ = rel
    uold_idx = isnothing(idx) ? zero(uold_p) : old_vals[idx]
    return α * uold_p + β * uold_idx + γ
end

function _i2oe_neighbor_relation(I, dim, side, vals, bcs, grid)
    T = float(eltype(vals))
    ax = axes(vals, dim)
    Ioff = side < 0 ? _decrement_index(I, dim) : _increment_index(I, dim)
    if Ioff[dim] in ax
        return (zero(T), one(T), Ioff, zero(T))
    end

    bc = side < 0 ? bcs[dim][1] : bcs[dim][2]
    if bc isa PeriodicBC
        Iq = _wrap_index_periodic(Ioff, ax, dim)
        return (zero(T), one(T), Iq, zero(T))
    elseif bc isa NeumannBC
        Iq = _wrap_index_neumann(Ioff, ax, dim)
        return (zero(T), one(T), Iq, zero(T))
    elseif bc isa NeumannGradientBC
        Ion, Iin, dist = _wrap_index_neumann_gradient(Ioff, ax, dim)
        Ion == I || throw(
            ArgumentError("SemiImplicitI2OE expected nearest ghost cell for NeumannGradientBC"),
        )
        return (one(T) + T(dist), -T(dist), Iin, zero(T))
    elseif bc isa DirichletBC
        xghost = _getindex(grid, Ioff)
        return (zero(T), zero(T), nothing, T(bc.f(xghost)))
    else
        error("boundary condition $bc is not supported by SemiImplicitI2OE")
    end
end

function _i2oe_face_velocity(velcomp, I, rel, dim)
    α, β, idx, _ = rel
    if isnothing(idx) || α != 0 || β != 1
        return velcomp[I]
    end
    return 0.5 * (velcomp[I] + velcomp[idx])
end

function _i2oe_face_measure(Δ, dim)
    N = length(Δ)
    N == 1 && return one(eltype(Δ))
    return prod(Δ[d] for d in 1:N if d != dim)
end

function _compute_terms(terms, ϕ, I, t)
    return sum(terms) do term
        return _compute_term(term, ϕ, I, t)
    end
end

_update_terms!(terms, ϕ, t) = foreach(term -> update_term!(term, ϕ, t), terms)

"""
    update_term!(term::LevelSetTerm, ϕ, t)

Called before computing the term at each stage of the time evolution.
"""
update_term!(term::LevelSetTerm, ϕ, t) = nothing

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
