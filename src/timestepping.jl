"""
    abstract type TimeIntegrator end

Abstract type for time integrators. See `subtypes(TimeIntegrator)` for a list of available
time integrators.
"""
abstract type TimeIntegrator end

"""
    struct ForwardEuler

First-order explicit Forward Euler time integration scheme.

```jldoctest; output = true
using LevelSetMethods
ForwardEuler()

# output

ForwardEuler (1st order explicit)
  └─ cfl: 0.5
```
"""
@kwdef struct ForwardEuler <: TimeIntegrator
    cfl::Float64 = 0.5
end
cfl(fe::ForwardEuler) = fe.cfl

"""
    struct RK2

Second order total variation dimishing Runge-Kutta scheme, also known as Heun's
predictor-corrector method.

```jldoctest; output = true
using LevelSetMethods
RK2()

# output

RK2 (2nd order TVD Runge-Kutta, Heun's method)
  └─ cfl: 0.5
```
"""
@kwdef struct RK2 <: TimeIntegrator
    cfl::Float64 = 0.5
end
cfl(rk2::RK2) = rk2.cfl

"""
    struct RK3

Third order total variation dimishing Runge-Kutta scheme.

```jldoctest; output = true
using LevelSetMethods
RK3()

# output

RK3 (3rd order TVD Runge-Kutta)
  └─ cfl: 0.5
```
"""
@kwdef struct RK3 <: TimeIntegrator
    cfl::Float64 = 0.5
end

cfl(rk3::RK3) = rk3.cfl

"""
    struct SemiImplicitI2OE

Semi-implicit finite-volume scheme of the I2OE family (Mikula et al.) for
advection problems.

```jldoctest; output = true
using LevelSetMethods
SemiImplicitI2OE()

# output

SemiImplicitI2OE (semi-implicit advection, Mikula et al.)
  └─ cfl: 2.0
```
"""
@kwdef struct SemiImplicitI2OE <: TimeIntegrator
    cfl::Float64 = 2.0
end
cfl(i2oe::SemiImplicitI2OE) = i2oe.cfl

function Base.show(io::IO, ::MIME"text/plain", s::ForwardEuler)
    println(io, "ForwardEuler (1st order explicit)")
    return print(io, "  └─ cfl: $(s.cfl)")
end

function Base.show(io::IO, ::MIME"text/plain", s::RK2)
    println(io, "RK2 (2nd order TVD Runge-Kutta, Heun's method)")
    return print(io, "  └─ cfl: $(s.cfl)")
end

function Base.show(io::IO, ::MIME"text/plain", s::RK3)
    println(io, "RK3 (3rd order TVD Runge-Kutta)")
    return print(io, "  └─ cfl: $(s.cfl)")
end

function Base.show(io::IO, ::MIME"text/plain", s::SemiImplicitI2OE)
    println(io, "SemiImplicitI2OE (semi-implicit advection, Mikula et al.)")
    return print(io, "  └─ cfl: $(s.cfl)")
end

# common integration logic
@noinline function _integrate!(ϕ::MeshField, integrator::TimeIntegrator, terms, reinit, tc, tf, Δt_max, log)
    src = ϕ
    buffers = _alloc_buffers(integrator, ϕ)
    α = cfl(integrator)
    nsteps = 0
    nterms = length(terms)
    update_times = zeros(nterms)
    compute_times = zeros(nterms)
    while tc <= tf - eps(tc)
        t_step = time_ns()
        fill!(update_times, 0.0)
        fill!(compute_times, 0.0)

        reinit_time, did_reinit = _timed_reinit!(src, reinit, nsteps)

        Δt_cfl = α * compute_cfl(terms, src, tc)
        Δt = min(Δt_max, Δt_cfl, tf - tc)

        src, buffers = _advance!(integrator, src, buffers, terms, tc, Δt, update_times, compute_times)

        tc += Δt
        nsteps += 1
        ϕ_min, ϕ_max = _level_set_extrema(src)
        _push_record!(log, tc, t_step, reinit_time, did_reinit, update_times, compute_times, ϕ_min, ϕ_max)
        @debug tc, Δt
    end
    src === ϕ || copy!(ϕ, src)
    return ϕ
end

# --- ForwardEuler ---

_alloc_buffers(::ForwardEuler, ϕ) = (deepcopy(ϕ),)

function _advance!(::ForwardEuler, src, (dst,), terms, tc, Δt, update_times, compute_times)
    update_bcs!(src, tc)
    _clear_buffer!(dst)
    for I in eachindex(src)
        dst[I] = src[I]
    end
    for k in eachindex(terms)
        t0 = time_ns()
        update_term!(terms[k], src, tc)
        update_times[k] += (time_ns() - t0) / 1.0e9
        t0 = time_ns()
        for I in eachindex(src)
            dst[I] -= Δt * _compute_term(terms[k], src, I, tc)
        end
        compute_times[k] += (time_ns() - t0) / 1.0e9
    end
    return dst, (src,)
end

# --- RK2 (Heun's method) ---

_alloc_buffers(::RK2, ϕ) = (deepcopy(ϕ), deepcopy(ϕ))

function _advance!(::RK2, src, (pred, corr), terms, tc, Δt, update_times, compute_times)
    # Stage 1: predictor and half-step accumulator
    update_bcs!(src, tc)
    _clear_buffer!(pred)
    _clear_buffer!(corr)
    for I in eachindex(src)
        pred[I] = src[I]
        corr[I] = src[I]
    end
    for k in eachindex(terms)
        t0 = time_ns()
        update_term!(terms[k], src, tc)
        update_times[k] += (time_ns() - t0) / 1.0e9
        t0 = time_ns()
        for I in eachindex(src)
            v = _compute_term(terms[k], src, I, tc)
            pred[I] -= Δt * v
            corr[I] -= 0.5 * Δt * v
        end
        compute_times[k] += (time_ns() - t0) / 1.0e9
    end
    # Stage 2: correct with slope at predictor
    update_bcs!(pred, tc + Δt)
    for k in eachindex(terms)
        t0 = time_ns()
        update_term!(terms[k], pred, tc + Δt)
        update_times[k] += (time_ns() - t0) / 1.0e9
        t0 = time_ns()
        for I in eachindex(src)
            corr[I] -= 0.5 * Δt * _compute_term(terms[k], pred, I, tc + Δt)
        end
        compute_times[k] += (time_ns() - t0) / 1.0e9
    end
    return corr, (src, pred)
end

# --- RK3 (Shu-Osher TVD) ---

_alloc_buffers(::RK3, ϕ) = (deepcopy(ϕ), deepcopy(ϕ))

function _advance!(::RK3, src, (buf1, buf2), terms, tc, Δt, update_times, compute_times)
    # Stage 1: buf1 = src - Δt*L(src)
    update_bcs!(src, tc)
    _clear_buffer!(buf1)
    for I in eachindex(src)
        buf1[I] = src[I]
    end
    for k in eachindex(terms)
        t0 = time_ns()
        update_term!(terms[k], src, tc)
        update_times[k] += (time_ns() - t0) / 1.0e9
        t0 = time_ns()
        for I in eachindex(src)
            buf1[I] -= Δt * _compute_term(terms[k], src, I, tc)
        end
        compute_times[k] += (time_ns() - t0) / 1.0e9
    end
    # Stage 2: buf2 = 3/4*src + 1/4*(buf1 - Δt*L(buf1))
    update_bcs!(buf1, tc + Δt)
    _clear_buffer!(buf2)
    for I in eachindex(src)
        buf2[I] = 0.75 * src[I] + 0.25 * buf1[I]
    end
    for k in eachindex(terms)
        t0 = time_ns()
        update_term!(terms[k], buf1, tc + Δt)
        update_times[k] += (time_ns() - t0) / 1.0e9
        t0 = time_ns()
        for I in eachindex(src)
            buf2[I] -= 0.25 * Δt * _compute_term(terms[k], buf1, I, tc + Δt)
        end
        compute_times[k] += (time_ns() - t0) / 1.0e9
    end
    # Stage 3: buf1 = (1/3)*src + (2/3)*(buf2 - Δt*L(buf2))
    update_bcs!(buf2, tc + 0.5 * Δt)
    _clear_buffer!(buf1)
    for I in eachindex(src)
        buf1[I] = (src[I] + 2 * buf2[I]) / 3
    end
    for k in eachindex(terms)
        t0 = time_ns()
        update_term!(terms[k], buf2, tc + 0.5 * Δt)
        update_times[k] += (time_ns() - t0) / 1.0e9
        t0 = time_ns()
        for I in eachindex(src)
            buf1[I] -= (2 / 3) * Δt * _compute_term(terms[k], buf2, I, tc + 0.5 * Δt)
        end
        compute_times[k] += (time_ns() - t0) / 1.0e9
    end
    return buf1, (src, buf2)
end

function _integrate!(
        ϕ::MeshField,
        integrator::SemiImplicitI2OE,
        terms,
        reinit,
        tc,
        tf,
        Δt_max,
        log,
    )
    # Check domain compatibility (FullDomain only for now)
    domain(ϕ) isa FullDomain || throw(ArgumentError("SemiImplicitI2OE only supports FullDomain"))

    _validate_i2oe_setup(ϕ, terms)
    term = only(terms)
    vals = values(ϕ)
    old_vals = similar(vals)
    T = float(eltype(vals))
    N = ndims(ϕ)
    velocity_components = ntuple(_ -> zeros(T, size(vals)), N)

    α = cfl(integrator)
    nsteps = 0
    nterms = length(terms)
    update_times = zeros(nterms)
    compute_times = zeros(nterms)
    while tc <= tf - eps(tc)
        t_step = time_ns()
        fill!(update_times, 0.0)
        fill!(compute_times, 0.0)

        reinit_time, did_reinit = _timed_reinit!(ϕ, reinit, nsteps)

        update_bcs!(ϕ, tc)
        _timed_update_terms!(terms, ϕ, tc, update_times)
        Δt_cfl = α * compute_cfl(terms, ϕ, tc)
        Δt = min(Δt_max, Δt_cfl, tf - tc)

        copy!(old_vals, vals)
        _fill_advection_velocity_components!(velocity_components, term, ϕ, tc)

        t0_compute = time_ns()
        _i2oe_global_step!(vals, old_vals, velocity_components, ϕ, Δt)
        compute_times[1] += (time_ns() - t0_compute) / 1.0e9

        tc += Δt
        nsteps += 1
        ϕ_min, ϕ_max = _level_set_extrema(ϕ)
        _push_record!(log, tc, t_step, reinit_time, did_reinit, update_times, compute_times, ϕ_min, ϕ_max)
        @debug tc, Δt
    end
    return ϕ
end

function _validate_i2oe_setup(ϕ, terms)
    length(terms) == 1 && first(terms) isa AdvectionTerm || throw(
        ArgumentError("SemiImplicitI2OE requires exactly one AdvectionTerm"),
    )
    all(size(values(ϕ)) .>= 3) || throw(
        ArgumentError("SemiImplicitI2OE requires at least 3 grid nodes along each dimension"),
    )
    return nothing
end

function _fill_advection_velocity_components!(out, term::AdvectionTerm{<:MeshField}, ϕ::MeshField, _t)
    vel = velocity(term)
    N = ndims(ϕ)
    mesh(vel) == mesh(ϕ) ||
        throw(ArgumentError("advection velocity field must be defined on the same mesh"))
    for I in eachindex(ϕ)
        vI = vel[I]
        for dim in 1:N
            out[dim][I] = vI[dim]
        end
    end
    return out
end

function _fill_advection_velocity_components!(out, term::AdvectionTerm{<:Function}, ϕ::MeshField, t)
    vel = velocity(term)
    N = ndims(ϕ)
    g = mesh(ϕ)
    for I in eachindex(ϕ)
        vI = vel(g[I], t)
        for dim in 1:N
            out[dim][I] = vI[dim]
        end
    end
    return out
end

function _i2oe_global_step!(vals, old_vals, velocity_components, ϕ::MeshField, Δt)
    # Coupled I2OE update: solve one global sparse system built from all neighbors.
    T = eltype(vals)
    Δ = meshsize(ϕ)
    N = ndims(ϕ)
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
    N = ndims(vals)
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
        Iq = CartesianIndex(ntuple(s -> s == dim ? clamp(Ioff[s], first(ax), last(ax)) : Ioff[s], N))
        return (zero(T), one(T), Iq, zero(T))
    elseif bc isa LinearExtrapolationBC
        i, a, b = Ioff[dim], first(ax), last(ax)
        Ion_d, Iin_d, dist = i < a ? (a, a + 1, a - i) : (b, b - 1, i - b)
        Ion = CartesianIndex(ntuple(s -> s == dim ? Ion_d : Ioff[s], N))
        Iin = CartesianIndex(ntuple(s -> s == dim ? Iin_d : Ioff[s], N))
        Ion == I || throw(
            ArgumentError("SemiImplicitI2OE expected nearest ghost cell for LinearExtrapolationBC"),
        )
        return (one(T) + T(dist), -T(dist), Iin, zero(T))
    elseif bc isa DirichletBC
        xghost = _getindex(grid, Ioff)
        return (zero(T), zero(T), nothing, T(bc.f(xghost, bc.t)))
    else
        error("boundary condition $bc is not supported by SemiImplicitI2OE")
    end
end

function _i2oe_face_velocity(velcomp, I, rel, _dim)
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
