"""
    abstract type LevelSetTerm

A typical term in a level-set evolution equation.
"""
abstract type LevelSetTerm end

"""
    update_term!(term::LevelSetTerm, ϕ, t)

Update the internal state of a `LevelSetTerm` before computing its contribution.  This is
called at each stage of the time integration.
"""
update_term!(::LevelSetTerm, _, _) = nothing

"""
    compute_cfl(terms, ϕ::MeshField, t)

Compute the maximum stable time-step ``Δt`` for the given `terms` and level set `ϕ` at time `t`,
based on the Courant-Friedrichs-Lewy (CFL) condition.
"""
function compute_cfl(terms, ϕ, t)
    Δt = minimum(terms) do term
        return _compute_cfl(term, ϕ, t)
    end
    @assert Δt > 0 "invalid time-step based on CFL condition: Δt = $Δt"
    return Δt
end

# generic method, loops over indices
function _compute_cfl(term::LevelSetTerm, ϕ, t)
    dt = Inf
    for I in eachindex(ϕ)
        cfl = _compute_cfl(term, ϕ, I, t)
        dt = min(dt, cfl)
    end
    return dt
end

struct AdvectionTerm{V, S <: SpatialScheme, F} <: LevelSetTerm
    velocity::V
    scheme::S
    update_func::F
end
velocity(adv::AdvectionTerm) = adv.velocity
scheme(adv::AdvectionTerm) = adv.scheme
update_func(adv::AdvectionTerm) = adv.update_func

"""
    AdvectionTerm(𝐮[, scheme = WENO5(), update_func = nothing])

Advection term representing  `𝐮 ⋅ ∇ϕ`. Available `scheme`s are `Upwind` and `WENO5`.

If passed, `update_func` will be called as `update_func(𝐮, ϕ, t)` before computing the term
at each stage of the time evolution. This can be used to update the velocity field `𝐮`
depending not only on `t`, but also on the current level set `ϕ`.
"""
AdvectionTerm(𝐮, scheme = WENO5(), func = (_...) -> nothing) = AdvectionTerm(𝐮, scheme, func)

function update_term!(term::AdvectionTerm, ϕ, t)
    u = velocity(term)
    f = update_func(term)
    return f(u, ϕ, t)
end

Base.show(io::IO, _::AdvectionTerm) = print(io, "𝐮 ⋅ ∇ ϕ")

@inline function _compute_term(term::AdvectionTerm{V}, ϕ::AbstractMeshField, I, t) where {V}
    sch = scheme(term)
    N = ndims(ϕ)
    𝐮 = if V <: MeshField
        velocity(term)[I]
    elseif V <: Function
        x = getnode(mesh(ϕ), I)
        velocity(term)(x, t)
    else
        error("velocity field type $V not supported")
    end
    # for dimension dim, compute the upwind derivative and multiply by the
    # velocity
    return sum(1:N) do dim
        v = 𝐮[dim]
        if v > 0
            if sch === Upwind()
                return v * D⁻(ϕ, I, dim)
            elseif sch === WENO5()
                return v * weno5⁻(ϕ, I, dim)
            else
                error("scheme $sch not implemented")
            end
        else
            if sch === Upwind()
                return v * D⁺(ϕ, I, dim)
            elseif sch === WENO5()
                return v * weno5⁺(ϕ, I, dim)
            else
                error("scheme $sch not implemented")
            end
        end
    end
end

function _compute_cfl(term::AdvectionTerm{V}, ϕ, I, t) where {V}
    # equation 3.10 of Osher and Fedkiw
    𝐮 = if V <: MeshField
        velocity(term)[I]
    elseif V <: Function
        x = getnode(mesh(ϕ), I)
        velocity(term)(x, t)
    else
        error("velocity field type $V not supported")
    end
    Δx = meshsize(ϕ)
    return 1 / maximum(abs.(𝐮) ./ Δx)
end

"""
    struct CurvatureTerm{V} <: LevelSetTerm

Level-set curvature term representing `bκ|∇ϕ|`, where `κ = ∇ ⋅ (∇ϕ/|∇ϕ|) ` is
the curvature.
"""
struct CurvatureTerm{V} <: LevelSetTerm
    b::V
end
coefficient(cterm::CurvatureTerm) = cterm.b

Base.show(io::IO, _::CurvatureTerm) = print(io, "b κ|∇ϕ|")

function _compute_term(term::CurvatureTerm, ϕ::AbstractMeshField, I, t)
    N = ndims(ϕ)
    κ = curvature(ϕ, I)
    b = coefficient(term)
    bI = if b isa MeshField
        b[I]
    elseif b isa Function
        x = getnode(mesh(ϕ), I)
        b(x, t)
    else
        error("curvature field type $b not supported")
    end
    # compute |∇ϕ|
    ϕ2 = sum(1:N) do dim
        return D⁰(ϕ, I, dim)^2
    end
    # update
    return bI * κ * sqrt(ϕ2)
end

function _compute_cfl(term::CurvatureTerm, ϕ, I, t)
    b = coefficient(term)
    bI = if b isa MeshField
        b[I]
    elseif b isa Function
        x = getnode(mesh(ϕ), I)
        b(x, t)
    else
        error("curvature field type $b not supported")
    end
    Δx = minimum(meshsize(ϕ))
    return (Δx)^2 / (2 * abs(bI))
end

"""
    struct NormalMotionTerm{V,F} <: LevelSetTerm

Level-set advection term representing  `v |∇ϕ|`. This `LevelSetTerm` should be
used for internally generated velocity fields; for externally generated velocities you may
use `AdvectionTerm` instead.

If passed, `update_func` will be called as `update_func(v, ϕ, t)` before computing the term
at each stage of the time evolution.
"""
@kwdef struct NormalMotionTerm{V, F} <: LevelSetTerm
    speed::V
    update_func::F = (_...) -> nothing
end
speed(adv::NormalMotionTerm) = adv.speed
update_func(term::NormalMotionTerm) = term.update_func

NormalMotionTerm(v) = NormalMotionTerm(v, (_...) -> nothing)

function update_term!(term::NormalMotionTerm, ϕ, t)
    v = speed(term)
    f = update_func(term)
    return f(v, ϕ, t)
end

Base.show(io::IO, _::NormalMotionTerm) = print(io, "v|∇ϕ|")

function _compute_term(term::NormalMotionTerm, ϕ::AbstractMeshField, I, t)
    N = ndims(ϕ)
    u = speed(term)
    v = if u isa MeshField
        u[I]
    elseif u isa Function
        x = getnode(mesh(ϕ), I)
        u(x, t)
    else
        error("velocity field type $u not supported")
    end
    ∇⁺², ∇⁻² = sum(1:N) do dim
        # for first-order, dont use +- 0.5h ...
        h = meshsize(ϕ, dim)
        neg = D⁻(ϕ, I, dim) + 0.5h * limiter(D2⁻⁻(ϕ, I, dim), D2⁰(ϕ, I, dim))
        pos = D⁺(ϕ, I, dim) - 0.5h * limiter(D2⁺⁺(ϕ, I, dim), D2⁰(ϕ, I, dim))
        return SVector(
            positive(neg) .^ 2 + negative(pos) .^ 2,
            negative(neg) .^ 2 + positive(pos) .^ 2,
        )
    end
    return positive(v) * sqrt(∇⁺²) + negative(v) * sqrt(∇⁻²)
end

function _compute_cfl(term::NormalMotionTerm, ϕ, I, t)
    u = speed(term)
    v = if u isa MeshField
        u[I]
    elseif u isa Function
        x = getnode(mesh(ϕ), I)
        u(x, t)
    else
        error("velocity field type $u not supported")
    end
    Δx = minimum(meshsize(ϕ))
    return Δx / abs(v)
end

@inline positive(x) = x > zero(x) ? x : zero(x)
@inline negative(x) = x < zero(x) ? x : zero(x)

# eq. (6.28): Minmod limiter — zero when signs differ (TVD), min(|x|,|y|)*sign when same sign
function limiter(x, y)
    x * y > zero(x) || return zero(x)
    return abs(x) <= abs(y) ? x : y
end

"""
    struct EikonalReinitializationTerm <: LevelSetTerm

A [`LevelSetTerm`](@ref) representing `sign(ϕ)(|∇ϕ| - 1)`, which drives the level set
toward a signed distance function by solving the Eikonal equation `|∇ϕ| = 1` via
pseudo-time marching.

!!! note "Comparison with NewtonReinitializer"
    [`NewtonReinitializer`](@ref) is generally preferred: it is applied between time steps
    (not inside the PDE), preserves the interface to high order, and converges in a single
    pass. `EikonalReinitializationTerm` is a simpler alternative that requires no
    interpolation or KDTree, but needs many time steps to propagate corrections from the
    interface and can cause mass loss.

There are two constructors:

  - `EikonalReinitializationTerm(ϕ₀)`: freezes the sign from the initial level set `ϕ₀`
    (equation 7.5 of Osher & Fedkiw). Recommended when the interface may drift.
  - `EikonalReinitializationTerm()`: recomputes the sign from the current `ϕ` at each
    step (equation 7.6 of Osher & Fedkiw).
"""
struct EikonalReinitializationTerm{T} <: LevelSetTerm
    S₀::T
end

function EikonalReinitializationTerm(ϕ₀::MeshField)
    Δx = minimum(meshsize(ϕ₀))
    # equation 7.5 of Osher and Fedkiw
    S₀ = map(nodeindices(mesh(ϕ₀))) do I
        return ϕ₀[I] / sqrt(ϕ₀[I]^2 + Δx^2)
    end
    return EikonalReinitializationTerm(S₀)
end
EikonalReinitializationTerm() = EikonalReinitializationTerm(nothing)

function Base.show(io::IO, t::EikonalReinitializationTerm)
    S₀ = t.S₀
    if isnothing(S₀)
        print(io, "sign(ϕ) (|∇ϕ| - 1)")
    else
        print(io, "sign(ϕ₀) (|∇ϕ| - 1)")
    end
    return io
end

function _compute_term(term::EikonalReinitializationTerm, ϕ, I, _t)
    S₀ = term.S₀
    if isnothing(S₀)
        # equation 7.6 of Osher and Fedkiw: sign computed from current ϕ
        norm_∇ϕ = _compute_∇_norm(sign(ϕ[I]), ϕ, I)
        Δx = minimum(meshsize(ϕ))
        S = ϕ[I] / sqrt(ϕ[I]^2 + norm_∇ϕ^2 * Δx^2)
    else
        # equation 7.5 of Osher and Fedkiw: upwind direction must match frozen sign
        norm_∇ϕ = _compute_∇_norm(sign(S₀[I]), ϕ, I)
        S = S₀[I]
    end
    return S * (norm_∇ϕ - 1.0)
end

_compute_cfl(::EikonalReinitializationTerm, ϕ, _I, _t) = minimum(meshsize(ϕ))

function _compute_∇_norm(v, ϕ, I)
    N = ndims(ϕ)
    mA0², mB0² = sum(1:N) do dim
        h = meshsize(ϕ, dim)
        A = D⁻(ϕ, I, dim) + 0.5 * h * limiter(D2⁻⁻(ϕ, I, dim), D2⁰(ϕ, I, dim))
        B = D⁺(ϕ, I, dim) - 0.5 * h * limiter(D2⁺⁺(ϕ, I, dim), D2⁰(ϕ, I, dim))
        if v > 0
            SVector(positive(A)^2, negative(B)^2)
        else
            SVector(negative(A)^2, positive(B)^2)
        end
    end
    return sqrt(mA0² + mB0²)
end
