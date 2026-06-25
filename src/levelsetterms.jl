"""
    abstract type LevelSetTerm

A typical term in a level-set evolution equation.
"""
abstract type LevelSetTerm end

"""
    update_term!(term::LevelSetTerm, ϕ, t)

Update the internal state of a `LevelSetTerm` before computing its contribution. Called before
the CFL estimate and at each stage of the time integration.
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
    Δt > 0 || throw(ArgumentError("invalid time-step based on CFL condition: Δt = $Δt (check for NaN/Inf in velocity or speed)"))
    return Δt
end

# generic method, loops over indices
function _compute_cfl(term::LevelSetTerm, ϕ, t)
    dt = Inf
    for I in active_nodeindices(ϕ)
        cfl = _compute_cfl(term, ϕ, I, t)
        dt = min(dt, cfl)
    end
    return dt
end

# Evaluate a term's coefficient field (velocity, speed, curvature weight) at node `I`: a
# mesh field is read directly, a function is called as `f(x, t)`.
_eval_field(f::AbstractMeshField, ϕ, I, t) = f[I]
_eval_field(f::Function, ϕ, I, t) = f(getnode(ϕ, I), t)

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

@inline function _compute_term(term::AdvectionTerm, ϕ::AbstractMeshField, I, t)
    sch = scheme(term)
    N = ndims(ϕ)
    𝐮 = _eval_field(velocity(term), ϕ, I, t)
    # upwind derivative along each dimension, biased by the sign of the velocity
    return sum(1:N) do dim
        v = 𝐮[dim]
        v * (v > 0 ? _upwind⁻(sch, ϕ, I, dim) : _upwind⁺(sch, ϕ, I, dim))
    end
end

# Left/right-biased first derivative for a given spatial scheme.
_upwind⁻(::Upwind, ϕ, I, dim) = D⁻(ϕ, I, dim)
_upwind⁺(::Upwind, ϕ, I, dim) = D⁺(ϕ, I, dim)
_upwind⁻(::WENO5, ϕ, I, dim) = weno5⁻(ϕ, I, dim)
_upwind⁺(::WENO5, ϕ, I, dim) = weno5⁺(ϕ, I, dim)

function _compute_cfl(term::AdvectionTerm, ϕ, I, t)
    # CFL for the unsplit multidimensional update: Δt Σ_d |u_d|/Δx_d ≤ 1 (Osher & Fedkiw
    # eq. 3.10). The sum over dimensions — not the max — is what von Neumann analysis requires.
    𝐮 = _eval_field(velocity(term), ϕ, I, t)
    Δx = meshsize(ϕ)
    return 1 / sum(abs.(𝐮) ./ Δx)
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
    bI = _eval_field(coefficient(term), ϕ, I, t)
    # compute |∇ϕ|
    ϕ2 = sum(1:N) do dim
        return D⁰(ϕ, I, dim)^2
    end
    # update
    return bI * κ * sqrt(ϕ2)
end

function _compute_cfl(term::CurvatureTerm, ϕ, I, t)
    bI = _eval_field(coefficient(term), ϕ, I, t)
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
    v = _eval_field(speed(term), ϕ, I, t)
    ∇⁺², ∇⁻² = sum(1:N) do dim
        # second-order ENO: bias the one-sided differences by ±0.5h·minmod(D², D²ᶜ)
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
    # Same structure as the advection CFL: v|∇ϕ| is a Hamilton-Jacobi term whose characteristic
    # speed along dimension d is bounded by |v|, so Δt Σ_d |v|/Δx_d ≤ 1.
    v = _eval_field(speed(term), ϕ, I, t)
    Δx = meshsize(ϕ)
    return 1 / sum(abs(v) ./ Δx)
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

!!! note "Comparison with `reinitialize!`"
    [`reinitialize!`](@ref) (the Newton closest-point method) is generally preferred: it is
    applied between time steps (not inside the PDE), preserves the interface to high order,
    and converges in a single pass. `EikonalReinitializationTerm` is a simpler alternative
    that requires no interpolation or KDTree, but needs many time steps to propagate
    corrections from the interface and can cause mass loss.

There are two constructors:

  - `EikonalReinitializationTerm(ϕ₀)`: freezes the sign from the initial level set `ϕ₀`
    (equation 7.5 of Osher & Fedkiw). Recommended when the interface may drift. `ϕ₀` may be a
    full-grid or narrow-band field; the frozen sign lives on the same active set.
  - `EikonalReinitializationTerm()`: recomputes the sign from the current `ϕ` at each
    step (equation 7.6 of Osher & Fedkiw).
"""
struct EikonalReinitializationTerm{T <: Union{Nothing, AbstractMeshField}} <: LevelSetTerm
    S₀::T
end

# equation 7.5 of Osher and Fedkiw: freeze the smoothed sign of ϕ₀ on its own active set.
# `map` over the field keeps S₀ the same kind as ϕ₀ (dense or band), so it indexes like ϕ.
function EikonalReinitializationTerm(ϕ₀::AbstractMeshField)
    Δx = minimum(meshsize(ϕ₀))
    S₀ = map(v -> v / sqrt(v^2 + Δx^2), ϕ₀)
    return EikonalReinitializationTerm{typeof(S₀)}(S₀)
end
EikonalReinitializationTerm() = EikonalReinitializationTerm{Nothing}(nothing)

function Base.show(io::IO, t::EikonalReinitializationTerm)
    S₀ = t.S₀
    if isnothing(S₀)
        print(io, "sign(ϕ) (|∇ϕ| - 1)")
    else
        print(io, "sign(ϕ₀) (|∇ϕ| - 1)")
    end
    return io
end

function _compute_term(term::EikonalReinitializationTerm, ϕ::AbstractMeshField, I::CartesianIndex, _t)
    S₀ = term.S₀
    if isnothing(S₀)
        # equation 7.6 of Osher and Fedkiw: sign computed from current ϕ
        norm_∇ϕ = _compute_∇_norm(sign(ϕ[I]), ϕ, I)
        Δx = minimum(meshsize(ϕ))
        denom = sqrt(ϕ[I]^2 + norm_∇ϕ^2 * Δx^2)
        S = iszero(denom) ? zero(denom) : ϕ[I] / denom
    else
        # equation 7.5 of Osher and Fedkiw: upwind direction must match frozen sign
        norm_∇ϕ = _compute_∇_norm(sign(S₀[I]), ϕ, I)
        S = S₀[I]
    end
    return S * (norm_∇ϕ - 1)
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
