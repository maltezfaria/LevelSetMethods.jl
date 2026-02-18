"""
    abstract type LevelSetTerm

A typical term in a level-set evolution equation.
"""
abstract type LevelSetTerm end

function compute_cfl(terms, ϕ, t)
    return minimum(terms) do term
        return _compute_cfl(term, ϕ, t)
    end
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
AdvectionTerm(𝐮, scheme = WENO5(), func = (x...) -> nothing) = AdvectionTerm(𝐮, scheme, func)

function update_term!(term::AdvectionTerm, ϕ, t)
    u = velocity(term)
    f = update_func(term)
    return f(u, ϕ, t)
end

Base.show(io::IO, t::AdvectionTerm) = print(io, "𝐮 ⋅ ∇ ϕ")

@inline function _compute_term(term::AdvectionTerm{V}, ϕ, I, t) where {V}
    sch = scheme(term)
    N = dimension(ϕ)
    𝐮 = if V <: MeshField
        velocity(term)[I]
    elseif V <: Function
        x = mesh(ϕ)[I]
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
        x = mesh(ϕ)[I]
        velocity(term)(x, t)
    else
        error("velocity field type $V not supported")
    end
    Δx = meshsize(ϕ)
    return 1 / maximum(abs.(𝐮) ./ Δx)
end

"""
    struct CurvatureTerm{V,M} <: LevelSetTerm

Level-set curvature term representing `bκ|∇ϕ|`, where `κ = ∇ ⋅ (∇ϕ/|∇ϕ|) ` is
the curvature.
"""
struct CurvatureTerm{V} <: LevelSetTerm
    b::V
end
coefficient(cterm::CurvatureTerm) = cterm.b

Base.show(io::IO, t::CurvatureTerm) = print(io, "b κ|∇ϕ|")

function _compute_term(term::CurvatureTerm, ϕ, I, t)
    N = dimension(ϕ)
    κ = curvature(ϕ, I)
    b = coefficient(term)
    bI = if b isa MeshField
        b[I]
    elseif b isa Function
        x = mesh(ϕ)[I]
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
        x = mesh(ϕ)[I]
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
    update_func::F = (x...) -> nothing
end
speed(adv::NormalMotionTerm) = adv.speed
update_func(term::NormalMotionTerm) = term.update_func

NormalMotionTerm(v) = NormalMotionTerm(v, (x...) -> nothing)

function update_term!(term::NormalMotionTerm, ϕ, t)
    v = speed(term)
    f = update_func(term)
    return f(v, ϕ, t)
end

Base.show(io::IO, t::NormalMotionTerm) = print(io, "v|∇ϕ|")

function _compute_term(term::NormalMotionTerm, ϕ, I, t)
    N = dimension(ϕ)
    u = speed(term)
    v = if u isa MeshField
        u[I]
    elseif u isa Function
        x = mesh(ϕ)[I]
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
        x = mesh(ϕ)[I]
        u(x, t)
    else
        error("velocity field type $u not supported")
    end
    Δx = minimum(meshsize(ϕ))
    return Δx / abs(v)
end

@inline positive(x) = x > zero(x) ? x : zero(x)
@inline negative(x) = x < zero(x) ? x : zero(x)

# eq. (6.20-6.21)
function g(x, y)
    tmp = zero(x)
    if x > zero(x)
        tmp += x * x
    end
    if y < zero(x)
        tmp += y * y
    end
    return sqrt(tmp)
end

# eq. (6.28)
function limiter(x, y)
    x * y < zero(x) || return zero(x)
    return abs(x) <= abs(y) ? x : y
end

"""
    struct ReinitializationTerm <: LevelSetTerm

Level-set term representing  `sign(ϕ) (|∇ϕ| - 1)`. This `LevelSetTerm` should be
used for reinitializing the level set into a signed distance function: for a sufficiently
large number of time steps this term allows one to solve the Eikonal equation |∇ϕ| = 1.

There are two ways of constructing a `ReinitializationTerm`:

  - using `ReinitializationTerm(ϕ₀::LevelSet)` precomputes the `sign` term on the initial
    level set `ϕ₀`, as in equation 7.5 of Osher and Fedkiw;
  - using `ReinitializationTerm()` constructs a term that computes the `sign` term
    on-the-fly at each time step, as in equation 7.6 of Osher and Fedkiw.
"""
struct ReinitializationTerm{T} <: LevelSetTerm
    S₀::T
end

function ReinitializationTerm(ϕ₀::LevelSet)
    Δx = minimum(meshsize(ϕ₀))
    # equation 7.5 of Osher and Fedkiw
    S₀ = map(CartesianIndices(mesh(ϕ₀))) do I
        return ϕ₀[I] / sqrt(ϕ₀[I]^2 + Δx^2)
    end
    return ReinitializationTerm(S₀)
end
ReinitializationTerm() = ReinitializationTerm(nothing)

function Base.show(io::IO, t::ReinitializationTerm)
    S₀ = t.S₀
    if isnothing(S₀)
        print(io, "sign(ϕ) (|∇ϕ| - 1)")
    else
        print(io, "sign(ϕ₀) (|∇ϕ| - 1)")
    end
    return io
end

function _compute_term(term::ReinitializationTerm, ϕ, I, t)
    S₀ = term.S₀
    norm_∇ϕ = _compute_∇_norm(sign(ϕ[I]), ϕ, I)
    if isnothing(S₀)
        # equation 7.6 of Osher and Fedkiw
        Δx = minimum(meshsize(ϕ))
        S = ϕ[I] / sqrt(ϕ[I]^2 + norm_∇ϕ^2 * Δx^2)
    else
        # precomputed S₀ term
        S = S₀[I]
    end
    return S * (norm_∇ϕ - 1.0)
end

_compute_cfl(term::ReinitializationTerm, ϕ, I, t) = minimum(meshsize(ϕ))

function _compute_∇_norm(v, ϕ, I)
    # FIXME: use version from NormalTerm
    N = dimension(ϕ)
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
