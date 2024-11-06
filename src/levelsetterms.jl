"""
    abstract type LevelSetTerm

A typical term in a level-set evolution equation.
"""
abstract type LevelSetTerm end

function compute_cfl(terms, ϕ, t)
    minimum(terms) do term
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

struct AdvectionTerm{V,S<:SpatialScheme} <: LevelSetTerm
    velocity::V
    scheme::S
end
velocity(adv::AdvectionTerm) = adv.velocity
scheme(adv::AdvectionTerm) = adv.scheme

"""
    AdvectionTerm(𝐮, scheme = WENO5())

Advection term representing  `𝐮 ⋅ ∇ϕ`. Available `scheme`s are `Upwind` and `WENO5`.
"""
AdvectionTerm(𝐮, scheme = WENO5()) = AdvectionTerm(𝐮, scheme)

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
    sum(1:N) do dim
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
    struct NormalMotionTerm{V,M} <: LevelSetTerm

Level-set advection term representing  `v |∇ϕ|`. This `LevelSetTerm` should be
used for internally generated velocity fields; for externally generated velocities you may
use `AdvectionTerm` instead.
"""
@kwdef struct NormalMotionTerm{V} <: LevelSetTerm
    speed::V
end
speed(adv::NormalMotionTerm) = adv.speed

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
used for reinitializing the level set into a signed distance function: for a
sufficiently large number of time steps this term allows one to solve the
Eikonal equation |∇ϕ| = 1.
"""
@kwdef struct ReinitializationTerm <: LevelSetTerm end

Base.show(io::IO, t::ReinitializationTerm) = print(io, "sign(ϕ) (|∇ϕ| - 1)")

function _compute_term(term::ReinitializationTerm, ϕ, I, t)
    v = sign(ϕ[I])
    ∇ = _compute_∇_normal_motion(v, ϕ, I)
    return (∇ - 1.0) * v
end

_compute_cfl(term::ReinitializationTerm, ϕ, I, t) = minimum(meshsize(ϕ))
