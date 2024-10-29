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
struct CurvatureTerm{V,M} <: LevelSetTerm
    b::MeshField{V,M}
end
coefficient(cterm::CurvatureTerm) = cterm.b

Base.show(io::IO, t::CurvatureTerm) = print(io, "b κ|∇ϕ|")

function _compute_term(term::CurvatureTerm, ϕ, I, t)
    N = dimension(ϕ)
    b = coefficient(term)
    κ = curvature(ϕ, I)
    # compute |∇ϕ|
    ϕ2 = sum(1:N) do dim
        return D⁰(ϕ, I, dim)^2
    end
    # update
    return b[I] * κ * sqrt(ϕ2)
end

function _compute_cfl(term::CurvatureTerm, ϕ, I, t)
    b = coefficient(term)[I]
    Δx = minimum(meshsize(ϕ))
    return (Δx)^2 / (2 * abs(b))
end

function curvature(ϕ::LevelSet, I)
    N = dimension(ϕ)
    Δ = minimum(meshsize(ϕ))/100
    if N == 2
        ϕx  = D⁰(ϕ, I, 1)
        ϕy  = D⁰(ϕ, I, 2)
        ϕxx = D2⁰(ϕ, I, 1)
        ϕyy = D2⁰(ϕ, I, 2)
        ϕxy = D2(ϕ, I, (2, 1))
        normsq = ϕx^2 + ϕy^2 + Δ^2
        κ = (ϕxx * ϕy^2 - 2 * ϕy * ϕx * ϕxy + ϕyy * ϕx^2) / normsq^(3 / 2)
        return κ
    elseif N == 3
        ϕx  = D⁰(ϕ, I, 1)
        ϕy  = D⁰(ϕ, I, 2)
        ϕz  = D⁰(ϕ, I, 3)
        ϕxx = D2⁰(ϕ, I, 1)
        ϕyy = D2⁰(ϕ, I, 2)
        ϕzz = D2⁰(ϕ, I, 3)
        ϕxy = D2(ϕ, I, (2, 1))
        ϕxz = D2(ϕ, I, (3, 1))
        ϕyz = D2(ϕ, I, (3, 2))
        normsq = ϕx^2 + ϕy^2 + ϕz^2 + Δ^2
        κ = (
                (ϕyy + ϕzz) * ϕx^2 + (ϕxx + ϕzz) * ϕy^2 + (ϕxx + ϕyy) * ϕz^2
                - 2 * (ϕx * ϕy * ϕxy + ϕx * ϕz * ϕxz + ϕy * ϕz * ϕyz)
            ) / normsq^2
        return κ
    else
        notimplemented()
    end
end

function curvature(ϕ::LevelSet)
    tmp = zeros(size(values(ϕ)))
    for I in eachindex(ϕ)
        tmp[I] = curvature(ϕ, I)
    end
    return tmp
end

"""
    struct NormalMotionTerm{V,M} <: LevelSetTerm

Level-set advection term representing  `v |∇ϕ|`. This `LevelSetTerm` should be
used for internally generated velocity fields; for externally generated
velocities you may use `AdvectionTerm` instead.
"""
@kwdef struct NormalMotionTerm{V,M} <: LevelSetTerm
    speed::MeshField{V,M}
end
speed(adv::NormalMotionTerm) = adv.speed

Base.show(io::IO, t::NormalMotionTerm) = print(io, "v|∇ϕ|")

function _compute_term(term::NormalMotionTerm, ϕ, I, t)
    N = dimension(ϕ)
    u = speed(term)
    v = u[I]
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
    u = speed(term)[I]
    Δx = minimum(meshsize(ϕ))
    return Δx / abs(u)
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
