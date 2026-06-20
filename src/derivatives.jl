# Spatial discretization schemes selecting how a level-set term approximates its derivatives.
abstract type SpatialScheme end

"""
    Upwind <: SpatialScheme

First-order upwind discretization of the spatial derivatives in a level-set term. Cheap and
robust, but introduces significant numerical diffusion; prefer [`WENO5`](@ref) unless cost
is the overriding concern.
"""
struct Upwind <: SpatialScheme end

"""
    WENO5 <: SpatialScheme

Fifth-order WENO (Weighted Essentially Non-Oscillatory) discretization of the spatial
derivatives in a level-set term. More expensive than [`Upwind`](@ref), but much more
accurate in smooth regions while suppressing oscillations near steep gradients.
"""
struct WENO5 <: SpatialScheme end

"""
    D⁰(ϕ,I,dim)

Centered finite difference scheme for first order derivative at grid point `I`
along dimension `dim`.
"""
function D⁰(ϕ::AbstractMeshField, I::CartesianIndex, dim::Int)
    h = meshsize(ϕ, dim)
    Im = _decrement_index(I, dim)
    Ip = _increment_index(I, dim)
    return (ϕ[Ip] - ϕ[Im]) / (2h)
end

"""
    D⁺(ϕ,I,dim)

Forward finite difference scheme for first order derivative at grid point `I`
along dimension `dim`.
"""
function D⁺(ϕ::AbstractMeshField, I::CartesianIndex, dim::Int)
    h = meshsize(ϕ, dim)
    Ip = _increment_index(I, dim)
    return (ϕ[Ip] - ϕ[I]) / h
end

"""
    D⁻(ϕ,I,dim)

Backward finite difference scheme for first order derivative at grid point `I`
along dimension `dim`.
"""
function D⁻(ϕ::AbstractMeshField, I::CartesianIndex, dim::Int)
    h = meshsize(ϕ, dim)
    Im = _decrement_index(I, dim)
    return (ϕ[I] - ϕ[Im]) / h
end

# Fifth-order WENO reconstruction from the five one-sided differences of the biased stencil,
# ordered from the upwind end inward (see section 3.4 of Osher-Fedkiw). Shared by `weno5⁻`/`weno5⁺`.
function _weno5(v1, v2, v3, v4, v5)
    # third-order estimates
    dϕ1 = (1 / 3) * v1 - (7 / 6) * v2 + (11 / 6) * v3
    dϕ2 = -(1 / 6) * v2 + (5 / 6) * v3 + (1 / 3) * v4
    dϕ3 = (1 / 3) * v3 + (5 / 6) * v4 - (1 / 6) * v5
    # smoothness indicators
    S1 = (13 / 12) * (v1 - 2 * v2 + v3)^2 + (1 / 4) * (v1 - 4 * v2 + 3 * v3)^2
    S2 = (13 / 12) * (v2 - 2 * v3 + v4)^2 + (1 / 4) * (v2 - v4)^2
    S3 = (13 / 12) * (v3 - 2 * v4 + v5)^2 + (1 / 4) * (3 * v3 - 4 * v4 + v5)^2
    # fudge factor
    ϵ = 1.0e-6 * max(v1^2, v2^2, v3^2, v4^2, v5^2) + 1.0e-99
    # weights
    α1 = 0.1 / (S1 + ϵ)^2
    α2 = 0.6 / (S2 + ϵ)^2
    α3 = 0.3 / (S3 + ϵ)^2
    ω1 = α1 / (α1 + α2 + α3)
    ω2 = α2 / (α1 + α2 + α3)
    ω3 = α3 / (α1 + α2 + α3)
    # WENO approximation
    return ω1 * dϕ1 + ω2 * dϕ2 + ω3 * dϕ3
end

"""
    weno5⁻(ϕ, I, dim)

Fifth-order WENO (Weighted Essentially Non-Oscillatory) reconstruction of the
derivative at grid point `I` along dimension `dim`, using a left-biased stencil.
"""
function weno5⁻(ϕ::AbstractMeshField, I::CartesianIndex, dim::Int)
    Im = _decrement_index(I, dim)
    Imm = _decrement_index(Im, dim)
    Ip = _increment_index(I, dim)
    Ipp = _increment_index(Ip, dim)
    return _weno5(
        D⁻(ϕ, Imm, dim),
        D⁻(ϕ, Im, dim),
        D⁻(ϕ, I, dim),
        D⁻(ϕ, Ip, dim),
        D⁻(ϕ, Ipp, dim),
    )
end

"""
    weno5⁺(ϕ, I, dim)

Fifth-order WENO (Weighted Essentially Non-Oscillatory) reconstruction of the
derivative at grid point `I` along dimension `dim`, using a right-biased stencil.
"""
function weno5⁺(ϕ::AbstractMeshField, I::CartesianIndex, dim::Int)
    Im = _decrement_index(I, dim)
    Imm = _decrement_index(Im, dim)
    Ip = _increment_index(I, dim)
    Ipp = _increment_index(Ip, dim)
    return _weno5(
        D⁺(ϕ, Ipp, dim),
        D⁺(ϕ, Ip, dim),
        D⁺(ϕ, I, dim),
        D⁺(ϕ, Im, dim),
        D⁺(ϕ, Imm, dim),
    )
end

"""
    D2⁰(ϕ,I,dim)

Centered finite difference scheme for second order derivative at grid point `I`
along dimension `dim`. E.g. if `dim=1`, this approximates `∂ₓₓ`.
"""
function D2⁰(ϕ::AbstractMeshField, I::CartesianIndex, dim::Int)
    h = meshsize(ϕ, dim)
    Im = _decrement_index(I, dim)
    Ip = _increment_index(I, dim)
    return (ϕ[Ip] - 2ϕ[I] + ϕ[Im]) / h^2
end

"""
    D2(ϕ,I,dims)

Finite difference scheme for second order derivative at grid point `I`
along the dimensions `dims`.

If `dims[1] == dims[2]`, it is more efficient to call `D2⁰(ϕ,I,dims[1])`.
"""
function D2(ϕ::AbstractMeshField, I::CartesianIndex, dims::NTuple{2, Int})
    h = meshsize(ϕ, dims[1])
    Ip = _increment_index(I, dims[1])
    Im = _decrement_index(I, dims[1])
    return (D⁰(ϕ, Ip, dims[2]) - D⁰(ϕ, Im, dims[2])) / (2 * h)
end

"""
    D2⁺⁺(ϕ,I,dim)

Forward finite difference scheme for second order derivative at grid point `I`
along dimension `dim`. E.g. if `dim=1`, this approximates `∂ₓₓ`.
"""
function D2⁺⁺(ϕ::AbstractMeshField, I::CartesianIndex, dim::Int)
    h = meshsize(ϕ, dim)
    Ip = _increment_index(I, dim, 1)
    Ipp = _increment_index(I, dim, 2)
    return (ϕ[I] - 2ϕ[Ip] + ϕ[Ipp]) / h^2
end

"""
    D2⁻⁻(ϕ,I,dim)

Backward finite difference scheme for second order derivative at grid point `I`
along dimension `dim`. E.g. if `dim=1`, this approximates `∂ₓₓ`.
"""
function D2⁻⁻(ϕ::AbstractMeshField, I::CartesianIndex, dim::Int)
    h = meshsize(ϕ, dim)
    Im = _decrement_index(I, dim, 1)
    Imm = _decrement_index(I, dim, 2)
    return (ϕ[Imm] - 2ϕ[Im] + ϕ[I]) / h^2
end

# Shift `I` by `nb` steps along dimension `dim` (negative `nb` shifts backward).
_increment_index(I::CartesianIndex, dim::Integer, nb::Integer = 1) =
    Base.setindex(I, I[dim] + nb, dim)
_decrement_index(I::CartesianIndex, dim::Integer, nb::Integer = 1) =
    _increment_index(I, dim, -nb)
