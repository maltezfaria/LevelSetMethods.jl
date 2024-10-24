abstract type SpatialScheme end

struct Upwind <: SpatialScheme end

struct WENO5 <: SpatialScheme end

"""
    D⁰(ϕ::CartesianMeshField,I,dim)

Centered finite difference scheme for first order derivative at grid point `I`
along dimension `dim`.
"""
function D⁰(ϕ::CartesianMeshField, I, dim)
    h  = meshsize(ϕ, dim)
    Im = _decrement_index(I, dim)
    Ip = _increment_index(I, dim)
    return (ϕ[Ip] - ϕ[Im]) / (2h)
end

"""
    D⁺(ϕ::CartesianMeshField,I,dim)

Forward finite difference scheme for first order derivative at grid point `I`
along dimension `dim`.
"""
@inline function D⁺(ϕ::CartesianMeshField, I, dim)
    h  = meshsize(ϕ, dim)
    Ip = _increment_index(I, dim)
    return (ϕ[Ip] - ϕ[I]) / h
end

function D⁺⁺(ϕ::CartesianMeshField, I, dim)
    h   = meshsize(ϕ, dim)
    Ip  = _increment_index(I, dim)
    Ipp = _increment_index(I, dim, 2)
    return (-1.5 * ϕ[I] + 2 * ϕ[Ip] - 1 / 2 * ϕ[Ipp]) / h
end

"""
    D⁻(ϕ::CartesianMeshField,I,dim)

Backward finite difference scheme for first order derivative at grid point `I`
along dimension `dim`.
"""
function D⁻(ϕ::CartesianMeshField, I, dim)
    h  = meshsize(ϕ, dim)
    Im = _decrement_index(I, dim)
    return (ϕ[I] - ϕ[Im]) / h
end

function D⁻⁻(ϕ::CartesianMeshField, I, dim)
    h = meshsize(ϕ, dim)
    Im = _decrement_index(I, dim)
    Imm = _decrement_index(I, dim, 2)
    return (1.5 * ϕ[I] - 2 * ϕ[Im] + 1 / 2 * ϕ[Imm]) / h
end

function weno5⁻(ϕ::CartesianMeshField, I, dim)
    # see section 3.4 of Osher-Fedwik
    Im  = _decrement_index(I, dim)
    Imm = _decrement_index(Im, dim)
    Ip  = _increment_index(I, dim)
    Ipp = _increment_index(Ip, dim)
    # finite differences
    v1 = D⁻(ϕ, Imm, dim)
    v2 = D⁻(ϕ, Im, dim)
    v3 = D⁻(ϕ, I, dim)
    v4 = D⁻(ϕ, Ip, dim)
    v5 = D⁻(ϕ, Ipp, dim)
    # third-order estimates
    dϕ1 = (1 / 3) * v1 - (7 / 6) * v2 + (11 / 6) * v3
    dϕ2 = -(1 / 6) * v2 + (5 / 6) * v3 + (1 / 3) * v4
    dϕ3 = (1 / 3) * v3 + (5 / 6) * v4 - (1 / 6) * v5
    # smoothness indicators
    S1 = (13 / 12) * (v1 - 2 * v2 + v3)^2 + (1 / 4) * (v1 - 4 * v2 + 3 * v3)^2
    S2 = (13 / 12) * (v2 - 2 * v3 + v4)^2 + (1 / 4) * (v2 - v4)^2
    S3 = (13 / 12) * (v3 - 2 * v4 + v5)^2 + (1 / 4) * (3 * v3 - 4 * v4 + v5^3)^2
    # fudge factor
    ϵ = 1e-6 * max(v1^2, v2^2, v3^2, v4^2, v5^2) + 1e-99
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

function weno5⁺(ϕ::CartesianMeshField, I, dim)
    # see section 3.4 of Osher-Fedwik
    Im  = _decrement_index(I, dim)
    Imm = _decrement_index(Im, dim)
    Ip  = _increment_index(I, dim)
    Ipp = _increment_index(Ip, dim)
    # finite differences
    v1 = D⁺(ϕ, Ipp, dim)
    v2 = D⁺(ϕ, Ip, dim)
    v3 = D⁺(ϕ, I, dim)
    v4 = D⁺(ϕ, Im, dim)
    v5 = D⁺(ϕ, Imm, dim)
    # third-order estimates
    dϕ1 = (1 / 3) * v1 - (7 / 6) * v2 + (11 / 6) * v3
    dϕ2 = -(1 / 6) * v2 + (5 / 6) * v3 + (1 / 3) * v4
    dϕ3 = (1 / 3) * v3 + (5 / 6) * v4 - (1 / 6) * v5
    # smoothness indicators
    S1 = (13 / 12) * (v1 - 2 * v2 + v3)^2 + (1 / 4) * (v1 - 4 * v2 + 3 * v3)^2
    S2 = (13 / 12) * (v2 - 2 * v3 + v4)^2 + (1 / 4) * (v2 - v4)^2
    S3 = (13 / 12) * (v3 - 2 * v4 + v5)^2 + (1 / 4) * (3 * v3 - 4 * v4 + v5^3)^2
    # fudge factor
    ϵ = 1e-6 * max(v1^2, v2^2, v3^2, v4^2, v5^2) + 1e-99
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
    D2⁰(ϕ::CartesianMeshField,I,dim)

Centered finite difference scheme for second order derivative at grid point `I`
along dimension `dim`. E.g. if `dim=1`, this approximates `∂ₓₓ`.
"""
function D2⁰(ϕ::CartesianMeshField, I, dim)
    h  = meshsize(ϕ, dim)
    Im = _decrement_index(I, dim)
    Ip = _increment_index(I, dim)
    return (ϕ[Ip] - 2ϕ[I] + ϕ[Im]) / h^2
end

"""
    D2(ϕ::CartesianMeshField,I,dims)

Finite difference scheme for second order derivative at grid point `I`
along the dimensions `dims`.

If `dims[1] == dims[2]`, it is more efficient to call `D2⁰(ϕ,I,dims[1])`.
"""
function D2(ϕ, I, dims)
    h  = meshsize(ϕ)
    Ip = _increment_index(I, dims[1])
    Im = _decrement_index(I, dims[1])
    return (D⁰(ϕ, Ip, dims[2]) - D⁰(ϕ, Im, dims[2])) / (2 * h[dims[1]])
end

"""
    D2⁺⁺(ϕ::CartesianMeshField,I,dim)

Upward finite difference scheme for second order derivative at grid point `I`
along dimension `dim`. E.g. if `dim=1`, this approximates `∂ₓₓ`.
"""
function D2⁺⁺(ϕ::CartesianMeshField, I, dim)
    h   = meshsize(ϕ, dim)
    Ip  = _increment_index(I, dim, 1)
    Ipp = _increment_index(I, dim, 2)
    return (ϕ[I] - 2ϕ[Ip] + ϕ[Ipp]) / h^2
end

"""
    D2⁻⁺(ϕ::CartesianMeshField,I,dim)

Backward finite difference scheme for second order derivative at grid point `I`
along dimension `dim`. E.g. if `dim=1`, this approximates `∂ₓₓ`.
"""
function D2⁻⁻(ϕ::CartesianMeshField, I, dim)
    h   = meshsize(ϕ, dim)
    Im  = _decrement_index(I, dim, 1)
    Imm = _decrement_index(I, dim, 2)
    return (ϕ[Imm] - 2ϕ[Im] + ϕ[I]) / h^2
end

# Helper functions
function _increment_index(I::CartesianIndex, dim::Integer, nb::Integer = 1)
    N = length(I)
    @assert 1 ≤ dim ≤ length(I)
    return I + CartesianIndex(ntuple(i -> i == dim ? nb : 0, N))
end

function _decrement_index(I::CartesianIndex, dim::Integer, nb::Integer = 1)
    N = length(I)
    @assert 1 ≤ dim ≤ length(I)
    return I + CartesianIndex(ntuple(i -> i == dim ? -nb : 0, N))
end
