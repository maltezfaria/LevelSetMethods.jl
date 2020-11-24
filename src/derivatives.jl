"""
    D⁰(ϕ::MeshField,I,dim)

Centered finite difference scheme for first order derivative at grid point `I`
along dimension `dim`.
"""
function D⁰(ϕ::MeshField,I,dim)
    h    = meshsize(ϕ,dim)
    Im   = _decrement_index(I,dim)
    Ip   = _increment_index(I,dim)
    return (ϕ[Ip] - ϕ[Im]) / (2h)
end

"""
    D⁺(ϕ::MeshField,I,dim)

Forward finite difference scheme for first order derivative at grid point `I`
along dimension `dim`.
"""
@inline function D⁺(ϕ::MeshField,I,dim)
    h  = meshsize(ϕ,dim)
    Ip = _increment_index(I,dim)
    return (ϕ[Ip] - ϕ[I]) / h
end

"""
    D⁻(ϕ::MeshField,I,dim)

Backward finite difference scheme for first order derivative at grid point `I`
along dimension `dim`.
"""
@inline function D⁻(ϕ::MeshField,I,dim)
    h    = meshsize(ϕ,dim)
    Im   = _decrement_index(I,dim)
    return (ϕ[I] - ϕ[Im]) / h
end

"""
    D2⁰(ϕ::MeshField,I,dim)

Centered finite difference scheme for second order derivative at grid point `I`
along dimension `dim`. E.g. if `dim=1`, this approximates `∂ₓₓ`.
"""
function D2⁰(ϕ::MeshField,I,dim)
    h    = meshsize(ϕ,dim)
    Im   = _decrement_index(I,dim)
    Ip   = _increment_index(I,dim)
    return (ϕ[Ip] - 2ϕ[I] + ϕ[Im]) / h^2
end

"""
    D2(ϕ::MeshField,I,dims)

Finite difference scheme for second order derivative at grid point `I`
along the dimensions `dims`.

If `dims[1] == dims[2]`, it is more efficient to call `D2⁰(ϕ,I,dims[1])`.
"""
function D2(ϕ,I,dims)
    h  = meshsize(ϕ)
    Ip = _increment_index(I,dims[1])
    Im = _decrement_index(I,dims[1])
    (D⁰(ϕ,Ip,dims[2]) - D⁰(ϕ,Im,dims[2]))/(2*h[dims[1]])
end

"""
    D2⁺⁺(ϕ::MeshField,I,dim)

Upward finite difference scheme for second order derivative at grid point `I`
along dimension `dim`. E.g. if `dim=1`, this approximates `∂ₓₓ`.
"""
function D2⁺⁺(ϕ::MeshField,I,dim)
    h    = meshsize(ϕ,dim)
    Ip   = _increment_index(I,dim,1)
    Ipp  = _increment_index(I,dim,2)
    return (ϕ[I] - 2ϕ[Ip] + ϕ[Ipp]) / h^2
end

"""
    D2⁻⁺(ϕ::MeshField,I,dim)

Backward finite difference scheme for second order derivative at grid point `I`
along dimension `dim`. E.g. if `dim=1`, this approximates `∂ₓₓ`.
"""
function D2⁻⁻(ϕ::MeshField,I,dim)
    h    = meshsize(ϕ,dim)
    Im   = _decrement_index(I,dim,1)
    Imm  = _decrement_index(I,dim,2)
    return (ϕ[Imm] - 2ϕ[Im] + ϕ[I]) / h^2
end


# Helper functions
function _increment_index(I::CartesianIndex,dim::Integer,nb::Integer=1)
    N = length(I)
    @assert 1 ≤ dim ≤ length(I)
    return I + CartesianIndex(ntuple(i -> i==dim ? nb : 0,N))
end

function _decrement_index(I::CartesianIndex,dim::Integer,nb::Integer=1)
    N = length(I)
    @assert 1 ≤ dim ≤ length(I)
    return I + CartesianIndex(ntuple(i -> i==dim ? -nb : 0,N))
end
