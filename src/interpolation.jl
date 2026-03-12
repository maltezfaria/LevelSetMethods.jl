using ForwardDiff
using StaticArrays
using LinearAlgebra: mul!, dot, norm

struct BernsteinPolynomial{N, T} <: Function
    coeffs::Array{T, N}
    low_corner::SVector{N, T}
    high_corner::SVector{N, T}
end

"""
    BernsteinPolynomial(c::AbstractArray, lc, hc)

Create a multidimensional [Bernstein
polynomial](https://en.wikipedia.org/wiki/Bernstein_polynomial#Generalizations_to_higher_dimension)
with coefficients `c` defined on the hyperrectangle `[lc[1], hc[1]] × … × [lc[N], hc[N]]`.

Calling `p(x)` evaluates the polynomial at the point `x = (x[1], …, x[N])` by the formula

```math
p(x_1,\\dots,x_D)=\\sum_{i_j=0}^{d_j}c_{i_1\\dots i_D}\\prod_{j=1}^D\\binom{d_j}{i_j}(x_j-l_j)^{i_j}(r_j-x_j)^{d_j-i_j}
```

where ``l_j = lc[j]`` and ``r_j = hc[j]`` are the lower and upper bounds of the
hyperrectangle, respectively, and ``d_j = size(c)[j] - 1`` is the degree of the polynomial
in dimension `j`.

See also [`berninterp`]( @docs/src/95-reference.md).
"""
function BernsteinPolynomial(c::AbstractArray, lc, hc)
    N = ndims(c)
    T = eltype(c)
    coeffs = c isa Array{T, N} ? c : Array{T, N}(c)
    return BernsteinPolynomial{N, T}(coeffs, SVector{N}(lc), SVector{N}(hc))
end

coefficients(p::BernsteinPolynomial) = p.coeffs
low_corner(p::BernsteinPolynomial) = p.low_corner
high_corner(p::BernsteinPolynomial) = p.high_corner
degree(p::BernsteinPolynomial) = size(coefficients(p)) .- 1

# evaluation
function (p::BernsteinPolynomial{N})(x) where {N}
    x_ = SVector{N}(x) # try conversion to SVector
    l = low_corner(p)
    r = high_corner(p)
    x₀ = (x_ - l) ./ (r - l)
    c = coefficients(p)
    return _evaluate_bernstein(x₀, c, Val{N}(), 1, length(c))
end

@fastmath function _evaluate_bernstein(
        x::SVector{N},
        c::AbstractArray,
        ::Val{dim},
        i1,
        len,
    ) where {N, dim}
    n = size(c, dim)
    @inbounds xd = x[dim]
    # inspired by https://personal.math.ubc.ca/~cass/graphics/text/www/pdf/a6.pdf and the
    # FastChebInterp.jl package
    if dim == 1
        s = 1 - xd
        @inbounds P = c[i1]
        C = (n - 1) * xd
        for k in 1:(n - 1)
            @inbounds P = P * s + C * c[i1 + k]
            C = C * (n - k - 1) / (k + 1) * xd
        end
        return P
    else
        Δi = len ÷ n # column-major stride of current dimension

        # we recurse downward on dim for cache locality,
        # since earlier dimensions are contiguous
        dim′ = Val{dim - 1}()

        s = 1 - xd
        P = _evaluate_bernstein(x, c, dim′, i1, Δi)
        C = (n - 1) * xd
        for k in 1:(n - 1)
            P = P * s + C * _evaluate_bernstein(x, c, dim′, i1 + k * Δi, Δi)
            C = C * (n - k - 1) / (k + 1) * xd
        end
        return P
    end
end

# --- Automatic Derivatives via ForwardDiff ---

function gradient(p::BernsteinPolynomial, x)
    return ForwardDiff.gradient(p, SVector{length(x)}(x))
end

function hessian(p::BernsteinPolynomial, x)
    return ForwardDiff.hessian(p, SVector{length(x)}(x))
end

function value_and_gradient(p::BernsteinPolynomial, x)
    x_vec = SVector{length(x)}(x)
    res = DiffResults.GradientResult(x_vec)
    ForwardDiff.gradient!(res, p, x_vec)
    return DiffResults.value(res), DiffResults.gradient(res)
end

function value_gradient_hessian(p::BernsteinPolynomial, x)
    x_vec = SVector{length(x)}(x)
    res = DiffResults.HessianResult(x_vec)
    ForwardDiff.hessian!(res, p, x_vec)
    return DiffResults.value(res), DiffResults.gradient(res), DiffResults.hessian(res)
end

# --- Mesh-based Interpolation ---

"""
    struct PiecewisePolynomialInterpolation{Φ, N, T}

A high-performance piecewise polynomial interpolant for a `MeshField`.
`matrix` is the precomputed transformation mapping patch grid-values to cell-local Bernstein coefficients.
"""
mutable struct PiecewisePolynomialInterpolation{Φ, N, T}
    ϕ::Φ
    mat::Matrix{T}
    coeffs::Array{T, N}
    vals::Array{T, N}
    temp1::Array{T, N}
    temp2::Array{T, N}
    current_I::CartesianIndex{N}

    function PiecewisePolynomialInterpolation(ϕ::Φ, K::Int) where {Φ}
        grid = mesh(ϕ)
        N = dimension(grid)
        T = eltype(ϕ)

        stencil_K = isodd(K) ? K : K + 1
        nc, nn = K + 1, stencil_K + 1
        nodes = ntuple(i -> (i - 1) / stencil_K, Val(nn))
        a, b = (stencil_K - 1) / (2 * stencil_K), (stencil_K + 1) / (2 * stencil_K)
        B(i, k, x) = binomial(k, i) * (x^i) * ((1 - x)^(k - i))
        V = [B(j - 1, K, (nodes[i] - a) / (b - a)) for i in 1:nn, j in 1:nc]

        mat = Matrix{T}(pinv(V))
        coeffs = zeros(T, ntuple(_ -> nc, N))
        vals = zeros(T, ntuple(_ -> nn, N))

        # Intermediate buffers
        if N == 1
            temp1 = zeros(T, 0)
            temp2 = zeros(T, 0)
        elseif N == 2
            temp1 = zeros(T, nc, nn)
            temp2 = zeros(T, 0, 0)
        elseif N == 3
            temp1 = zeros(T, nc, nn, nn)
            temp2 = zeros(T, nc, nc, nn)
        else
            error("N > 3 not supported yet")
        end

        current_I = CartesianIndex(ntuple(_ -> 0, Val(N)))

        return new{Φ, N, T}(ϕ, mat, coeffs, vals, temp1, temp2, current_I)
    end
end

Base.ndims(::PiecewisePolynomialInterpolation{Φ, N}) where {Φ, N} = N

"""
    compute_index(itp::PiecewisePolynomialInterpolation, x)

Compute the multi-index of the cell containing point `x`.
"""
function compute_index(itp::PiecewisePolynomialInterpolation, x)
    grid = mesh(itp.ϕ)
    N = ndims(itp)
    return ntuple(
        d -> floor(Int, (x[d] - grid.lc[d]) / meshsize(grid)[d]) + 1,
        N,
    ) |> CartesianIndex
end

"""
    fill_coefficients!(itp::PiecewisePolynomialInterpolation, base_idxs::CartesianIndex)

Fill the internal buffer of `itp` with the Bernstein coefficients for the cell at `base_idxs`.
"""
@inline function fill_coefficients!(
        itp::PiecewisePolynomialInterpolation{Φ, N, T},
        base_idxs::CartesianIndex{N},
    ) where {Φ, N, T}
    ϕ = itp.ϕ
    mat = itp.mat
    nc, nn = size(mat)
    KS = nn - 1 # order of the interpolation stencil
    off = -(KS - 1) ÷ 2
    vN = Val(N)
    # 1. Gather grid values into vals
    for I in CartesianIndices(itp.vals)
        J = CartesianIndex(ntuple(d -> base_idxs[d] + off + I[d] - 1, N))
        @inbounds itp.vals[I] = ϕ[J]
    end

    # 2. Apply Tensor Transformation dimension-by-dimension using mul!
    if N == 1
        mul!(itp.coeffs, mat, itp.vals)
    elseif N == 2
        mul!(itp.temp1, mat, itp.vals)
        mul!(itp.coeffs, itp.temp1, mat')
    elseif N == 3
        mul!(reshape(itp.temp1, nc, nn * nn), mat, reshape(itp.vals, nn, nn * nn))
        for r in 1:nn
            mul!(view(itp.temp2, :, :, r), view(itp.temp1, :, :, r), mat')
        end
        mul!(reshape(itp.coeffs, nc * nc, nc), reshape(itp.temp2, nc * nc, nn), mat')
    end
    return itp.current_I = base_idxs
end

"""
    make_interpolant(itp::PiecewisePolynomialInterpolation, I::CartesianIndex)

Create a `BernsteinPolynomial` for the cell at multi-index `I`.
"""
function make_interpolant(itp::PiecewisePolynomialInterpolation{Φ, N}, I::CartesianIndex{N}) where {Φ, N}
    I == itp.current_I || fill_coefficients!(itp, I)
    grid = mesh(itp.ϕ)
    h = meshsize(grid)
    lc = grid.lc .+ (SVector(Tuple(I)) .- 1) .* h
    hc = lc .+ h
    return BernsteinPolynomial(itp.coeffs, lc, hc)
end

@inline function (itp::PiecewisePolynomialInterpolation)(x)
    I = compute_index(itp, x)
    p = make_interpolant(itp, I)
    return p(x)
end

@inline (itp::PiecewisePolynomialInterpolation)(x::Vararg{Real}) = itp(SVector(x))
@inline (itp::PiecewisePolynomialInterpolation)(x::Tuple) = itp(SVector(x))

function interpolate(ϕ, order::Int = 3)
    return PiecewisePolynomialInterpolation(ϕ, order)
end

"""
    cell_extrema(itp::PiecewisePolynomialInterpolation, I::CartesianIndex)

Compute the minimum and maximum values of the interpolant in the cell `I`.
"""
function cell_extrema(itp::PiecewisePolynomialInterpolation{Φ, N}, I::CartesianIndex{N}) where {Φ, N}
    fill_coefficients!(itp, I)
    return extrema(itp.coeffs)
end

"""
    proven_empty(itp::PiecewisePolynomialInterpolation, I::CartesianIndex; surface=false)

Return `true` if the cell `I` is guaranteed to not contain the interface (if `surface=true`)
or to not contain any part of the interior (if `surface=false`).
"""
function proven_empty(itp::PiecewisePolynomialInterpolation, I::CartesianIndex; surface = false)
    m, M = cell_extrema(itp, I)
    if surface
        return m * M > 0
    else
        return m > 0
    end
end
