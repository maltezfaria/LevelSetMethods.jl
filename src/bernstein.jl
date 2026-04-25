"""
    struct BernsteinPolynomial{N, T} <: Function

A multidimensional Bernstein polynomial with coefficients of type `T` in `N` dimensions,
defined on a hyperrectangle `[low_corner[i], high_corner[i]]`.
"""
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

`BernsteinPolynomial`s can be differentiated using ForwardDiff; see [`gradient`](@ref) and
[`hessian`](@ref).
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

"""
    gradient(p, x)

Compute the gradient of `p` at point `x` using ForwardDiff.
"""
function gradient(p, x)
    return ForwardDiff.gradient(p, SVector{length(x)}(x))
end

"""
    hessian(p, x)

Compute the hessian of `p` at point `x` using ForwardDiff.
"""
function hessian(p, x)
    return ForwardDiff.hessian(p, SVector{length(x)}(x))
end

function value_and_gradient(p, x)
    res = ForwardDiff.gradient!(DiffResults.GradientResult(x), p, x)
    return DiffResults.value(res), DiffResults.gradient(res)
end

"""
    value_gradient_hessian(p, x)

Fused computation of the value, gradient, and hessian of `p` at point `x` using ForwardDiff.
"""
function value_gradient_hessian(p, x)
    res = ForwardDiff.hessian!(DiffResults.HessianResult(x), p, x)
    return DiffResults.value(res), DiffResults.gradient(res), DiffResults.hessian(res)
end

# Batched right-multiply by Aᵀ: for r in 1:n_batch, C[:,:,r] = B[:,:,r] * Aᵀ,
# where C is (m, n) and B is (m, p), all stored flat in column-major order.
# C and B are accessed via linear indexing
function _rmult!(C, A::Matrix{T}, B, m, p, n, n_batch) where {T}
    @inbounds for r in 1:n_batch
        offset_B = (r - 1) * m * p
        offset_C = (r - 1) * m * n
        for j in 1:n
            for i in 1:m
                s = zero(T)
                for k in 1:p
                    s += A[j, k] * B[offset_B + (k - 1) * m + i]
                end
                C[offset_C + (j - 1) * m + i] = s
            end
        end
    end
    return C
end

# Apply the N-fold Kronecker product (mat ⊗ mat ⊗ … ⊗ mat) to vals using the vec trick,
# writing the result into coeffs. temp1 and temp2 are flat pre-allocated scratch buffers,
# each of length nc * nv^(N-1).
function _apply_kron!(
        coeffs::AbstractArray{T, N},
        mat::Matrix{T},
        vals::Array{T, N},
        temp1::Vector{T},
        temp2::Vector{T},
    ) where {T, N}
    nc, nv = size(mat)
    N == 1 && return mul!(coeffs, mat, vals)
    # Apply mat along each dimension sequentially via batched right-multiply by matᵀ.
    # After step k, dims 1..k have size nc and dims k+1..N have size nv.
    _rmult!(temp1, mat, vals, 1, nv, nc, nv^(N - 1))
    src, dst = temp1, temp2
    for k in 2:N
        n_left = nc^(k - 1)
        n_right = k < N ? nv^(N - k) : 1
        out = k < N ? dst : coeffs
        _rmult!(out, mat, src, n_left, nv, nc, n_right)
        src, dst = dst, src
    end
    return coeffs
end

"""
    mutable struct InterpolationData{N, T}

Buffers and precomputed data for piecewise polynomial interpolation. Stores the 1D
interpolation matrix and scratch arrays needed to evaluate Bernstein coefficients on a
cell, but holds no reference to the underlying field.

Construct via `InterpolationData(N, order, T)`.

!!! warning "Not thread-safe"
    The internal buffers (`coeffs`, `vals`, `temp1`, `temp2`) are shared mutable state.
    Create one `InterpolationData` per thread if concurrent evaluation is needed.
"""
mutable struct InterpolationData{N, T}
    order::Int
    mat::Matrix{T}
    coeffs::Array{T, N}
    vals::Array{T, N}
    Ic::CartesianIndex{N}   # sentinel CartesianIndex(0,…,0) means "no cached cell"
    temp1::Vector{T}
    temp2::Vector{T}
end

"""
    InterpolationData(N::Int, order::Int, T::Type)

Allocate an `InterpolationData` for `N`-dimensional fields with element type `T` and
polynomial interpolation of degree `order`.
"""
function InterpolationData(N::Int, order::Int, ::Type{T}) where {T}
    # Build a 1D interpolation matrix mapping `order+1` values at equispaced nodes in [0,1]
    # and returning the coefficients of the Bernstein basis defined on the interval
    # [floor(order/2)/order, ceil(order/2)/order], which is symmetric around 0.5 and
    # contains the central node at 0.5. When order is even, we use a stencil of size
    # order+1 and do a least-squares fit. This matrix is used to compute the interpolant in
    # a cell given values on a super-cell around it.
    stencil_order = isodd(order) ? order : order + 1
    nc, nv = order + 1, stencil_order + 1
    nodes = ntuple(i -> (i - 1) / stencil_order, nv)
    a, b = (stencil_order - 1) / (2 * stencil_order), (stencil_order + 1) / (2 * stencil_order)
    B = (i, k, x) -> binomial(k, i) * (x^i) * ((1 - x)^(k - i))
    V = [B(j - 1, order, (nodes[i] - a) / (b - a)) for i in 1:nv, j in 1:nc]
    mat = pinv(V)
    coeffs = zeros(T, ntuple(_ -> nc, N))
    vals = zeros(T, ntuple(_ -> nv, N))
    buf_size = N > 1 ? nc * nv^(N - 1) : 0
    temp1 = zeros(T, buf_size)
    temp2 = zeros(T, buf_size)
    Ic = CartesianIndex(ntuple(_ -> 0, N))
    return InterpolationData{N, T}(order, mat, coeffs, vals, Ic, temp1, temp2)
end

function Base.copy(itp::InterpolationData{N, T}) where {N, T}
    return InterpolationData{N, T}(
        itp.order,
        copy(itp.mat),
        copy(itp.coeffs),
        copy(itp.vals),
        CartesianIndex(ntuple(_ -> 0, Val(N))),  # reset sentinel — new owner must rebuild
        copy(itp.temp1),
        copy(itp.temp2),
    )
end
