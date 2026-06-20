"""
    struct BernsteinPolynomial{N, T}

A multidimensional Bernstein polynomial with coefficients of type `T` in `N` dimensions,
defined on a hyperrectangle `[low_corner[i], high_corner[i]]`.
"""
struct BernsteinPolynomial{N, T}
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
    gradient(p::BernsteinPolynomial, x)

Compute the gradient of `p` at point `x` using ForwardDiff.
"""
function gradient(p::BernsteinPolynomial, x)
    return ForwardDiff.gradient(p, SVector{length(x)}(x))
end

"""
    hessian(p::BernsteinPolynomial, x)

Compute the hessian of `p` at point `x` using ForwardDiff.
"""
function hessian(p::BernsteinPolynomial, x)
    return ForwardDiff.hessian(p, SVector{length(x)}(x))
end

"""
    value_and_gradient(p::BernsteinPolynomial, x)

Fused computation of the value and gradient of `p` at point `x` using ForwardDiff.
"""
function value_and_gradient(p::BernsteinPolynomial, x)
    res = ForwardDiff.gradient!(DiffResults.GradientResult(x), p, x)
    return DiffResults.value(res), DiffResults.gradient(res)
end

"""
    value_gradient_hessian(p::BernsteinPolynomial, x)

Fused computation of the value, gradient, and hessian of `p` at point `x` using ForwardDiff.
"""
function value_gradient_hessian(p::BernsteinPolynomial, x)
    res = ForwardDiff.hessian!(DiffResults.HessianResult(x), p, x)
    return DiffResults.value(res), DiffResults.gradient(res), DiffResults.hessian(res)
end
