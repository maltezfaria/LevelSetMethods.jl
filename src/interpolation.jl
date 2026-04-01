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

"""
    struct PiecewisePolynomialInterpolant{Φ, N, T}

A piecewise polynomial interpolant built over a `MeshField`.

The domain is partitioned into cells of the underlying mesh. On each cell a
`BernsteinPolynomial` is fitted to the surrounding stencil of grid values, so
that evaluating the interpolant at a point `x` amounts to:
1. locating the cell containing `x`,
2. (lazily) computing the local Bernstein coefficients, and
3. evaluating the resulting polynomial.

Construct via [`interpolate`](@ref). The returned object `itp` supports:
- `itp(x)` — evaluate at point `x`
- [`make_interpolant`](@ref)`(itp, I)` — local `BernsteinPolynomial` for cell `I`
- [`gradient`](@ref)`(itp, x)` / [`hessian`](@ref)`(itp, x)` — via ForwardDiff
- [`cell_extrema`](@ref)`(itp, I)` — certified bounds via convex-hull property
- [`proven_empty`](@ref)`(itp, I)` — certified absence of interface or interior

The struct is mutable because the local coefficients and stencil values are
cached in place (`coeffs`, `vals`) to avoid allocation on repeated evaluations.

!!! warning "Not thread-safe"
    The internal buffers (`coeffs`, `vals`, `temp1`, `temp2`, `Ic`) are shared
    state. Concurrent evaluation from multiple threads will cause data races.
    Create one interpolant per thread if parallelism is needed.
"""
mutable struct PiecewisePolynomialInterpolant{Φ, N, T}
    ϕ::Φ                    # underlying mesh field
    mat::Matrix{T}          # map from grid to bernstein vals in 1D
    coeffs::Array{T, N}     # buffer for bernstein coeffs
    vals::Array{T, N}       # buffer for grid values in the stencil
    Ic::CartesianIndex{N}   # multi-index of the currently loaded cell
    # scratch buffers for applying the N-fold kron mat to vals
    temp1::Vector{T}
    temp2::Vector{T}
end

function PiecewisePolynomialInterpolant(ϕ::Φ, order::Int) where {Φ}
    grid = mesh(ϕ)
    N = ndims(grid)
    T = eltype(ϕ)

    # Build a 1D interpolation matrix mapping `order+1` values at equispaced nodes in [0,1]
    # and returning the coefficients of the Bernstein basis defined on the interval
    # [floor(order/2)/order, ceil(order/2)/order], which is symmetric around 0.5 and
    # contains the central node at 0.5. When order is even, we use a stencil of size
    # order+1 and do a least-squares fit. This matrix is used to compute the interpolant in
    # a cell given values on a super-cell around it.
    stencil_order = isodd(order) ? order : order + 1
    nc, nv = order + 1, stencil_order + 1
    nodes = ntuple(i -> (i - 1) / stencil_order, Val(nv))
    a, b = (stencil_order - 1) / (2 * stencil_order), (stencil_order + 1) / (2 * stencil_order)
    B = (i, k, x) -> binomial(k, i) * (x^i) * ((1 - x)^(k - i))
    V = [B(j - 1, order, (nodes[i] - a) / (b - a)) for i in 1:nv, j in 1:nc]

    mat = pinv(V)
    coeffs = zeros(T, ntuple(_ -> nc, N))
    vals = zeros(T, ntuple(_ -> nv, N))

    # Intermediate buffers: each needs nc * nv^(N-1) elements (the largest intermediate size)
    buf_size = N > 1 ? nc * nv^(N - 1) : 0
    temp1 = zeros(T, buf_size)
    temp2 = zeros(T, buf_size)

    Ic = CartesianIndex(ntuple(_ -> 0, Val(N)))

    return PiecewisePolynomialInterpolant(ϕ, mat, coeffs, vals, Ic, temp1, temp2)
end

Base.ndims(::PiecewisePolynomialInterpolant{Φ, N}) where {Φ, N} = N

"""
    compute_index(itp::PiecewisePolynomialInterpolant, x)

Compute the multi-index of the cell containing point `x`. If `x` is outside the domain, the
index of the closest cell is returned (clamping to the boundary).
"""
function compute_index(itp::PiecewisePolynomialInterpolant, x)
    grid = mesh(itp.ϕ)
    N = ndims(itp)
    cell_ax = cellindices(grid)
    return ntuple(
        d -> clamp(
            floor(Int, (x[d] - grid.lc[d]) / meshsize(grid)[d]) + 1,
            first(cell_ax.indices[d]), last(cell_ax.indices[d])
        ),
        N,
    ) |> CartesianIndex
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
    fill_coefficients!(itp::PiecewisePolynomialInterpolant, base_idxs::CartesianIndex)

Fill the internal buffer of `itp` with the Bernstein coefficients for the cell at `base_idxs`.
"""
@inline function fill_coefficients!(
        itp::PiecewisePolynomialInterpolant{Φ, N, T},
        base_idxs::CartesianIndex{N},
    ) where {Φ, N, T}
    ϕ = itp.ϕ
    mat = itp.mat
    nc, nn = size(mat)
    KS = nn - 1 # order of the interpolation stencil
    off = -(KS - 1) ÷ 2
    # Gather grid values into vals. Since the field ϕ may generate values on demand via BCs,
    # we can't just copy or have a view
    for I in CartesianIndices(itp.vals)
        J = CartesianIndex(ntuple(d -> base_idxs[d] + off + I[d] - 1, N))
        @inbounds itp.vals[I] = ϕ[J]
    end
    # The Vandermonde matrix is a Kronecker product V = V₁ ⊗ … ⊗ V₁
    _apply_kron!(itp.coeffs, mat, itp.vals, itp.temp1, itp.temp2)
    itp.Ic = base_idxs
    return itp
end

"""
    make_interpolant(itp::PiecewisePolynomialInterpolant, I::CartesianIndex)

Create a `BernsteinPolynomial` for the cell at multi-index `I`.
"""
function make_interpolant(itp::PiecewisePolynomialInterpolant{Φ, N}, I::CartesianIndex{N}) where {Φ, N}
    I == itp.Ic || fill_coefficients!(itp, I)
    cell = getcell(mesh(itp.ϕ), I)
    return BernsteinPolynomial(itp.coeffs, cell.lc, cell.hc)
end

function Base.show(io::IO, ::MIME"text/plain", itp::PiecewisePolynomialInterpolant)
    order = size(itp.mat, 1) - 1
    N = ndims(itp)
    println(io, "PiecewisePolynomialInterpolant")
    println(io, "  ├─ order: $order")
    println(io, "  └─ field: MeshField on CartesianGrid in ℝ$(_superscript(N))")
    return _show_fields(io, itp.ϕ; prefix = "     ")
end

@inline _evaluate(p::P, x) where {P} = p(x)

@inline function (itp::PiecewisePolynomialInterpolant)(x)
    I = compute_index(itp, x)
    p = make_interpolant(itp, I)
    return _evaluate(p, x)
end

@inline (itp::PiecewisePolynomialInterpolant)(x::Vararg{Real}) = itp(SVector(x))
@inline (itp::PiecewisePolynomialInterpolant)(x::Tuple) = itp(SVector(x))

"""
    interpolate(ϕ::MeshField, order::Int = 3)

Create a piecewise polynomial interpolant of the given `order` for `ϕ`.

A deep copy of `ϕ` is made so that the interpolant is independent of future modifications to
`ϕ`. If `ϕ` has no boundary conditions, `ExtrapolationBC{2}` is added automatically on all
sides (this is necessary to evaluate the interpolant near the boundary, where the stencil
may require out-of-bounds values).

The returned object `itp` behaves like a function and supports:
- `itp(x)`: evaluate the interpolant at point `x`
- [`make_interpolant`](@ref)`(itp, I)`: return the local interpolant for cell `I`
- [`gradient`](@ref)`(itp, x)`: gradient at `x` (via `make_interpolant` + ForwardDiff)
- [`hessian`](@ref)`(itp, x)`: hessian at `x` (via `make_interpolant` + ForwardDiff)
- [`cell_extrema`](@ref)`(itp, I)`: lower and upper bounds of the interpolant in cell `I`

To avoid unnecessary copying, call `update!(itp, ϕ)` to update the interpolant's internal
field with new values from `ϕ` while reusing the existing interpolation matrix and scratch
buffers; see [`update!`](@ref) for details.
"""
function interpolate(ϕ, order::Int = 3)
    ϕ_copy = deepcopy(ϕ)
    if !has_boundary_conditions(ϕ_copy)
        N = ndims(mesh(ϕ_copy))
        bc = ntuple(_ -> (ExtrapolationBC{2}(), ExtrapolationBC{2}()), N)
        ϕ_copy = add_boundary_conditions(ϕ_copy, bc)
    end
    return PiecewisePolynomialInterpolant(ϕ_copy, order)
end

"""
    update!(itp::PiecewisePolynomialInterpolant, ϕ)

Copy the values of `ϕ` into the interpolant's internal field and invalidate the cell
cache. This is cheaper than calling [`interpolate`](@ref) again because it reuses the
existing interpolation matrix and scratch buffers.
"""
function update!(itp::PiecewisePolynomialInterpolant{Φ, N}, ϕ) where {Φ, N}
    copy!(itp.ϕ, ϕ)
    itp.Ic = CartesianIndex(ntuple(_ -> 0, Val(N)))
    return itp
end

"""
    cell_extrema(itp::PiecewisePolynomialInterpolant, I::CartesianIndex)

Compute the minimum and maximum values of the interpolant in the cell `I`.
"""
function cell_extrema(itp::PiecewisePolynomialInterpolant{Φ, N}, I::CartesianIndex{N}) where {Φ, N}
    p = make_interpolant(itp, I)
    return extrema(coefficients(p))
end

"""
    proven_empty(itp::PiecewisePolynomialInterpolant, I::CartesianIndex; surface=false)

Return `true` if the cell `I` is guaranteed to not contain the interface (if `surface=true`)
or to not contain any part of the interior (if `surface=false`).

Note that `proven_empty` being `false` does not mean the cell is non-empty, but rather that
we can't guarantee emptiness based on the convex hull property of the Bernstein basis.
"""
function proven_empty(itp::PiecewisePolynomialInterpolant, I::CartesianIndex; surface = false)
    m, M = cell_extrema(itp, I)
    if surface
        return m * M > 0
    else
        return m > 0
    end
end
