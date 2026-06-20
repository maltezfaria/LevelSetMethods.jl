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
    _interpolation_matrix(order, T)

Build the 1D interpolation matrix mapping the `stencil_order + 1` equispaced nodal values on
a super-cell to the `order + 1` coefficients of the Bernstein basis on the central cell.

The matrix depends only on `order` and the element type `T` — not on the cell or the spatial
dimension — so it is built once per [`InterpolatedField`](@ref) and shared (read-only) by
every evaluating task.
"""
function _interpolation_matrix(order::Int, ::Type{T}) where {T}
    # Map `order+1` values at equispaced nodes in [0,1] to the coefficients of the Bernstein
    # basis on [floor(order/2)/order, ceil(order/2)/order], symmetric around 0.5 and
    # containing the central node at 0.5. For even `order` we use a stencil of size order+1
    # and do a least-squares fit.
    stencil_order = isodd(order) ? order : order + 1
    nc, nv = order + 1, stencil_order + 1
    nodes = ntuple(i -> (i - 1) / stencil_order, nv)
    a, b = (stencil_order - 1) / (2 * stencil_order), (stencil_order + 1) / (2 * stencil_order)
    B = (i, k, x) -> binomial(k, i) * (x^i) * ((1 - x)^(k - i))
    V = [B(j - 1, order, (nodes[i] - a) / (b - a)) for i in 1:nv, j in 1:nc]
    return Matrix{T}(pinv(V))
end

"""
    mutable struct InterpolationScratch{N, T}

Per-task working memory and per-cell memo for evaluating an [`InterpolatedField`](@ref). It
holds the scratch arrays written on every cell fill (`coeffs`, `vals`, `temp1`, `temp2`)
together with the memo key (`Ic`, `gen`) identifying which cell the current `coeffs` are for.
It carries no interpolation matrix (that is shared by the field) and no reference to the
underlying field.

Construct via `InterpolationScratch(N, order, T)`.

!!! warning "Task confinement"
    All fields are mutable state and must be used by one task at a time;
    [`InterpolatedField`](@ref) guarantees this by keeping one `InterpolationScratch` per
    task (via `OncePerTask`).
"""
mutable struct InterpolationScratch{N, T}
    coeffs::Array{T, N}
    vals::Array{T, N}
    temp1::Vector{T}
    temp2::Vector{T}
    Ic::CartesianIndex{N}   # cached cell; sentinel CartesianIndex(0,…,0) means "none"
    gen::UInt               # field generation `coeffs` was filled at (see InterpolatedField)
end

"""
    InterpolationScratch(N::Int, order::Int, T::Type)

Allocate the per-task scratch for `N`-dimensional fields with element type `T` and polynomial
interpolation of degree `order`.
"""
function InterpolationScratch(N::Int, order::Int, ::Type{T}) where {T}
    stencil_order = isodd(order) ? order : order + 1
    nc, nv = order + 1, stencil_order + 1
    coeffs = zeros(T, ntuple(_ -> nc, N))
    vals = zeros(T, ntuple(_ -> nv, N))
    buf_size = N > 1 ? nc * nv^(N - 1) : 0
    temp1 = zeros(T, buf_size)
    temp2 = zeros(T, buf_size)
    Ic = CartesianIndex(ntuple(_ -> 0, N))
    return InterpolationScratch{N, T}(coeffs, vals, temp1, temp2, Ic, UInt(0))
end

"""
    fill_coefficients!(ϕ::AbstractMeshField, mat, scratch::InterpolationScratch, base_idxs::CartesianIndex)

Fill `scratch.coeffs` with the Bernstein coefficients for the cell at `base_idxs`, using the
shared interpolation matrix `mat` and the surrounding stencil of grid values of `ϕ`.
"""
@inline function fill_coefficients!(
        ϕ::AbstractMeshField, mat::Matrix, scratch::InterpolationScratch, base_idxs::CartesianIndex{N},
    ) where {N}
    nc, nn = size(mat)
    KS = nn - 1 # order of the interpolation stencil
    off = -(KS - 1) ÷ 2
    # Gather grid values into vals. Since ϕ may generate values on demand via BCs,
    # we can't just copy or have a view.
    for I in CartesianIndices(scratch.vals)
        J = CartesianIndex(ntuple(d -> base_idxs[d] + off + I[d] - 1, N))
        @inbounds scratch.vals[I] = ϕ[J]
    end
    _apply_kron!(scratch.coeffs, mat, scratch.vals, scratch.temp1, scratch.temp2)
    return ϕ
end

"""
    mutable struct InterpolatedField{F, T, B}

A continuous field obtained by equipping a discrete [`AbstractMeshField`](@ref) with
piecewise polynomial interpolation. Evaluating `cf(x)` returns the value of the local
[`BernsteinPolynomial`](@ref) on the cell containing `x`; [`gradient`](@ref) and
[`hessian`](@ref) differentiate the same local patch.

The Bernstein basis is chosen for its convex-hull property: the interpolant on a cell is
bounded by the extrema of its coefficients, which [`proven_empty`](@ref) exploits to
discard cells with no interface or interior.

Construct via `InterpolatedField(ϕ, order)`.

!!! note "Thread safety"
    Evaluating concurrently from multiple tasks is safe (each task gets its own scratch; the
    shared `mat` is only read). Mutating the field (`setindex!`, `copy!`) while other tasks
    evaluate it is not safe; mutations bump the `gen` counter, which invalidates every task's
    cached cell coefficients.
"""
mutable struct InterpolatedField{F <: AbstractMeshField, T, B <: Base.OncePerTask}
    const field::F
    const order::Int
    const mat::Matrix{T}  # shared, read-only interpolation operator (built once)
    const buffer::B       # OncePerTask{InterpolationScratch}: per-task scratch + memo
    gen::UInt             # bumped on mutation to invalidate per-task caches
end

function InterpolatedField(ϕ::AbstractMeshField, order::Integer)
    N = ndims(ϕ)
    # Ensure boundary conditions for stencil access.
    # If ϕ has no boundary conditions, wrap it with ExtrapolationBC of the same order
    if !has_boundary_conditions(ϕ)
        bc = ExtrapolationBC(order)
        ϕ = _add_boundary_conditions(ϕ, bc)
    end
    T = valtype(ϕ)
    mat = _interpolation_matrix(order, T)
    buffer = Base.OncePerTask{InterpolationScratch{N, T}}(() -> InterpolationScratch(N, order, T))
    return InterpolatedField{typeof(ϕ), T, typeof(buffer)}(ϕ, order, mat, buffer, UInt(0))
end
InterpolatedField(ϕ::AbstractMeshField; order::Integer) = InterpolatedField(ϕ, order)

# Restrict an interpolated field to a narrow band, preserving its interpolation order. Lives
# here rather than with the other `NarrowBandMeshField` constructors because it dispatches on
# `InterpolatedField`, which is only defined in this file.
function NarrowBandMeshField(cf::InterpolatedField; nlayers::Int = 3)
    return InterpolatedField(NarrowBandMeshField(cf.field; nlayers), cf.order)
end

"""
    _itp_buffer(cf::InterpolatedField)

Return the calling task's private [`InterpolationScratch`](@ref) for `cf`, lazily created on
first use and cached per task by the `OncePerTask` buffer. No lock is needed and concurrent
tasks never share scratch memory.
"""
@inline _itp_buffer(cf::InterpolatedField) = cf.buffer()

mesh(cf::InterpolatedField) = mesh(cf.field)
compute_index(cf::InterpolatedField, x) = compute_index(mesh(cf), x)

# invalidation signaled by bumping the generation counter
_invalidate!(cf::InterpolatedField) = (cf.gen += 1; cf)

function Base.setindex!(cf::InterpolatedField, val, I...)
    setindex!(cf.field, val, I...)
    return _invalidate!(cf)
end

function Base.copy!(dest::InterpolatedField, src::AbstractMeshField)
    copy!(dest.field, src)
    return _invalidate!(dest)
end

function Base.copy!(dest::InterpolatedField, src::InterpolatedField)
    copy!(dest.field, src.field)
    return _invalidate!(dest)
end

# display methods
function _show_fields(io::IO, cf::InterpolatedField; prefix = "  ")
    return _show_fields(io, cf.field; prefix)
end

function Base.show(io::IO, mime::MIME"text/plain", cf::InterpolatedField)
    print(io, "InterpolatedField (order $(cf.order)) wrapping ")
    return show(io, mime, cf.field)
end

"""
    make_interpolant(cf::InterpolatedField, I::CartesianIndex)

Return a `BernsteinPolynomial` for the cell at multi-index `I`, lazily computing
(and caching) its Bernstein coefficients from the surrounding stencil of grid values.

!!! warning "Aliased coefficients"
    The returned polynomial's `coeffs` array aliases the calling task's scratch buffer.
    It remains valid until the same task calls `make_interpolant` on `cf` with a
    different cell index (or `cf` is mutated). Copy `coefficients(p)` if the polynomial
    must outlive the current cell iteration.
"""
function make_interpolant(cf::InterpolatedField, I::CartesianIndex)
    scratch = _itp_buffer(cf)
    g = cf.gen
    if !(I == scratch.Ic && scratch.gen == g)
        fill_coefficients!(cf.field, cf.mat, scratch, I)
        scratch.Ic = I
        scratch.gen = g
    end
    cell = _getcell(mesh(cf.field), I)
    return BernsteinPolynomial(scratch.coeffs, cell.lc, cell.hc)
end

"""
    local_interpolant(cf::InterpolatedField, x)

Return the local [`BernsteinPolynomial`](@ref) representing the field in the cell
containing the point `x`.

!!! warning "Aliased coefficients"
    The returned polynomial aliases the calling task's scratch buffer; see
    [`make_interpolant`](@ref).
"""
local_interpolant(cf::InterpolatedField, x) = make_interpolant(cf, compute_index(cf, x))

"""
    cell_extrema(cf::InterpolatedField, I::CartesianIndex)

Compute the minimum and maximum values of the local Bernstein interpolant in cell `I`.
"""
function cell_extrema(cf::InterpolatedField, I::CartesianIndex)
    p = make_interpolant(cf, I)
    return extrema(coefficients(p))
end

"""
    proven_empty(cf::InterpolatedField, I::CartesianIndex; surface=false)

Return `true` if cell `I` is guaranteed to contain no interface (when
`surface=true`) or no interior (when `surface=false`), based on the convex-hull
property of the Bernstein basis.
"""
function proven_empty(cf::InterpolatedField, I::CartesianIndex; surface = false)
    m, M = cell_extrema(cf, I)
    return surface ? (m * M > 0) : (m > 0)
end

# Evaluate the field, its gradient, and its Hessian at a point `x` by differentiating the
# local Bernstein patch. `x` may be any point-like object (`SVector`, `Tuple`, or
# `AbstractVector`); the `Real...` methods let callers pass coordinates as separate scalars.
@inline (cf::InterpolatedField)(x) = local_interpolant(cf, x)(x)
@inline (cf::InterpolatedField)(x::Real...) = cf(x)

"""
    gradient(cf::InterpolatedField, x)

Evaluate the spatial gradient vector of the interpolated field at the point `x`.
"""
gradient(cf::InterpolatedField, x) = gradient(local_interpolant(cf, x), x)
gradient(cf::InterpolatedField, x::Real...) = gradient(cf, x)

"""
    hessian(cf::InterpolatedField, x)

Evaluate the spatial Hessian matrix of the interpolated field at the point `x`.
"""
hessian(cf::InterpolatedField, x) = hessian(local_interpolant(cf, x), x)
hessian(cf::InterpolatedField, x::Real...) = hessian(cf, x)

"""
    value_and_gradient(cf::InterpolatedField, x)

Evaluate the value and spatial gradient vector of the interpolated field at the point `x`.
"""
value_and_gradient(cf::InterpolatedField, x) = value_and_gradient(local_interpolant(cf, x), x)
value_and_gradient(cf::InterpolatedField, x::Real...) = value_and_gradient(cf, x)

"""
    value_gradient_hessian(cf::InterpolatedField, x)

Evaluate the value, spatial gradient vector, and Hessian matrix of the interpolated field
at the point `x`.
"""
value_gradient_hessian(cf::InterpolatedField, x) = value_gradient_hessian(local_interpolant(cf, x), x)
value_gradient_hessian(cf::InterpolatedField, x::Real...) = value_gradient_hessian(cf, x)
