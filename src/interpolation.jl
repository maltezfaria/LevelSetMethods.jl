"""
    compute_index(ϕ::AbstractMeshField, x)

Compute the multi-index of the cell containing point `x`. If `x` is outside
the domain, the index of the closest cell is returned (clamped to the boundary).
"""
function compute_index(ϕ::AbstractMeshField, x)
    grid = mesh(ϕ)
    N = ndims(ϕ)
    cell_ax = cellindices(grid)
    return ntuple(
        d -> clamp(
            floor(Int, (x[d] - grid.lc[d]) / meshsize(grid)[d]) + 1,
            first(cell_ax.indices[d]), last(cell_ax.indices[d])
        ),
        N,
    ) |> CartesianIndex
end

"""
    fill_coefficients!(ϕ::AbstractMeshField, base_idxs::CartesianIndex)

Fill the internal `InterpolationData` buffer of `ϕ` with the Bernstein coefficients
for the cell at `base_idxs`.
"""
@inline function fill_coefficients!(ϕ::AbstractMeshField, base_idxs::CartesianIndex{N}) where {N}
    itp = interp_data(ϕ)
    mat = itp.mat
    nc, nn = size(mat)
    KS = nn - 1 # order of the interpolation stencil
    off = -(KS - 1) ÷ 2
    # Gather grid values into vals. Since ϕ may generate values on demand via BCs,
    # we can't just copy or have a view.
    for I in CartesianIndices(itp.vals)
        J = CartesianIndex(ntuple(d -> base_idxs[d] + off + I[d] - 1, N))
        @inbounds itp.vals[I] = ϕ[J]
    end
    _apply_kron!(itp.coeffs, mat, itp.vals, itp.temp1, itp.temp2)
    itp.Ic = base_idxs
    return ϕ
end

"""
    make_interpolant(ϕ::AbstractMeshField, I::CartesianIndex)

Return a `BernsteinPolynomial` for the cell at multi-index `I`, lazily computing
(and caching) its Bernstein coefficients from the surrounding stencil of grid values.

!!! warning "Aliased coefficients"
    The returned polynomial's `coeffs` array aliases the internal buffer of `ϕ`'s
    `InterpolationData`. It remains valid only until the next `make_interpolant` call on
    `ϕ` with a different cell index. Copy `coefficients(p)` if the polynomial must outlive
    the current cell iteration.
"""
function make_interpolant(ϕ::AbstractMeshField, I::CartesianIndex)
    itp = interp_data(ϕ)
    I == itp.Ic || fill_coefficients!(ϕ, I)
    cell = _getcell(mesh(ϕ), I)
    return BernsteinPolynomial(itp.coeffs, cell.lc, cell.hc)
end

"""
    cell_extrema(ϕ::AbstractMeshField, I::CartesianIndex)

Compute the minimum and maximum values of the local Bernstein interpolant in cell `I`.
"""
function cell_extrema(ϕ::AbstractMeshField, I::CartesianIndex)
    p = make_interpolant(ϕ, I)
    return extrema(coefficients(p))
end

"""
    proven_empty(ϕ::AbstractMeshField, I::CartesianIndex; surface=false)

Return `true` if cell `I` is guaranteed to contain no interface (when
`surface=true`) or no interior (when `surface=false`), based on the convex-hull
property of the Bernstein basis.
"""
function proven_empty(ϕ::AbstractMeshField, I::CartesianIndex; surface = false)
    m, M = cell_extrema(ϕ, I)
    return surface ? (m * M > 0) : (m > 0)
end

# Make a MeshField with InterpolationData callable as a continuous function.
@inline function (ϕ::MeshField{V, M, B, <:InterpolationData})(x) where {V, M, B}
    I = compute_index(ϕ, x)
    p = make_interpolant(ϕ, I)
    return p(x)
end

@inline (ϕ::MeshField{V, M, B, <:InterpolationData})(x::Vararg{Real}) where {V, M, B} =
    ϕ(SVector(x))
@inline (ϕ::MeshField{V, M, B, <:InterpolationData})(x::Tuple) where {V, M, B} =
    ϕ(SVector(x))

# Make a NarrowBandMeshField with InterpolationData callable as a continuous function.
@inline function (nb::NarrowBandMeshField{V, M, B, T, <:InterpolationData})(x) where {V, M, B, T}
    I = compute_index(nb, x)
    p = make_interpolant(nb, I)
    return p(x)
end

@inline (nb::NarrowBandMeshField{V, M, B, T, <:InterpolationData})(x::Vararg{Real}) where {V, M, B, T} =
    nb(SVector(x))
@inline (nb::NarrowBandMeshField{V, M, B, T, <:InterpolationData})(x::Tuple) where {V, M, B, T} =
    nb(SVector(x))
