"""
    struct NarrowBandDomain{T} <: AbstractDomain

Domain for a narrow-band level set.

- `halfwidth`: half-width of the narrow band, typically on the order of a few grid spacings.

Active indices are the keys of the associated values dict and need not be stored separately.
"""
struct NarrowBandDomain{T} <: AbstractDomain
    halfwidth::T
end

"""
    const NarrowBandLevelSet{N, T, B}

Alias for [`MeshField`](@ref) on a `CartesianGrid{N,T}` with values stored as
a `Dict{CartesianIndex{N},T}` and a [`NarrowBandDomain{T}`](@ref). `B` is
the type of the boundary conditions.
"""
const NarrowBandLevelSet{N, T, B} =
    MeshField{Dict{CartesianIndex{N}, T}, CartesianGrid{N, T}, B, NarrowBandDomain{T}}

active_indices(nb::NarrowBandLevelSet) = keys(values(nb))
halfwidth(nb::NarrowBandLevelSet) = domain(nb).halfwidth
Base.eachindex(nb::NarrowBandLevelSet) = active_indices(nb)
_eachindex(::NarrowBandDomain, nb) = active_indices(nb)
Base.eltype(::NarrowBandLevelSet{N, T}) where {N, T} = T

# Build the active-index dict by evaluating `f_at_idx(I)` at every grid node and
# keeping only those where `|v| < γ`. `f_at_idx` is either `I -> ϕ[I]` (LevelSet path)
# or `I -> f(grid[I])` (function path).
function _nb_dict(f_at_idx, grid::CartesianGrid{N}, γ::T) where {N, T}
    vals = Dict{CartesianIndex{N}, T}()
    for I in CartesianIndices(grid)
        v = T(f_at_idx(I))
        abs(v) < γ && (vals[I] = v)
    end
    return vals
end

"""
    NarrowBandLevelSet(ϕ::LevelSet, halfwidth::Real; reinitialize = true)

Construct a `NarrowBandLevelSet` from a full-grid `LevelSet`. Active nodes are those
where `|ϕ[I]| < halfwidth`. Boundary conditions are inherited from `ϕ`.

If `reinitialize` is `true` (the default), `ϕ` is first reinitialized to a signed
distance function using [`NewtonReinitializer`](@ref).
"""
function NarrowBandLevelSet(ϕ::LevelSet, halfwidth::Real; reinitialize::Bool = true)
    bcs = boundary_conditions(ϕ)   # preserve the caller's BCs (may be nothing)
    if reinitialize
        ϕ = deepcopy(ϕ)
        # reinit needs BCs for gradient computation; add temporary ones if missing
        if !has_boundary_conditions(ϕ)
            ϕ = add_boundary_conditions(ϕ, ExtrapolationBC(2))
        end
        reinitialize!(ϕ, NewtonReinitializer())
    end
    grid = mesh(ϕ)
    T = float(eltype(ϕ))
    γ = T(halfwidth)
    vals = _nb_dict(I -> ϕ[I], grid, γ)
    return MeshField(vals, grid, bcs, NarrowBandDomain(γ))
end

"""
    NarrowBandLevelSet(ϕ::LevelSet; nlayers = 3, reinitialize = true)

Construct a `NarrowBandLevelSet` with halfwidth automatically computed as
`nlayers * minimum(meshsize(ϕ))`. `nlayers` sets the number of cell layers
on each side of the interface included in the band.
"""
function NarrowBandLevelSet(ϕ::LevelSet; nlayers::Int = 3, reinitialize::Bool = true)
    return NarrowBandLevelSet(ϕ, nlayers * minimum(meshsize(ϕ)); reinitialize)
end

"""
    NarrowBandLevelSet(f, grid::CartesianGrid, halfwidth::Real; bc = nothing)

Construct a `NarrowBandLevelSet` by evaluating `f` at each node of `grid` and
keeping only those where `|f(x)| < halfwidth`. No dense array is allocated.

!!! warning
    Since the `halfwidth` threshold is applied to the raw values of `f`, the resulting band
    width in physical space will only match `halfwidth` if `f` is already a signed distance
    function. Otherwise the band width will depend on the gradient of `f` near the interface
    and may not correspond to a fixed number of cell layers.
"""
function NarrowBandLevelSet(f, grid::CartesianGrid, halfwidth::Real; bc = nothing)
    T = float(eltype(eltype(grid)))
    γ = T(halfwidth)
    vals = _nb_dict(I -> f(grid[I]), grid, γ)
    return MeshField(vals, grid, bc, NarrowBandDomain(γ))
end

"""
    NarrowBandLevelSet(f, grid::CartesianGrid; nlayers = 8, bc = nothing)

Construct a `NarrowBandLevelSet` with halfwidth automatically computed as
`nlayers * minimum(meshsize(grid))`.

!!! warning
    The `nlayers` interpretation is only correct if `f` is already a signed distance
    function. Otherwise the band width in cell layers will not match `nlayers`.
"""
function NarrowBandLevelSet(f, grid::CartesianGrid; nlayers::Int = 8, bc = nothing)
    return NarrowBandLevelSet(f, grid, nlayers * minimum(meshsize(grid)); bc)
end

"""
    _base_lookup(nb::NarrowBandLevelSet, I) -> value

Entry point for value lookup on a `NarrowBandLevelSet` at index `I`, which is
assumed to be inside the grid (out-of-grid indices are handled by `_getindexbc`
before reaching this function).

Tries the dict first; if `I` is not stored (i.e. it is inside the grid but
outside the narrow band), falls back to [`_extrapolate_nb_rec`](@ref) to
approximate the value from nearby band nodes. Throws an error if no path to
stored values can be found.
"""
function _base_lookup(nb::NarrowBandLevelSet{N}, I) where {N}
    val = get(values(nb), I, nothing)
    val !== nothing && return val
    val = _extrapolate_nb_rec(nb, I, N)
    val !== nothing && return val
    error("extrapolation failed at index $I: no resolvable path to stored values")
end

"""
    _extrapolate_nb_rec(nb::NarrowBandLevelSet, I, max_dim) -> value or nothing

Approximate the value at an in-grid index `I` that is not stored in the band
dict, by bilinear/trilinear (degree-1) Lagrange extrapolation from nearby band values.

The algorithm processes dimensions `1` through `max_dim` in order. For each
dimension, it searches outward from `I` (nearest first, both sides) for an
anchor point in the dict and uses the two consecutive nodes starting there
(anchor and anchor+step) as a linear stencil.

Stencil values are resolved by calling `_extrapolate_nb_rec` recursively with
`max_dim = dim - 1`, so each stencil point can itself be extrapolated using
lower dimensions. This produces a tensor-product extrapolation that handles
indices outside the band in multiple dimensions simultaneously.

Returns `nothing` if no dimension yields a valid stencil.

# Why degree 1 and no higher?
Outside the band, `|ϕ| ≥ halfwidth`. The only property we need from extrapolated
ghost values is sign correctness (no spurious zeros). Linear extrapolation from a
well-conditioned SDF preserves sign as long as `halfwidth > Δx`, which any
reasonable band satisfies. Quadratic or higher extrapolation introduces polynomial
oscillations that can flip the sign, creating false interface crossings that corrupt
`NewtonSDF` and cause the band to migrate or collapse.
"""
function _extrapolate_nb_rec(nb::NarrowBandLevelSet{N, T}, I::CartesianIndex{N}, max_dim) where {N, T}
    haskey(values(nb), I) && return values(nb)[I]
    grid_axes = axes(nb)
    # Degree-1 (linear) stencil — see docstring for why we never go higher.
    P = 1
    for dim in 1:max_dim
        for k in 1:length(grid_axes[dim])
            for side in (-1, 1)
                anchor = I[dim] + side * k
                anchor in grid_axes[dim] || continue
                val = _lagrange_extrap_from(nb, I, dim, anchor, side, k, P)
                val !== nothing && return val
            end
        end
    end
    return nothing
end

"""
    _lagrange_extrap_from(nb, I, dim, anchor, side, k, P) -> value or nothing

Attempt to evaluate a degree-`P` Lagrange extrapolant at `I[dim]` using `P+1`
consecutive stencil nodes along dimension `dim`:

    anchor, anchor + side, anchor + 2*side, …, anchor + P*side

The stencil extends from `anchor` deeper into the band (away from `I`). The
target `I[dim]` is at distance `k` from `anchor` in the opposite direction,
corresponding to local coordinate `ξ = -k` relative to stencil nodes at
`ξ = 0, 1, …, P`. This matches the convention of `_lagrange_extrap_weight(j, k, P)`.

Each stencil value is resolved via `_extrapolate_nb_rec(nb, Ij, dim - 1)`,
using only dimensions lower than `dim`. Returns `nothing` if any stencil point
falls outside the grid or cannot be resolved.
"""
function _lagrange_extrap_from(nb::NarrowBandLevelSet{N, T}, I, dim, anchor, side, k, P) where {N, T}
    grid_axes = axes(nb)
    result = zero(float(T))
    for j in 0:P
        pos = anchor + side * j
        pos in grid_axes[dim] || return nothing
        Ij = CartesianIndex(ntuple(s -> s == dim ? pos : I[s], Val(N)))
        Vj = _extrapolate_nb_rec(nb, Ij, dim - 1)
        Vj === nothing && return nothing
        result += _lagrange_extrap_weight(j, k, P) * Vj
    end
    return result
end

function active_cells(nb::NarrowBandLevelSet)
    grid = mesh(nb)
    cell_axes = cellindices(grid)
    active_nodes = active_indices(nb)
    active_cells = Set{CartesianIndex}()

    for I in cell_axes
        N = ndims(grid)
        all_corners_active = true
        for offset in Iterators.product(ntuple(_ -> 0:1, Val(N))...)
            corner_idx = I + CartesianIndex(offset)
            if !(corner_idx in active_nodes)
                all_corners_active = false
                break
            end
        end
        all_corners_active && push!(active_cells, I)
    end
    return active_cells
end

reinitialize!(nb::NarrowBandLevelSet, ::Nothing, _) = nb

function reinitialize!(nb::NarrowBandLevelSet, r::NewtonReinitializer)
    sdf = NewtonSDF(nb; order = r.order, upsample = r.upsample, maxiters = r.maxiters, xtol = r.xtol, ftol = r.ftol)
    rebuild_band!(nb, sdf)
    return nb
end

function reinitialize!(nb::NarrowBandLevelSet, r::NewtonReinitializer, nsteps::Int)
    mod(nsteps, r.reinit_freq) == 0 || return nb
    return reinitialize!(nb, r)
end

"""
    rebuild_band!(nb::NarrowBandLevelSet, sdf)

Rebuild the active node set from the signed distance function `sdf` using a breadth-first
search seeded from all previously active nodes. The BFS expands axis-aligned neighbors and
adds a node whenever `|sdf(x)| < halfwidth`, stopping a branch when a node falls outside the
band.
"""
function rebuild_band!(nb::NarrowBandLevelSet{N, T}, sdf) where {N, T}
    grid = mesh(nb)
    γ = halfwidth(nb)
    grid_axes = axes(nb)
    vals = values(nb)

    # Use a vector queue for BFS, keeping track of the head explicitly. Duplicate indices in
    # a set to have O(1) membership check.
    queue = collect(keys(vals))
    queue_set = Set{CartesianIndex{N}}(queue)
    empty!(vals)

    head = 1
    while head <= length(queue)
        I = queue[head]
        head += 1
        v = sdf(grid[I])
        abs(v) >= γ && continue   # outside band — don't expand further
        vals[I] = v
        for d in 1:N, s in (-1, 1)
            J = _increment_index(I, d, s)
            J ∈ queue_set && continue
            all(d -> J[d] in grid_axes[d], 1:N) || continue
            push!(queue_set, J)
            push!(queue, J)
        end
    end
    return nb
end

"""
    _clear_buffer!(ϕ::MeshField)

Clear the active entries of a buffer before it is used as a write target in a
time-integration step. For a [`NarrowBandLevelSet`](@ref) this empties the
values dict so that stale entries from a previous band do not survive the
src/dst swap. For a full-domain field this is a no-op.
"""
_clear_buffer!(::MeshField) = nothing
_clear_buffer!(nb::NarrowBandLevelSet) = empty!(values(nb))
