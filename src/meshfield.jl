"""
    abstract type AbstractMeshField{N,T,V}

A mutable, node-centered discrete field on a [`CartesianGrid`](@ref): it attaches a value to
every node of the grid and can be both read and written by Cartesian multi-index.

The type parameters expose the field's basic shape: `N` is the spatial dimension, `T` the
grid's coordinate type, and `V` the type of the stored values (e.g. `Float64` for a level set,
`SVector{N,Float64}` for a velocity field).

# Interface

A concrete subtype must implement:

- [`mesh(ϕ)`](@ref): the underlying [`CartesianGrid`](@ref); every geometric quantity
  (`ndims`, `axes`, [`meshsize`](@ref), …) derives from it.
- `ϕ[I]` (`Base.getindex`): the value at node `I`. This must also return a value for indices
  just outside the grid, so that finite-difference stencils work near the boundary.
- `ϕ[I] = v` (`Base.setindex!`): write `v` at node `I`.

The following optional methods can be implemented for efficiency:

- [`active_nodeindices(ϕ)`](@ref): node indices the field is actually defined on; defaults to
  [`nodeindices(ϕ)`](@ref) (every grid node).
- [`active_cellindices(ϕ)`](@ref): likewise for cells; defaults to [`cellindices(ϕ)`](@ref).
- `copy(ϕ)`: clone the field; defaults to `deepcopy`. The time integrators allocate stage
  buffers with it, so overriding it to share the immutable mesh/bcs is cheaper.
- `copy!(dest, src)`: overwrite `dest`'s values (and active set) with `src`'s; a generic
  fallback copies through `values`. The time integrators recycle stage buffers with it.

See [`MeshField`](@ref) (dense) and [`NarrowBandMeshField`](@ref) (sparse band) for examples.
"""
abstract type AbstractMeshField{N, T, V} end

"""
    struct MeshField{N,T,V,B} <: AbstractMeshField{N,T,V}

A node-centered field defined on the entire mesh, described by its discrete values
at each node.

- `vals`: dense `Array{V,N}` of values at each node.
- `mesh`: the underlying [`CartesianGrid{N,T}`](@ref).
- `bcs`: boundary conditions, used for indexing outside the mesh bounds.

`Base.getindex(ϕ, I)` returns the value of the field at node index `I`. If `I` lies
outside the mesh bounds, `bcs` are applied to determine the value.

Use [`nodeindices`](@ref) and [`cellindices`](@ref) to iterate over active node and cell
indices respectively.
"""
struct MeshField{N, T, V, B} <: AbstractMeshField{N, T, V}
    vals::Array{V, N}
    mesh::CartesianGrid{N, T}
    bcs::B
end

# getters (on AbstractMeshField where applicable)
mesh(ϕ::AbstractMeshField) = ϕ.mesh
Base.values(ϕ::AbstractMeshField) = ϕ.vals
has_boundary_conditions(ϕ::AbstractMeshField) = !isnothing(ϕ.bcs)
boundary_conditions(ϕ::AbstractMeshField) = ϕ.bcs
Base.ndims(::AbstractMeshField{N}) where {N} = N
Base.valtype(::AbstractMeshField{N, T, V}) where {N, T, V} = V

"""
    isrealvalued(ϕ::AbstractMeshField)

Return `true` if the values of `ϕ` are real numbers. Real-valued fields are the ones that may
be interpreted as a level set (with their zero contour representing an interface).
"""
isrealvalued(ϕ::AbstractMeshField) = valtype(ϕ) <: Real

function check_real_valued(mf::AbstractMeshField)
    isrealvalued(mf) || throw(ArgumentError("Expected a real-valued MeshField, but got valtype $(valtype(mf))"))
    return nothing
end

function _check_bc(ϕ::AbstractMeshField)
    has_boundary_conditions(ϕ) || throw(
        ArgumentError(
            "this operation requires boundary conditions. Pass `bc=...` when constructing the MeshField."
        )
    )
    return nothing
end

meshsize(ϕ::AbstractMeshField, args...) = meshsize(mesh(ϕ), args...)

"""
    getnode(ϕ::AbstractMeshField, I)

Return the coordinates of node `I`, delegating to `getnode` on the underlying mesh.
"""
getnode(ϕ::AbstractMeshField, I) = getnode(mesh(ϕ), I)

"""
    getcell(ϕ::AbstractMeshField, I)

Return the cell at index `I`, delegating to `getcell` on the underlying mesh.
"""
getcell(ϕ::AbstractMeshField, I) = getcell(mesh(ϕ), I)

"""
    nodeindices(ϕ::AbstractMeshField)

Return all node indices of the underlying mesh. See [`active_nodeindices`](@ref) for the
subset a field is actually defined on.
"""
nodeindices(ϕ::AbstractMeshField) = nodeindices(mesh(ϕ))

"""
    cellindices(ϕ::AbstractMeshField)

Return all cell indices of the underlying mesh. See [`active_cellindices`](@ref) for the
subset a field is actually defined on.
"""
cellindices(ϕ::AbstractMeshField) = cellindices(mesh(ϕ))

"""
    compute_index(ϕ::AbstractMeshField, x)

Return the multi-index of the cell containing the point `x`, delegating to
[`compute_index(g::CartesianGrid, x)`](@ref) on the underlying mesh.
"""
compute_index(ϕ::AbstractMeshField, x) = compute_index(mesh(ϕ), x)

"""
    active_nodeindices(ϕ::AbstractMeshField)

Return the node indices `ϕ` is defined on. Defaults to [`nodeindices`](@ref) (every grid
node); sparse fields such as [`NarrowBandMeshField`](@ref) override it with their active
set.
"""
active_nodeindices(ϕ::AbstractMeshField) = nodeindices(ϕ)

"""
    active_cellindices(ϕ::AbstractMeshField)

Return the cell indices `ϕ` is defined on. Defaults to [`cellindices`](@ref) (every grid
cell); sparse fields override it with their active set.
"""
active_cellindices(ϕ::AbstractMeshField) = cellindices(ϕ)

"""
    _add_boundary_conditions(ϕ::MeshField, bc)

Return a new `MeshField` with `bc` as boundary conditions.  All of the underlying data is
aliased (shared) with the original `MeshField`.
"""
function _add_boundary_conditions(ϕ::MeshField, bc)
    N = ndims(ϕ)
    return MeshField(values(ϕ), mesh(ϕ), _normalize_bc(bc, N))
end

"""
    map(f, ϕ::MeshField)

Return a `MeshField` on the same mesh and boundary conditions as `ϕ`, with each value replaced
by `f` applied to it. The element type may change with `f`.
"""
Base.map(f, ϕ::MeshField) = MeshField(map(f, values(ϕ)), mesh(ϕ), boundary_conditions(ϕ))

"""
    copy(ϕ::MeshField)

Return a `MeshField` with an independent copy of the values of `ϕ`, sharing its (immutable)
mesh and boundary conditions.
"""
Base.copy(ϕ::MeshField) = MeshField(copy(values(ϕ)), mesh(ϕ), boundary_conditions(ϕ))

"""
    MeshField(vals, grid::CartesianGrid; bc=nothing)

Construct a node-centered `MeshField` with explicit values on a `grid`.

- `bc`: boundary conditions (normalized via [`_normalize_bc`](@ref)).
"""
function MeshField(vals, grid::CartesianGrid; bc = nothing)
    N = ndims(grid)
    bcs = isnothing(bc) ? nothing : _normalize_bc(bc, N)
    return MeshField(vals, grid, bcs)
end

"""
    MeshField(f::Function, grid; kwargs...)

Create a node-centered `MeshField` by evaluating `f` at each node of `grid`.  All keyword
arguments are passed to the `vals`-based constructor.

# Examples

```jldoctest; output = true
using LevelSetMethods, StaticArrays
grid = CartesianGrid((-1, -1), (1, 1), (5, 5))
# scalar field without boundary conditions
MeshField(x -> x[1]^2 + x[2]^2 - 0.5^2, grid)

# output

MeshField on CartesianGrid in ℝ²
  ├─ domain:  [-1.0, 1.0] × [-1.0, 1.0]
  ├─ nodes:   5 × 5
  ├─ spacing: h = (0.5, 0.5)
  ├─ valtype: Float64
  └─ values:  min = -0.25,  max = 1.75
```
"""
function MeshField(f::Function, grid::CartesianGrid; kwargs...)
    vals = map(I -> f(getnode(grid, I)), nodeindices(grid))
    return MeshField(vals, grid; kwargs...)
end

function Base.getindex(ϕ::MeshField, I::CartesianIndex{N}) where {N}
    I in CartesianIndices(values(ϕ)) && return values(ϕ)[I]
    has_boundary_conditions(ϕ) || _throw_out_of_grid(ϕ, I)
    return _getindexbc(ϕ, I, Val(N))
end
function Base.getindex(ϕ::AbstractMeshField, I...)
    return ϕ[CartesianIndex(I...)]
end

function _throw_out_of_grid(ϕ::AbstractMeshField, I)
    throw(
        ArgumentError(
            """
            index $I lies outside the grid, but the field has no boundary conditions to \
            resolve it. Attach boundary conditions (e.g. the `bc` keyword when constructing \
            the field) to index outside the domain.\
            """
        ),
    )
end

"""
    _getindexbc(ϕ::AbstractMeshField, I, Val(dim)) -> value

Recursive resolutions `ϕ` at an out-of-grid index `I` via boundary conditions. The recursion
peels one dimension at a time from `dim` down to `0`; each level that is in range is
skipped, and each out-of-range level is resolved by accumulating the boundary condition's
[`bc_stencil`](@ref), recursing on the remaining dimensions so corner ghosts (out of bounds
in several dimensions) compose correctly. At level `0` every component is back in range, so
the value is read through the field's own in-grid `getindex` (`ϕ[I]`).

`dim` is carried as a `Val` so that it is a compile-time constant: the per-dimension lookups
`axes(ϕ)[dim]` and `boundary_conditions(ϕ)[dim]` then resolve to concrete types, keeping the
boundary-condition dispatch static even when the conditions differ between dimensions.
"""
function _getindexbc(ϕ::AbstractMeshField, I, ::Val{dim}) where {dim}
    dim == 0 && return ϕ[I]
    ax = axes(ϕ)[dim]
    (I[dim] in ax) && return _getindexbc(ϕ, I, Val(dim - 1))
    bcs = boundary_conditions(ϕ)[dim]
    bc = I[dim] < first(ax) ? bcs[1] : bcs[2]
    T = float(valtype(ϕ))
    acc = zero(T)
    for (w, Iʲ) in bc_stencil(bc, I, ax, dim)
        acc += T(w) * _getindexbc(ϕ, Iʲ, Val(dim - 1))
    end
    return acc
end


function Base.setindex!(ϕ::AbstractMeshField, val, I...)
    setindex!(values(ϕ), val, I...)
    return ϕ
end

function Base.axes(ϕ::AbstractMeshField)
    sz = size(mesh(ϕ))
    return ntuple(d -> Base.OneTo(sz[d]), Val(ndims(ϕ)))
end

"""
    Base.copy(ϕ::AbstractMeshField)

Clone `ϕ`. Generic fallback (`deepcopy`); concrete fields override it to share their immutable
mesh and boundary conditions and only copy the values.
"""
Base.copy(ϕ::AbstractMeshField) = deepcopy(ϕ)

"""
    Base.copy!(dest::AbstractMeshField, src::AbstractMeshField)

Make `dest` hold the same values as `src`, leaving its mesh and boundary conditions untouched.
For a sparse [`NarrowBandMeshField`](@ref) this also replaces `dest`'s active set with `src`'s
(the `Dict` `copy!` syncs keys), so the two agree on which nodes are stored — the time
integrator relies on this to keep a reused stage buffer's band in sync with the evolving field.
"""
function Base.copy!(dest::AbstractMeshField, src::AbstractMeshField)
    copy!(values(dest), values(src))
    return dest
end

function _show_fields(io::IO, ϕ::MeshField; prefix = "  ")
    _show_fields(io, mesh(ϕ); prefix, last = false)
    if has_boundary_conditions(ϕ)
        println(io, "$(prefix)├─ bc:     $(_bc_str(boundary_conditions(ϕ)))")
    end
    return if valtype(ϕ) <: Real
        v = values(ϕ)
        vmin, vmax = extrema(v)
        println(io, "$(prefix)├─ valtype: $(valtype(ϕ))")
        print(io, "$(prefix)└─ values:  min = $(round(vmin; sigdigits = 4)),  max = $(round(vmax; sigdigits = 4))")
    else
        print(io, "$(prefix)└─ valtype: $(valtype(ϕ))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", ϕ::MeshField)
    println(io, "MeshField on CartesianGrid in ℝ$(_superscript(ndims(ϕ)))")
    return _show_fields(io, ϕ)
end

"""
    struct NarrowBandMeshField{N,T,V,B} <: AbstractMeshField{N,T,V}

A node-centered field defined on a *narrow band*: a sparse set of *active* nodes, with values
stored only there. The fields are:

- `vals`: sparse `Dict{CartesianIndex{N},V}` mapping active node indices to values.
- `mesh`: the underlying [`CartesianGrid{N,T}`](@ref).
- `bcs`: boundary conditions, used for indexing outside the mesh bounds.
- `nlayers`: halo depth of the band — the number of node layers grown around the cut cells by
  [`update_band!`](@ref), and the depth it rebuilds to during evolution.

Use [`active_nodeindices`](@ref) and [`active_cellindices`](@ref) to iterate over active
node and cell indices respectively.

Indexing is defined on the band and the surrounding stencil halo; an in-grid node farther
out throws. At present, periodic boundary conditions are not supported.
"""
struct NarrowBandMeshField{N, T, V, B} <: AbstractMeshField{N, T, V}
    vals::Dict{CartesianIndex{N}, V}
    mesh::CartesianGrid{N, T}
    bcs::B
    nlayers::Int
    function NarrowBandMeshField(vals::Dict{CartesianIndex{N}, V}, mesh::CartesianGrid{N, T}, bcs::B, nlayers::Int) where {N, T, V, B}
        # the band and its halo are not periodic-aware (they do not wrap at the grid edge)
        any(p -> any(b -> b isa PeriodicBC, p), something(bcs, ())) &&
            throw(ArgumentError("PeriodicBC is not supported on a NarrowBandMeshField"))
        return new{N, T, V, B}(vals, mesh, bcs, nlayers)
    end
end

"""
    nlayers(nb::NarrowBandMeshField)

Return the halo depth of the band (see [`NarrowBandMeshField`](@ref)).
"""
nlayers(nb::NarrowBandMeshField) = nb.nlayers

"""
    active_nodeindices(nb::NarrowBandMeshField)

Return the set of active (in-band) node indices of `nb`.
"""
active_nodeindices(nb::NarrowBandMeshField) = keys(values(nb))

"""
    active_cellindices(nb::NarrowBandMeshField)

Return the set of active cell indices of `nb`: cells whose corners are all in-band nodes.
"""
function active_cellindices(nb::NarrowBandMeshField{N}) where {N}
    active = active_nodeindices(nb)
    cell_axes = cellindices(mesh(nb))
    offsets = CartesianIndices(ntuple(_ -> 0:1, Val(N)))
    return Set(J for J in active if J in cell_axes && all(J + off in active for off in offsets))
end

"""
    _add_boundary_conditions(nb::NarrowBandMeshField, bc)

Return a new `NarrowBandMeshField` with `bc` as boundary conditions. All underlying data is
aliased with the original.
"""
function _add_boundary_conditions(nb::NarrowBandMeshField, bc)
    N = ndims(nb)
    return NarrowBandMeshField(values(nb), mesh(nb), _normalize_bc(bc, N), nb.nlayers)
end

"""
    NarrowBandMeshField(vals, grid; bc=nothing, nlayers=3)

Construct a `NarrowBandMeshField` from a pre-built `vals` dict whose keys are the active
nodes. Useful for an auxiliary field on an *adopted* band (e.g. a velocity), where the
active set is filled from another field rather than derived from `vals`.
"""
function NarrowBandMeshField(vals, grid::CartesianGrid; bc = nothing, nlayers::Int = 3)
    N = ndims(grid)
    bcs = isnothing(bc) ? nothing : _normalize_bc(bc, N)
    return NarrowBandMeshField(vals, grid, bcs, nlayers)
end

function _show_fields(io::IO, nb::NarrowBandMeshField; prefix = "  ")
    _show_fields(io, mesh(nb); prefix, last = false)
    if has_boundary_conditions(nb)
        println(io, "$(prefix)├─ bc:     $(_bc_str(boundary_conditions(nb)))")
    end
    println(io, "$(prefix)├─ active:  $(length(active_nodeindices(nb))) nodes ($(nb.nlayers)-layer halo)")
    return if valtype(nb) <: Real
        vals_iter = Base.values(values(nb))
        vmin, vmax = extrema(vals_iter)
        println(io, "$(prefix)├─ valtype: $(valtype(nb))")
        print(io, "$(prefix)└─ values:  min = $(round(vmin; sigdigits = 4)),  max = $(round(vmax; sigdigits = 4))")
    else
        print(io, "$(prefix)└─ valtype: $(valtype(nb))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", nb::NarrowBandMeshField)
    println(io, "NarrowBandMeshField on CartesianGrid in ℝ$(_superscript(ndims(nb)))")
    return _show_fields(io, nb)
end

"""
    NarrowBandMeshField(ϕ::MeshField; nlayers = 3)

Construct a topological narrow-band field from a full-grid [`MeshField`](@ref). The
narrow-band field is defined on the band and a halo of `nlayers` nodes around the cut-cells.
"""
function NarrowBandMeshField(ϕ::MeshField; nlayers::Int = 3)
    grid = mesh(ϕ)
    N = ndims(grid)
    T = float(valtype(ϕ))
    # seed every node with its value, then let `update_band!` restrict to the band. Not the
    # most efficient, but simple
    vals = Dict{CartesianIndex{N}, T}()
    for I in nodeindices(grid)
        vals[I] = T(ϕ[I])
    end
    nb = NarrowBandMeshField(vals, grid, boundary_conditions(ϕ), nlayers)
    update_band!(nb)
    return nb
end

function Base.getindex(nb::NarrowBandMeshField{N}, I::CartesianIndex{N}) where {N}
    I in CartesianIndices(axes(nb)) && return _ingrid_value(nb, I)
    has_boundary_conditions(nb) || _throw_out_of_grid(nb, I)
    return _getindexbc(nb, I, Val(N))
end

"""
    map(f, nb::NarrowBandMeshField)

Return a `NarrowBandMeshField` with the same mesh, boundary conditions and band as `nb`, with
each stored value replaced by `f` applied to it. The element type may change with `f`.
"""
function Base.map(f, nb::NarrowBandMeshField)
    vals = Dict(I => f(v) for (I, v) in values(nb))
    return NarrowBandMeshField(vals, mesh(nb), boundary_conditions(nb), nb.nlayers)
end

"""
    copy(nb::NarrowBandMeshField)

Return a `NarrowBandMeshField` with an independent copy of the band values of `nb`, sharing
its (immutable) mesh and boundary conditions.
"""
function Base.copy(nb::NarrowBandMeshField)
    return NarrowBandMeshField(copy(values(nb)), mesh(nb), boundary_conditions(nb), nb.nlayers)
end

"""
    _ingrid_value(nb::NarrowBandMeshField, I) -> value

Value at the in-grid index `I`. Returns the stored value if `I` is a band node; otherwise
extrapolates it from the nearest band node (see [`_extrapolate_to_ghost`](@ref)).
"""
function _ingrid_value(nb::NarrowBandMeshField{N}, I) where {N}
    val = get(values(nb), I, nothing)
    val !== nothing && return val
    return _extrapolate_to_ghost(nb, I)
end

"""
    _extrapolate_to_ghost(nb::NarrowBandMeshField, I) -> value

Extrapolate a value at the out-of-band index `I` so that finite-difference/WENO stencils
reaching just past the band edge get a consistent value. It is the linear (affine) Lagrange
extrapolant anchored at the nearest band node.
"""
function _extrapolate_to_ghost(nb::NarrowBandMeshField{N}, I) where {N}
    d = values(nb)
    T = float(valtype(nb))
    P = _nearest_band_node(d, I)
    # nothing within the halo to extrapolate from: signal rather than fabricate a value
    isnothing(P) &&
        throw(ArgumentError("index $I is more than $(_BAND_SEARCH_RADIUS) nodes from the band"))
    ϕP = T(d[P])
    # Affine extrapolant anchored at the nearest band node, with the gradient estimated per
    # axis from P's in-band neighbours (an axis-aligned simplex). Exact for affine fields.
    val = ϕP
    for dim in 1:N
        δ = I[dim] - P[dim]
        iszero(δ) || (val += δ * _axis_slope(d, P, dim, ϕP))
    end
    # Never let extrapolation invent a sign change (a spurious interface) far from the band.
    return (iszero(ϕP) || sign(val) == sign(ϕP)) ? val : ϕP
end

const _BAND_SEARCH_RADIUS = 6

@generated function _ring_offsets(::Val{N}) where {N}
    R = _BAND_SEARCH_RADIUS
    offsets = vec(collect(CartesianIndices(ntuple(_ -> (-R):R, N))))
    sort!(offsets; by = off -> sum(abs2, Tuple(off)))
    return :($offsets)
end

# The band node nearest `I` (Euclidean), or `nothing` if none lies within the search radius.
# Off-grid indices need no special casing: the band dict only holds in-grid keys, so they
# simply miss `haskey`. A stencil-reachable ghost always has a node within a few rings.
function _nearest_band_node(d, I::CartesianIndex{N}) where {N}
    for off in _ring_offsets(Val(N))
        haskey(d, I + off) && return I + off
    end
    return nothing
end

@inline function _axis_slope(d, P, dim, ϕP::T) where {T}
    vp = get(d, _increment_index(P, dim, 1), nothing)
    vp !== nothing && return T(vp) - ϕP
    vm = get(d, _increment_index(P, dim, -1), nothing)
    vm !== nothing && return ϕP - T(vm)
    return zero(T)
end

"""
    update_band!(ϕ::AbstractMeshField; nlayers)

Rebuild the active node set topologically: every node within `nlayers` axis-aligned steps of
a *cut* cell (one whose corner values straddle zero). `ϕ` must be **real-valued** (a level
set). Values are filled via `ϕ`'s own indexing (stored value or affine extrapolation).

`nlayers` defaults to the band's stored halo depth (see [`NarrowBandMeshField`](@ref)).

A full-grid [`MeshField`](@ref) already holds every node, so there is no band to rebuild and
this is a no-op — making it safe to call regardless of the state type. [`integrate!`](@ref)
calls it automatically after each step.
"""
update_band!(ϕ::AbstractMeshField; kwargs...) = ϕ

function update_band!(nb::NarrowBandMeshField{N}; nlayers::Int = nb.nlayers) where {N}
    check_real_valued(nb)
    grid_axes = axes(nb)
    old_vals = values(nb)
    T = valtype(nb)
    inbounds(J) = all(d -> J[d] in grid_axes[d], 1:N)
    corners = CartesianIndices(ntuple(_ -> 0:1, Val(N)))
    box = CartesianIndices(ntuple(_ -> (-nlayers):nlayers, Val(N)))
    # offsets from a cell's lower corner of every node within `nlayers` L¹-steps of a corner
    grow = unique(c + o for c in corners for o in box if sum(abs, Tuple(o)) <= nlayers)

    # cut cells (all 2ᴺ corners present, values straddling zero) stamp `grow` into the band
    new_keys = Set{CartesianIndex{N}}()
    for I in keys(old_vals)
        vmin = typemax(T)
        vmax = typemin(T)
        ok = all(corners) do c
            v = get(old_vals, I + c, nothing)
            v === nothing || !inbounds(I + c) ? false : (vmin = min(vmin, v); vmax = max(vmax, v); true)
        end
        (ok && vmin <= 0 <= vmax) || continue
        for off in grow
            inbounds(I + off) && push!(new_keys, I + off)
        end
    end

    # fill via getindex (stored value or affine extrapolation), then swap in place
    new_vals = empty(old_vals)
    for J in new_keys
        new_vals[J] = nb[J]
    end
    copy!(old_vals, new_vals)
    return nb
end
