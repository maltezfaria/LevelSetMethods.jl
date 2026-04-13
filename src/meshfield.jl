"""
    abstract type AbstractMeshField

Abstract type for node-centered fields on a mesh. Concrete subtypes:
- [`MeshField`](@ref): full-domain field backed by a dense `Array`.
- [`NarrowBandMeshField`](@ref): narrow-band field backed by a sparse `Dict`.
"""
abstract type AbstractMeshField end

"""
    struct MeshField{V,M,B,I}

A node-centered field defined on the entire mesh, described by its discrete values
at each node.

- `vals`: dense array of values at each node.
- `mesh`: the underlying mesh (e.g. [`CartesianGrid`](@ref)).
- `bcs`: boundary conditions, used for indexing outside the mesh bounds.
- `itp_data`: optional `InterpolationData` for piecewise polynomial interpolation
  (`nothing` if the field was constructed without an interpolation order).

`Base.getindex(ϕ, I)` returns the value of the field at node index `I`. If `I` lies
outside the mesh bounds, `bcs` are applied to determine the value.

Use [`nodeindices`](@ref) and [`cellindices`](@ref) to iterate over active node and cell
indices respectively.
"""
struct MeshField{V, M, B, I} <: AbstractMeshField
    vals::V
    mesh::M
    bcs::B
    itp_data::I
end

# getters (on AbstractMeshField where applicable)
mesh(ϕ::AbstractMeshField) = ϕ.mesh
Base.values(ϕ::AbstractMeshField) = ϕ.vals
has_boundary_conditions(ϕ::AbstractMeshField) = !isnothing(ϕ.bcs)
boundary_conditions(ϕ::AbstractMeshField) = ϕ.bcs
interp_data(ϕ::AbstractMeshField) = ϕ.itp_data
has_interpolation(ϕ::AbstractMeshField) = !isnothing(ϕ.itp_data)
Base.valtype(ϕ::AbstractMeshField) = valtype(values(ϕ))

function check_real_valued(mf::AbstractMeshField)
    valtype(mf) <: Real || throw(ArgumentError("Expected a real-valued MeshField, but got valtype $(valtype(mf))"))
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
    _add_boundary_conditions(ϕ::MeshField, bc)

Return a new `MeshField` with `bc` as boundary conditions.  All of the underlying data is
aliased (shared) with the original `MeshField`. The `itp_data` is copied with its cache
invalidated so the new field owns independent interpolation state.
"""
function _add_boundary_conditions(ϕ::MeshField, bc)
    N = ndims(ϕ)
    itp = interp_data(ϕ)
    new_itp = isnothing(itp) ? nothing : copy(itp)
    return MeshField(values(ϕ), mesh(ϕ), _normalize_bc(bc, N), new_itp)
end

"""
    update_bcs!(ϕ::AbstractMeshField, t)

Update the current time in all [`DirichletBC`](@ref) boundary conditions of `ϕ`.
Called automatically by the time-stepper at each stage.
"""
function update_bcs!(ϕ::AbstractMeshField, t)
    has_boundary_conditions(ϕ) || (return ϕ)
    for bc_pair in boundary_conditions(ϕ)
        update_bc!(bc_pair[1], t)
        update_bc!(bc_pair[2], t)
    end
    return ϕ
end

"""
    MeshField(vals, grid::AbstractMesh; bc=nothing, interp_order=nothing)

Construct a node-centered `MeshField` with explicit values on a `grid`.

- `bc`: boundary conditions (normalized via [`_normalize_bc`](@ref)).
- `interp_order`: optional polynomial order for piecewise interpolation.

If `interp_order` is provided but `bc` is `nothing`, `bc` defaults to
`ExtrapolationBC{2}` to ensure valid stencils at the boundary.
"""
function MeshField(vals, grid::AbstractMesh; bc = nothing, interp_order = nothing)
    N = ndims(grid)
    T = valtype(vals)
    bcs = isnothing(bc) ? nothing : _normalize_bc(bc, N)

    # If interpolation is requested but no BCs are provided, default to 2nd-order
    # extrapolation to allow stencil access at boundaries.
    if !isnothing(interp_order) && isnothing(bcs)
        bcs = ntuple(_ -> (ExtrapolationBC{2}(), ExtrapolationBC{2}()), N)
    end

    itp = isnothing(interp_order) ? nothing : InterpolationData(N, interp_order, T)
    return MeshField(vals, grid, bcs, itp)
end

"""
    MeshField(f::Function, grid; kwargs...)

Create a node-centered `MeshField` by evaluating `f` at each node of `grid`.
All keyword arguments are passed to the `vals`-based constructor.

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
  ├─ eltype:  Float64
  └─ values:  min = -0.25,  max = 1.75
```
"""
function MeshField(f::Function, grid::AbstractMesh; kwargs...)
    vals = map(I -> f(getnode(grid, I)), nodeindices(grid))
    return MeshField(vals, grid; kwargs...)
end

# geometric dimension
Base.ndims(f::AbstractMeshField) = ndims(mesh(f))

# Base.length
Base.length(ϕ::AbstractMeshField) = length(eachindex(ϕ))

# overload base methods for convenience
function Base.getindex(ϕ::AbstractMeshField, I::CartesianIndex{N}) where {N}
    if has_boundary_conditions(ϕ)
        return _getindexbc(ϕ, I, N)
    else
        return _base_lookup(ϕ, I)
    end
end
function Base.getindex(ϕ::AbstractMeshField, I...)
    return ϕ[CartesianIndex(I...)]
end

_base_lookup(ϕ::MeshField, I) = getindex(values(ϕ), I)

function _getindexbc(ϕ::AbstractMeshField, I, dim)
    dim == 0 && return _base_lookup(ϕ, I)
    bcs = boundary_conditions(ϕ)[dim]
    ax = axes(ϕ)[dim]
    (I[dim] in axes(ϕ)[dim]) && (return _getindexbc(ϕ, I, dim - 1))
    bc = I[dim] < first(ax) ? bcs[1] : bcs[2]
    if bc isa PeriodicBC
        I′ = _wrap_index_periodic(I, ax, dim)
        return _getindexbc(ϕ, I′, dim - 1)
    elseif bc isa ExtrapolationBC
        return _apply_extrapolation_bc(ϕ, I, bc, ax, dim)
    elseif bc isa DirichletBC
        grid = mesh(ϕ)
        x = _getindex(grid, I)
        T = eltype(ϕ)
        return T(bc.f(x, bc.t))
    else
        error("Unknown boundary condition $bc")
    end
end

function _wrap_index_periodic(I::CartesianIndex{N}, ax, dim) where {N}
    i = I[dim]
    return ntuple(N) do d
        if d == dim
            if i < first(ax)
                return (last(ax) - (first(ax) - i))
            elseif i > last(ax)
                return (first(ax) + (i - last(ax)))
            end
        end
        return I[d]
    end |> CartesianIndex
end

"""
    _apply_extrapolation_bc(ϕ, I, ::ExtrapolationBC{P}, ax, dim)

Return the extrapolated value of `ϕ` at out-of-bounds index `I` in dimension
`dim` using degree-P Lagrange extrapolation (see [`_lagrange_extrap_weight`](@ref)).
The boundary node and its `P` interior neighbors are used as stencil nodes.
"""
function _apply_extrapolation_bc(ϕ, I::CartesianIndex{N}, ::ExtrapolationBC{P}, ax, dim) where {N, P}
    k = I[dim] < first(ax) ? (first(ax) - I[dim]) : (I[dim] - last(ax))
    b = I[dim] < first(ax) ? first(ax) : last(ax)
    # d = ±1 flips direction so both boundaries map to the same local coordinate
    d = I[dim] < first(ax) ? 1 : -1
    result = zero(float(eltype(ϕ)))
    for j in 0:P
        Ij = ntuple(s -> s == dim ? b + d * j : I[s], Val(N)) |> CartesianIndex
        Vj = _getindexbc(ϕ, Ij, dim - 1)
        result += _lagrange_extrap_weight(j, k, P) * Vj
    end
    return result
end


function Base.setindex!(ϕ::AbstractMeshField, val, I...)
    _invalidate_itp!(ϕ)
    setindex!(values(ϕ), val, I...)
    return ϕ
end

_invalidate_itp!(ϕ::MeshField{V, M, B, Nothing}) where {V, M, B} = nothing
function _invalidate_itp!(ϕ::MeshField{V, M, B, <:InterpolationData{N}}) where {V, M, B, N}
    ϕ.itp_data.Ic = CartesianIndex(ntuple(_ -> 0, Val(N)))
    return nothing
end
Base.eltype(ϕ::MeshField) = eltype(values(ϕ))

function Base.axes(ϕ::AbstractMeshField)
    sz = size(mesh(ϕ))
    return ntuple(d -> Base.OneTo(sz[d]), Val(ndims(ϕ)))
end

Base.eachindex(ϕ::MeshField) = nodeindices(mesh(ϕ))

"""
    nodeindices(ϕ::MeshField)

Return the active node indices of `ϕ`. For a `MeshField` this is `nodeindices(mesh(ϕ))`.
"""
nodeindices(ϕ::MeshField) = nodeindices(mesh(ϕ))

"""
    cellindices(ϕ::MeshField)

Return the active cell indices of `ϕ`. For a `MeshField` this is `cellindices(mesh(ϕ))`.
"""
cellindices(ϕ::MeshField) = cellindices(mesh(ϕ))

"""
    Base.copy!(dest::AbstractMeshField, src::AbstractMeshField)

Copy the values from `src` to `dest`. The meshes, boundary conditions, and domains of the
`dest` fields are not modified. The interpolation cache of `dest` is invalidated.
"""
function Base.copy!(dest::MeshField, src::MeshField)
    _invalidate_itp!(dest)
    copy!(values(dest), values(src))
    return dest
end

function _show_fields(io::IO, ϕ::MeshField{<:Any, <:CartesianGrid}; prefix = "  ")
    _show_fields(io, mesh(ϕ); prefix, last = false)
    if has_boundary_conditions(ϕ)
        println(io, "$(prefix)├─ bc:     $(_bc_str(boundary_conditions(ϕ)))")
    end
    return if eltype(ϕ) <: Real
        v = values(ϕ)
        vmin, vmax = extrema(v)
        println(io, "$(prefix)├─ eltype:  $(eltype(ϕ))")
        print(io, "$(prefix)└─ values:  min = $(round(vmin; sigdigits = 4)),  max = $(round(vmax; sigdigits = 4))")
    else
        print(io, "$(prefix)└─ eltype:  $(eltype(ϕ))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", ϕ::MeshField{<:Any, <:CartesianGrid})
    println(io, "MeshField on CartesianGrid in ℝ$(_superscript(ndims(ϕ)))")
    return _show_fields(io, ϕ)
end

# ---- NarrowBandMeshField --------------------------------------------------------
# Struct and basic methods here; constructors that call NewtonReinitializer are in narrowband.jl.

"""
    struct NarrowBandMeshField{V,M,B,T,I}

A node-centered field defined on a narrow band around the interface, described by its
discrete values at each *active* (in-band) node.

- `vals`: sparse `Dict{CartesianIndex{N},T}` mapping active node indices to values.
- `mesh`: the underlying mesh (e.g. [`CartesianGrid`](@ref)).
- `bcs`: boundary conditions, used for indexing outside the mesh bounds.
- `halfwidth`: half-width of the narrow band in physical units (typically a few grid spacings).
- `itp_data`: optional `InterpolationData` for piecewise polynomial interpolation.

Use [`nodeindices`](@ref) and [`cellindices`](@ref) to iterate over active node and cell
indices respectively.
"""
struct NarrowBandMeshField{V, M, B, T, I} <: AbstractMeshField
    vals::V
    mesh::M
    bcs::B
    halfwidth::T
    itp_data::I
end

halfwidth(nb::NarrowBandMeshField) = nb.halfwidth
Base.eltype(nb::NarrowBandMeshField) = valtype(nb)

"""
    nodeindices(nb::NarrowBandMeshField)

Return the set of active (in-band) node indices of `nb`.
"""
nodeindices(nb::NarrowBandMeshField) = keys(values(nb))

"""
    cellindices(nb::NarrowBandMeshField)

Return the set of active cell indices of `nb`: cells whose corners are all in-band nodes.
"""
function cellindices(nb::NarrowBandMeshField{<:Any, <:AbstractMesh{N}}) where {N}
    grid = mesh(nb)
    cell_axes = cellindices(grid)
    active_nodes = nodeindices(nb)
    cells = Set{CartesianIndex{N}}()
    checked = Set{CartesianIndex{N}}()
    offsets = Iterators.product(ntuple(_ -> 0:1, Val(N))...)
    for I in active_nodes
        for offset in offsets
            J = I - CartesianIndex(offset)
            J in cell_axes || continue
            J in checked && continue
            push!(checked, J)
            if all(J + CartesianIndex(off) in active_nodes for off in offsets)
                push!(cells, J)
            end
        end
    end
    return cells
end

Base.eachindex(nb::NarrowBandMeshField) = nodeindices(nb)

_invalidate_itp!(nb::NarrowBandMeshField{V, M, B, T, Nothing}) where {V, M, B, T} = nothing
function _invalidate_itp!(nb::NarrowBandMeshField{V, M, B, T, <:InterpolationData{N}}) where {V, M, B, T, N}
    nb.itp_data.Ic = CartesianIndex(ntuple(_ -> 0, Val(N)))
    return nothing
end

function Base.copy!(dest::NarrowBandMeshField, src::NarrowBandMeshField)
    _invalidate_itp!(dest)
    empty!(values(dest))
    merge!(values(dest), values(src))
    return dest
end

"""
    _add_boundary_conditions(nb::NarrowBandMeshField, bc)

Return a new `NarrowBandMeshField` with `bc` as boundary conditions, preserving
`halfwidth`. All underlying data is aliased with the original.
"""
function _add_boundary_conditions(nb::NarrowBandMeshField, bc)
    N = ndims(nb)
    itp = interp_data(nb)
    new_itp = isnothing(itp) ? nothing : copy(itp)
    return NarrowBandMeshField(values(nb), mesh(nb), _normalize_bc(bc, N), nb.halfwidth, new_itp)
end

"""
    _base_lookup(nb::NarrowBandMeshField, I) -> value

Lookup the value at in-grid index `I`. Tries the dict first; if not stored, falls back
to linear extrapolation from nearby band nodes.
"""
function _base_lookup(nb::NarrowBandMeshField{<:Any, <:AbstractMesh{N}}, I) where {N}
    val = get(values(nb), I, nothing)
    val !== nothing && return val
    val = _extrapolate_nb_rec(nb, I, N)
    val !== nothing && return val
    error("extrapolation failed at index $I: no resolvable path to stored values")
end

function _extrapolate_nb_rec(nb::NarrowBandMeshField{<:Any, <:AbstractMesh{N}}, I::CartesianIndex{N}, max_dim) where {N}
    haskey(values(nb), I) && return values(nb)[I]
    grid_axes = axes(nb)
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

function _lagrange_extrap_from(nb::NarrowBandMeshField{<:Any, <:AbstractMesh{N}}, I, dim, anchor, side, k, P) where {N}
    T = eltype(nb)
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

"""
    _clear_buffer!(nb::NarrowBandMeshField)

Empty the active-node dict before reuse as a write target in time integration.
"""
_clear_buffer!(::MeshField) = nothing
_clear_buffer!(nb::NarrowBandMeshField) = empty!(values(nb))

"""
    NarrowBandMeshField(vals, grid; halfwidth, bc=nothing, interp_order=nothing)

Construct a `NarrowBandMeshField` from a pre-built `vals` dict. `halfwidth` is required.
"""
function NarrowBandMeshField(vals, grid::AbstractMesh; halfwidth, bc = nothing, interp_order = nothing)
    N = ndims(grid)
    T = valtype(vals)
    bcs = isnothing(bc) ? nothing : _normalize_bc(bc, N)
    if !isnothing(interp_order) && isnothing(bcs)
        bcs = ntuple(_ -> (ExtrapolationBC{2}(), ExtrapolationBC{2}()), N)
    end
    itp = isnothing(interp_order) ? nothing : InterpolationData(N, interp_order, T)
    return NarrowBandMeshField(vals, grid, bcs, halfwidth, itp)
end

"""
    NarrowBandMeshField(f, grid, halfwidth; bc=nothing, interp_order=nothing)

Construct a narrow-band field by evaluating `f` at each node of `grid` and keeping
only those where `|f(x)| < halfwidth`. No dense array is allocated.

Pass `interp_order=k` to enable piecewise polynomial interpolation (same semantics
as [`MeshField`](@ref)).

!!! warning
    Since the `halfwidth` threshold is applied to the raw values of `f`, the resulting band
    width in physical space will only match `halfwidth` if `f` is already a signed distance
    function.
"""
function NarrowBandMeshField(f::Function, grid::AbstractMesh, halfwidth::Real; bc = nothing, interp_order = nothing)
    T = float(eltype(grid.lc))
    γ = T(halfwidth)
    vals = _nb_dict(I -> f(getnode(grid, I)), grid, γ)
    return NarrowBandMeshField(vals, grid; bc = bc, halfwidth = γ, interp_order = interp_order)
end

function _show_fields(io::IO, nb::NarrowBandMeshField{<:Any, <:CartesianGrid}; prefix = "  ")
    _show_fields(io, mesh(nb); prefix, last = false)
    if has_boundary_conditions(nb)
        println(io, "$(prefix)├─ bc:     $(_bc_str(boundary_conditions(nb)))")
    end
    hw = halfwidth(nb)
    nlayers = round(Int, hw / minimum(meshsize(mesh(nb))))
    println(io, "$(prefix)├─ active:  $(length(nodeindices(nb))) nodes ($nlayers layers, halfwidth = $(round(hw; sigdigits = 4)))")
    return if eltype(nb) <: Real
        vals_iter = Base.values(values(nb))
        vmin, vmax = extrema(vals_iter)
        println(io, "$(prefix)├─ eltype:  $(eltype(nb))")
        print(io, "$(prefix)└─ values:  min = $(round(vmin; sigdigits = 4)),  max = $(round(vmax; sigdigits = 4))")
    else
        print(io, "$(prefix)└─ eltype:  $(eltype(nb))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", nb::NarrowBandMeshField{<:Any, <:CartesianGrid})
    println(io, "NarrowBandMeshField on CartesianGrid in ℝ$(_superscript(ndims(nb)))")
    return _show_fields(io, nb)
end
