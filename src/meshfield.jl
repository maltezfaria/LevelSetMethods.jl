"""
    abstract type AbstractDomain end

Abstract type for the domain of a [`MeshField`](@ref).
"""
abstract type AbstractDomain end

"""
    struct FullDomain <: AbstractDomain

Represents a field defined on the entire mesh.
"""
struct FullDomain <: AbstractDomain end


"""
    struct MeshField{V,M,B,D}

A field described by its discrete values on a mesh.

- `vals`: the discrete values of the field.
- `mesh`: the underlying mesh (e.g. [`CartesianGrid`](@ref)).
- `bcs`: boundary conditions, used for indexing outside the mesh bounds.
- `domain`: the domain on which the field is defined (e.g. [`FullDomain`](@ref)).

`Base.getindex` of an `MeshField` is overloaded to handle indices that lie outside the
`CartesianIndices` of its `MeshField` by using `bcs`.
"""
struct MeshField{V, M, B, D <: AbstractDomain}
    vals::V
    mesh::M
    bcs::B
    domain::D
end

# getters
mesh(ϕ::MeshField) = ϕ.mesh
Base.values(ϕ::MeshField) = ϕ.vals
domain(ϕ::MeshField) = ϕ.domain
has_boundary_conditions(ϕ::MeshField) = !isnothing(ϕ.bcs)
boundary_conditions(ϕ::MeshField) = ϕ.bcs

meshsize(ϕ::MeshField, args...) = meshsize(mesh(ϕ), args...)

"""
    add_boundary_conditions(ϕ::MeshField, bc)

Return a new `MeshField` with `bc` as boundary conditions.  All of the underlying data is
aliased (shared) with the original `MeshField`.
"""
function add_boundary_conditions(ϕ::MeshField, bc)
    N = ndims(ϕ)
    return MeshField(values(ϕ), mesh(ϕ), _normalize_bc(bc, N), domain(ϕ))
end

"""
    update_bcs!(ϕ::MeshField, t)

Update the current time in all [`DirichletBC`](@ref) boundary conditions of `ϕ`.
Called automatically by the time-stepper at each stage.
"""
function update_bcs!(ϕ::MeshField, t)
    has_boundary_conditions(ϕ) || (return ϕ)
    for bc_pair in boundary_conditions(ϕ)
        update_bc!(bc_pair[1], t)
        update_bc!(bc_pair[2], t)
    end
    return ϕ
end

"""
    MeshField(vals, mesh, bcs)

Construct a `MeshField` with explicit values, mesh, and boundary conditions.
Defaults to `FullDomain`.
"""
MeshField(vals, mesh, bcs) = MeshField(vals, mesh, bcs, FullDomain())

"""
    MeshField(f::Function, m)

Create a `MeshField` by evaluating a function `f` on a mesh `m`.

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

```jldoctest; output = true
using LevelSetMethods, StaticArrays
grid = CartesianGrid((-1, -1), (1, 1), (5, 5))
# vector-valued field
MeshField(x -> SVector(x[1], x[2]), grid)

# output

MeshField on CartesianGrid in ℝ²
  ├─ domain:  [-1.0, 1.0] × [-1.0, 1.0]
  ├─ nodes:   5 × 5
  ├─ spacing: h = (0.5, 0.5)
  └─ eltype:  SVector{2, Float64}
```
"""
function MeshField(f::Function, m, bc = nothing)
    bc_ = isnothing(bc) ? nothing : _normalize_bc(bc, ndims(m))
    vals = map(f, m)
    return MeshField(vals, m, bc_, FullDomain())
end

# geometric dimension
Base.ndims(f::MeshField) = ndims(mesh(f))

# Base.length
Base.length(ϕ::MeshField) = length(eachindex(ϕ))

# overload base methods for convenience
function Base.getindex(ϕ::MeshField, I::CartesianIndex{N}) where {N}
    if has_boundary_conditions(ϕ)
        return _getindexbc(ϕ, I, N)
    else
        return _base_lookup(ϕ, I)
    end
end
function Base.getindex(ϕ::MeshField, I...)
    return ϕ[CartesianIndex(I...)]
end

# Recursive getindex with boundary conditions, processing one dimension per
# call (dim = N down to 1). If I[dim] is in-bounds, recurse; if out-of-bounds,
# apply the BC for that dimension then recurse to dim-1. Base case dim=0 does
# the raw array lookup. Corner ghost points (out-of-bounds in multiple
# dimensions) are handled correctly because each dimension's BC is applied in
# turn.
_base_lookup(ϕ::MeshField, I) = getindex(values(ϕ), I)

function _getindexbc(ϕ::MeshField, I, dim)
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


Base.setindex!(ϕ::MeshField, vals, I...) = setindex!(values(ϕ), vals, I...)
Base.eltype(ϕ::MeshField) = eltype(values(ϕ))

function Base.axes(ϕ::MeshField)
    sz = size(mesh(ϕ))
    return ntuple(d -> Base.OneTo(sz[d]), Val(ndims(ϕ)))
end

Base.eachindex(ϕ::MeshField) = _eachindex(domain(ϕ), ϕ)
_eachindex(::FullDomain, ϕ) = eachindex(mesh(ϕ))

"""
    Base.copy!(dest::MeshField, src::MeshField)

Copy the values from `src` to `dest`. The meshes, boundary conditions, and domains of the
`dest` fields are not modified.
"""
function Base.copy!(dest::MeshField, src::MeshField)
    copy!(values(dest), values(src))
    return dest
end

"""
    _show_fields(io, ϕ::MeshField; prefix="  ")

Print the fields of `ϕ` as indented tree lines: grid info (via `_show_fields` for
`CartesianGrid`), boundary conditions, narrow-band info (if applicable), element type,
and value range (for real-valued fields).
"""
function _show_fields(io::IO, ϕ::MeshField{<:Any, <:CartesianGrid}; prefix = "  ")
    _show_fields(io, mesh(ϕ); prefix, last = false)
    if has_boundary_conditions(ϕ)
        println(io, "$(prefix)├─ bc:     $(_bc_str(boundary_conditions(ϕ)))")
    end
    dom = domain(ϕ)
    if dom isa NarrowBandDomain
        hw = dom.halfwidth
        nlayers = round(Int, hw / minimum(meshsize(mesh(ϕ))))
        println(io, "$(prefix)├─ active:  $(length(active_indices(ϕ))) nodes ($nlayers layers, halfwidth = $(round(hw; sigdigits = 4)))")
    end
    return if eltype(ϕ) <: Real
        v = values(ϕ)
        vals_iter = v isa AbstractDict ? Base.values(v) : v
        vmin, vmax = extrema(vals_iter)
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
