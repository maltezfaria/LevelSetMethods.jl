"""
    struct MeshField{V,M,B}

A field described by its discrete values on a mesh.

- `vals`: the discrete values of the field (typically an `AbstractArray`).
- `mesh`: the underlying mesh (e.g. [`CartesianGrid`](@ref)).
- `bcs`: boundary conditions, used for indexing outside the mesh bounds.

`Base.getindex` of an `MeshField` is overloaded to handle indices that lie outside the
`CartesianIndices` of its `MeshField` by using `bcs`.
"""
struct MeshField{V, M, B}
    vals::V
    mesh::M
    bcs::B
end

# getters
mesh(ϕ::MeshField) = ϕ.mesh
Base.values(ϕ::MeshField) = ϕ.vals
has_boundary_conditions(ϕ::MeshField) = !isnothing(ϕ.bcs)
boundary_conditions(ϕ::MeshField) = ϕ.bcs

meshsize(ϕ::MeshField, args...) = meshsize(mesh(ϕ), args...)

"""
    add_boundary_conditions(ϕ::MeshField, bcs)

Return a new `MeshField` with the given boundary conditions `bcs`.
The underlying data `values(ϕ)` is aliased (shared) with the original field.
"""
add_boundary_conditions(ϕ::MeshField, bcs) = MeshField(values(ϕ), mesh(ϕ), bcs)

"""
    MeshField(f::Function, m)

Create a `MeshField` by evaluating a function `f` on a mesh `m`.
"""
function MeshField(f::Function, m)
    vals = map(f, m)
    return MeshField(vals, m, nothing)
end

# geometric dimension
dimension(f::MeshField) = dimension(mesh(f))

# overload base methods for convenience
function Base.getindex(ϕ::MeshField, I::CartesianIndex)
    return if has_boundary_conditions(ϕ)
        _getindex(ϕ, I)
    else
        getindex(values(ϕ), I)
    end
end
function Base.getindex(ϕ::MeshField, I...)
    return ϕ[CartesianIndex(I...)]
end

function _getindex(ϕ::MeshField, I::CartesianIndex{N}) where {N}
    return _getindexrec(ϕ, I, N)
end

# Ghost value at out-of-bounds index I (in dimension `dim`) by P-point Lagrange
# extrapolation. A local coordinate is set up so that both boundaries look the
# same: b is the boundary node, k ≥ 1 the distance to the ghost, and d = ±1
# steps into the interior. The P stencil nodes then sit at positions 0,1,…,P-1
# and the ghost at -k:
#
#   ghost    b   b+d  b+2d  …  b+(P-1)d
#     |      |    |    |         |
#    -k      0    1    2        P-1      ← local coordinate
#
# d = +1 for the left boundary, -1 for the right — the flip makes the right
# boundary look identical to the left, so the same weights apply to both.
# Node values are fetched via _getindexrec(dim-1), so BCs in other dimensions
# are applied automatically (handles corner ghost points correctly).
function _apply_extrapolation_bc(ϕ, I::CartesianIndex{N}, ::ExtrapolationBC{P}, ax, dim) where {N, P}
    k = I[dim] < first(ax) ? (first(ax) - I[dim]) : (I[dim] - last(ax))
    b = I[dim] < first(ax) ? first(ax) : last(ax)
    d = I[dim] < first(ax) ? 1 : -1   # direction into the interior
    result = zero(float(eltype(values(ϕ))))
    for j in 0:(P - 1)
        Ij = ntuple(s -> s == dim ? b + d * j : I[s], Val(N)) |> CartesianIndex
        Vj = _getindexrec(ϕ, Ij, dim - 1)
        result += _lagrange_extrap_weight(j, k, P) * Vj
    end
    return result
end

# Recursive getindex with boundary conditions, processing one dimension per
# call (dim = N down to 1). If I[dim] is in-bounds, recurse; if out-of-bounds,
# apply the BC for that dimension then recurse to dim-1. Base case dim=0 does
# the raw array lookup. Corner ghost points (out-of-bounds in multiple
# dimensions) are handled correctly because each dimension's BC is applied in
# turn.
function _getindexrec(ϕ, I, dim)
    dim == 0 && return getindex(values(ϕ), I)
    bcs = boundary_conditions(ϕ)[dim]
    ax = axes(ϕ)[dim]
    (I[dim] in axes(ϕ)[dim]) && (return _getindexrec(ϕ, I, dim - 1))
    bc = I[dim] < first(ax) ? bcs[1] : bcs[2]
    if bc isa PeriodicBC
        I′ = _wrap_index_periodic(I, ax, dim)
        return _getindexrec(ϕ, I′, dim - 1)
    elseif bc isa ExtrapolationBC
        return _apply_extrapolation_bc(ϕ, I, bc, ax, dim)
    elseif bc isa DirichletBC
        grid = mesh(ϕ)
        x = _getindex(grid, I)
        T = eltype(ϕ)
        return T(bc.f(x))
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


Base.setindex!(ϕ::MeshField, vals, I...) = setindex!(values(ϕ), vals, I...)

function _get_index(ϕ::MeshField, I::CartesianIndex)
    return axs = axes(ϕ)
end

Base.axes(ϕ::MeshField) = axes(values(ϕ))
Base.eltype(ϕ::MeshField) = eltype(values(ϕ))
Base.eachindex(ϕ::MeshField) = eachindex(mesh(ϕ))

"""
    const CartesianMeshField{V,M<:CartesianGrid} = MeshField{V,M}

[`MeshField`](@ref) over a [`CartesianGrid`](@ref).
"""
const CartesianMeshField{V, M <: CartesianGrid, B} = MeshField{V, M, B}

# Boundary conditions

function _getindex(ϕ::CartesianMeshField, I::CartesianIndex{N}, ::PeriodicBC, d) where {N}
    ax = axes(ϕ)[abs(d)]
    # compute mirror index to I[d]
    i = I[abs(d)]
    J = ntuple(N) do dir
        if dir == abs(d)
            d < 0 ? (last(ax) - (first(ax) - i)) : (first(ax) + (i - last(ax)))
        else
            I[dir]
        end
    end
    return getindex(values(ϕ), CartesianIndex(J))
end

# TODO: test this
function _getindex(
        ϕ::CartesianMeshField,
        I::CartesianIndex{N},
        bc::DirichletBC,
        d,
    ) where {N}
    # Compute the closest index to I that is within the domain and return value of bc there
    Iproj = clamp.(Tuple(I), axes(ϕ)) |> CartesianIndex
    x = mesh(ϕ)(Iproj)
    return bc.f(x)
end
