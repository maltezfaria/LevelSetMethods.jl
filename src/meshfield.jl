"""
    struct MeshField{V,M,B}

A field described by its discrete values on a mesh.

`Base.getindex` of an `MeshField` is overloaded to handle indices that lie outside the
`CartesianIndices` of its `MeshField` by using.
"""
struct MeshField{V,M,B}
    vals::V
    mesh::M
    bcs::B
end

# getters
mesh(ϕ::MeshField) = ϕ.mesh
Base.values(ϕ::MeshField) = ϕ.vals
has_boundary_conditions(ϕ::MeshField) = !isnothing(ϕ.bcs)
boundary_conditions(ϕ::MeshField) = ϕ.bcs

"""
    MeshField(f::Function, m)

Create a `MeshField` by evaluating a function `f` on a mesh `m`.
"""
function MeshField(f::Function, m)
    vals = map(f, m)
    return MeshField(vals, m)
end

# geometric dimension
dimension(f::MeshField) = dimension(mesh(f))

# overload base methods for convenience
function Base.getindex(ϕ::MeshField, I::CartesianIndex)
    if has_boundary_conditions(ϕ)
        I = wrap_index(ϕ, I)
    end
    return getindex(values(ϕ), I)
end
function Base.getindex(ϕ::MeshField, I...)
    return ϕ[CartesianIndex(I...)]
end

function Base.setindex!(ϕ::MeshField, vals, I::CartesianIndex)
    if has_boundary_conditions(ϕ)
        I = wrap_index(ϕ, I)
    end
    return setindex!(values(ϕ), vals, I)
end
Base.setindex!(ϕ::MeshField, vals, I...) = setindex!(ϕ, vals, CartesianIndex(I...))

function wrap_index(ϕ::MeshField, I::CartesianIndex{N}) where {N}
    bcs = boundary_conditions(ϕ)
    axs = axes(ϕ)
    It = ntuple(N) do d
        ax = axs[d]
        i = I[d]
        if i < first(ax)
            return wrap_index(ax, i, bcs[d][1]) # left
        elseif i > last(ax)
            return wrap_index(ax, i, bcs[d][2]) # right
        else
            return i
        end
    end
    return CartesianIndex(It)
end
wrap_index(ϕ::MeshField, I...) = wrap_index(ϕ, CartesianIndex(I))

Base.axes(ϕ::MeshField) = axes(values(ϕ))
Base.eltype(ϕ::MeshField) = eltype(values(ϕ))
Base.zero(ϕ::MeshField) = MeshField(zero(values(ϕ)), mesh(ϕ), boundary_condition(ϕ))
Base.similar(ϕ::MeshField) = MeshField(similar(values(ϕ)), mesh(ϕ), boundary_condition(ϕ))

"""
    LevelSet

Alias for [`MeshField`](@ref) with `vals` as an `AbstractArray` of `Real`s.
"""
const LevelSet{V<:AbstractArray{<:Real},M} = MeshField{V,M,Nothing}

function LevelSet(f::Function, m)
    vals = map(f, m)
    return MeshField(vals, m)
end

"""
    const CartesianMeshField{V,M<:CartesianGrid} = MeshField{V,M}

[`MeshField`](@ref) over a [`CartesianGrid`](@ref).
"""
const CartesianMeshField{V,M<:CartesianGrid} = MeshField{V,M}
