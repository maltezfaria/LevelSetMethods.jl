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

meshsize(ϕ::MeshField, args...) = meshsize(mesh(ϕ), args...)

add_boundary_conditions(ϕ::MeshField, bcs) = MeshField(values(ϕ), mesh(ϕ), bcs)
remove_boundary_conditions(ϕ::MeshField) = MeshField(values(ϕ), mesh(ϕ), nothing)

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
    if has_boundary_conditions(ϕ)
        _getindex(ϕ, I)
    else
        getindex(values(ϕ), I)
    end
end
function Base.getindex(ϕ::MeshField, I...)
    return ϕ[CartesianIndex(I...)]
end

function _getindex(ϕ::MeshField, I::CartesianIndex{N}) where {N}
    bcs = boundary_conditions(ϕ)
    axs = axes(ϕ)
    # identify the first dimension where the index is out of bounds and use the
    # corresponding boundary condition
    # FIXME: the code would probably fail if the index is out of bounds in more than one
    # dimension, but is this a valid use case?
    for d in 1:N
        ax = axs[d]
        i = I[d]
        if i < first(ax)
            return _getindex(ϕ, I, bcs[d][1], -d) # left
        elseif i > last(ax)
            return _getindex(ϕ, I, bcs[d][2], d) # right
        end
    end
    return getindex(values(ϕ), I)
end

Base.setindex!(ϕ::MeshField, vals, I...) = setindex!(values(ϕ), vals, I...)

function _get_index(ϕ::MeshField, I::CartesianIndex)
    return axs = axes(ϕ)
end

Base.axes(ϕ::MeshField) = axes(values(ϕ))
Base.eltype(ϕ::MeshField) = eltype(values(ϕ))
Base.eachindex(ϕ::MeshField) = eachindex(mesh(ϕ))

"""
    LevelSet

Alias for [`MeshField`](@ref) with `vals` as an `AbstractArray` of `Real`s.
"""
const LevelSet{V<:AbstractArray{<:Real},M,B} = MeshField{V,M,B}

function LevelSet(f::Function, m)
    vals = map(f, m)
    return MeshField(vals, m, nothing)
end

"""
    const CartesianMeshField{V,M<:CartesianGrid} = MeshField{V,M}

[`MeshField`](@ref) over a [`CartesianGrid`](@ref).
"""
const CartesianMeshField{V,M<:CartesianGrid} = MeshField{V,M}

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

function _getindex(ϕ::CartesianMeshField, I::CartesianIndex{N}, ::NeumannBC, d) where {N}
    ax = axes(ϕ)[abs(d)]
    # compute mirror index to I[d]
    i = I[abs(d)]
    J = ntuple(N) do dir
        if dir == abs(d)
            d < 0 ? (first(ax) + (first(ax) - i)) : (last(ax) - (i - last(ax)))
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
