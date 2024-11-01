"""
    struct MeshField{V,M,B}

A field described by its discrete values on a mesh.

`Base.getindex` of an `MeshField` is overloaded to handle indices that lie outside the
`CartesianIndices` of its `MeshField` by using `bcs`.
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
    return _getindexrec(ϕ, I, N)
end

function _getindexrec(ϕ, I, dim)
    dim == 0 && return getindex(values(ϕ), I)
    bcs = boundary_conditions(ϕ)[dim]
    ax = axes(ϕ)[dim]
    (I[dim] in axes(ϕ)[dim]) && (return _getindexrec(ϕ, I, dim - 1))
    bc = I[dim] < first(ax) ? bcs[1] : bcs[2]
    if bc isa PeriodicBC
        I′ = _wrap_index_periodic(I, ax, dim)
        return _getindexrec(ϕ, I′, dim - 1)
    elseif bc isa NeumannBC
        I′ = _wrap_index_neumann(I, ax, dim)
        return _getindexrec(ϕ, I′, dim - 1)
    elseif bc isa NeumannGradientBC
        Ion, Iin, dist = _wrap_index_neumann_gradient(I, ax, dim)
        Von = _getindexrec(ϕ, Ion, dim - 1)
        Vin = _getindexrec(ϕ, Iin, dim - 1)
        return Von + dist * (Von - Vin)
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
    ntuple(N) do d
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

function _wrap_index_neumann(I::CartesianIndex{N}, ax, dim) where {N}
    i = I[dim]
    ntuple(N) do d
        if d == dim
            if i < first(ax)
                return (first(ax) + (first(ax) - i))
            elseif i > last(ax)
                return (last(ax) - (i - last(ax)))
            end
        end
        return I[d]
    end |> CartesianIndex
end

# return the two closest indices in the axis for the given dimension
# as well as the distance from the closest one.
function _wrap_index_neumann_gradient(I::CartesianIndex{N}, ax, dim) where {N}
    i = I[dim]
    a, b = first(ax), last(ax)
    if i < a
        Ion = ntuple(d -> d == dim ? a : I[d], N) |> CartesianIndex
        Iin = ntuple(d -> d == dim ? a + 1 : I[d], N) |> CartesianIndex
        return Ion, Iin, a - i
    elseif i > b
        Ion = ntuple(d -> d == dim ? b : I[d], N) |> CartesianIndex
        Iin = ntuple(d -> d == dim ? b - 1 : I[d], N) |> CartesianIndex
        return Ion, Iin, i - b
    end
    return I, I, 0
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
const CartesianMeshField{V,M<:CartesianGrid,B} = MeshField{V,M,B}

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

function _getindex(
    ϕ::CartesianMeshField,
    I::CartesianIndex{N},
    ::NeumannGradientBC,
    d,
) where {N}
    ax = axes(ϕ)[abs(d)]
    # compute mirror index to I[d]
    # TODO
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
