"""
    abstract type AbstractMesh{N,T}

An abstract mesh structure in dimension `N` with primite data of type `T`.
"""
abstract type AbstractMesh{N,T} end

"""
    struct CartesianGrid{N,T} <: AbstractMesh{N,T}

An `N`-dimensional cartesian grid given as the tensor-product of `N` one-dimensional
`LinRange{T}` objects. The grid spacing is therefore constant per dimension.
"""
struct CartesianGrid{N,T} <: AbstractMesh{N,T}
    grid1d::NTuple{N,LinRange{T}}
end

grid1d(g::CartesianGrid)      = g.grid1d
grid1d(g::CartesianGrid, dim) = g.grid1d[dim]

dimension(g::CartesianGrid{N}) where {N} = N

xgrid(g::CartesianGrid) = g.grid1d[1]
ygrid(g::CartesianGrid) = g.grid1d[2]
zgrid(g::CartesianGrid) = g.grid1d[3]

# allow for arguments to be passed as e.g. (x,y) instead of tuple and promote
# type if needed
CartesianGrid(args...) = CartesianGrid(promote(args...))

meshsize(g::CartesianGrid)      = step.(grid1d(g))
meshsize(g::CartesianGrid, dim) = step(grid1d(g, dim))

Base.size(g::CartesianGrid) = length.(g.grid1d)
Base.length(g) = prod(size(g))

function Base.getindex(g::CartesianGrid, I)
    N = dimension(g)
    @assert N == length(I)
    ntuple(N) do dim
        i = I[dim]
        return g.grid1d[dim][i]
    end
end

function Base.getindex(g::CartesianGrid, I...)
    N = dimension(g)
    @assert N == length(I)
    ntuple(N) do dim
        i = I[dim]
        return g.grid1d[dim][i]
    end
end

Base.CartesianIndices(g::CartesianGrid) = CartesianIndices(size(g))

# iterate over all nodes
function Base.iterate(g::CartesianGrid)
    i = first(CartesianIndices(g))
    return g[i], i
end

function Base.iterate(g::CartesianGrid, state)
    idxs = CartesianIndices(g)
    next = iterate(idxs, state)
    if next === nothing
        return nothing
    else
        i, state = next
        return g[i], state
    end
end

# Base.IteratorSize(::Type{CartesianGrid{N}}) where {N} = Base.HasShape{N}()
Base.IteratorSize(::CartesianGrid{N}) where {N} = Base.HasShape{N}()
