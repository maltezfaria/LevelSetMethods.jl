"""
    abstract type AbstractMesh{N,T}

An abstract mesh structure in dimension `N` with primite data of type `T`.
"""
abstract type AbstractMesh{N,T} end

struct CartesianGrid{N,T} <: AbstractMesh{N,T}
    lc::SVector{N,T}
    hc::SVector{N,T}
    n::NTuple{N,Int}
end

"""
    CartesianGrid(lc, hc, n)

Create a uniform cartesian grid with lower corner `lc`, upper corner `hc` and and `n` nodes
in each direction.

# Examples

```jldoctest; output = true
using LevelSetMethods
a = (0, 0)
b = (1, 1)
n = (10, 4)
grid = CartesianGrid(a, b, n)

# output

CartesianGrid{2, Int64}([0, 0], [1, 1], (10, 4))
```
"""
function CartesianGrid(lc, hc, n)
    length(lc) == length(hc) == length(n) ||
        throw(ArgumentError("all arguments must have the same length"))
    N   = length(lc)
    lc_ = SVector{N,eltype(lc)}(lc...)
    hc_ = SVector{N,eltype(hc)}(hc...)
    n   = ntuple(i -> Int(n[i]), N)
    return CartesianGrid(promote(lc_, hc_)..., n)
end

grid1d(g::CartesianGrid{N}) where {N} = ntuple(i -> grid1d(g, i), N)
grid1d(g::CartesianGrid, dim) = LinRange(g.lc[dim], g.hc[dim], g.n[dim])

dimension(g::CartesianGrid{N}) where {N} = N

xgrid(g::CartesianGrid) = grid1d(g, 1)
ygrid(g::CartesianGrid) = grid1d(g, 2)
zgrid(g::CartesianGrid) = grid1d(g, 3)

meshsize(g::CartesianGrid)      = (g.hc .- g.lc) ./ (g.n .- 1)
meshsize(g::CartesianGrid, dim) = (g.hc[dim] - g.lc[dim]) / (g.n[dim] - 1)

Base.size(g::CartesianGrid) = g.n
Base.length(g) = prod(size(g))

function Base.getindex(g::CartesianGrid{N}, I::CartesianIndex{N}) where {N}
    I ∈ CartesianIndices(g) || throw(ArgumentError("index $I is out of bounds"))
    return _getindex(g, I)
end

Base.getindex(g::CartesianGrid, I::Int...) = g[CartesianIndex(I...)]

Base.eltype(g::CartesianGrid) = typeof(g.lc)

function _getindex(g::CartesianGrid, I::CartesianIndex)
    N = dimension(g)
    @assert N == length(I)
    ntuple(N) do dim
        return g.lc[dim] + (I[dim] - 1) / (g.n[dim] - 1) * (g.hc[dim] - g.lc[dim])
    end |> SVector
end
_getindex(g::CartesianGrid, I::Int...) = _getindex(g, CartesianIndex(I...))

Base.CartesianIndices(g::CartesianGrid) = CartesianIndices(size(g))
Base.eachindex(g::CartesianGrid) = CartesianIndices(g)

# NOTE: remove?
function interior_indices(g::CartesianGrid, P)
    N  = dimension(g)
    sz = size(g)
    I  = ntuple(N) do dim
        return (P+1):(sz[dim]-P)
    end
    return CartesianIndices(I)
end

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
