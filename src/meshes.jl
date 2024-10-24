"""
    abstract type AbstractMesh{N,T}

An abstract mesh structure in dimension `N` with primite data of type `T`.
"""
abstract type AbstractMesh{N,T} end

struct CartesianGrid{N,T} <: AbstractMesh{N,T}
    grid1d::NTuple{N,LinRange{T,Int64}}
end

"""
    CartesianGrid(lc, hc, n)

Create a uniform cartesian grid with lower corner `lc`, upper corner `hc` and and `n` nodes
in each direction.

# Examples

```jldoctest; output = false
a = (0, 0)
b = (1, 1)
n = (10, 4)
grid = CartesianGrid(a, b, n)

# output

CartesianGrid{2, Float64}((LinRange{Float64}(0.0, 1.0, 10), LinRange{Float64}(0.0, 1.0, 4)))
```
"""
function CartesianGrid(lc, hc, n)
    length(lc) == length(hc) == length(n) ||
        throw(ArgumentError("all arguments must have the same length"))
    grids = ntuple(i -> LinRange(lc[i], hc[i], n[i]), length(lc))
    return CartesianGrid(promote(grids...))
end

grid1d(g::CartesianGrid)      = g.grid1d
grid1d(g::CartesianGrid, dim) = g.grid1d[dim]

dimension(g::CartesianGrid{N}) where {N} = N

xgrid(g::CartesianGrid) = g.grid1d[1]
ygrid(g::CartesianGrid) = g.grid1d[2]
zgrid(g::CartesianGrid) = g.grid1d[3]

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
Base.eachindex(g::CartesianGrid) = CartesianIndices(g)

function interior_indices(g::CartesianGrid, P)
    N  = dimension(g)
    sz = size(g)
    I  = ntuple(N) do dim
        return P+1:sz[dim]-P
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
