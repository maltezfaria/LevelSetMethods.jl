"""
    abstract type AbstractMesh{N,T}

An abstract mesh structure in dimension `N` with primite data of type `T`.
"""
abstract type AbstractMesh{N, T} end

struct CartesianGrid{N, T} <: AbstractMesh{N, T}
    lc::SVector{N, T}
    hc::SVector{N, T}
    n::NTuple{N, Int}
end

"""
    CartesianGrid(lc, hc, n)

Create a uniform cartesian grid with lower corner `lc`, upper corner `hc` and `n` nodes
in each direction.

Node coordinates are accessed via [`getnode`](@ref); cell geometry via [`getcell`](@ref).
Use [`nodeindices`](@ref) to iterate over all node indices and [`cellindices`](@ref) to
iterate over all cell indices.

# Examples

```jldoctest; output = true
using LevelSetMethods
a = (0, 0)
b = (1, 1)
n = (10, 4)
grid = CartesianGrid(a, b, n)

# output

CartesianGrid in ℝ²
  ├─ domain:  [0.0, 1.0] × [0.0, 1.0]
  ├─ nodes:   10 × 4
  └─ spacing: h = (0.1111, 0.3333)
```
"""
function CartesianGrid(lc, hc, n)
    length(lc) == length(hc) == length(n) ||
        throw(ArgumentError("all arguments must have the same length"))
    N = length(lc)
    lc_ = SVector{N}(float.(lc))
    hc_ = SVector{N}(float.(hc))
    n_ = ntuple(i -> Int(n[i]), N)
    return CartesianGrid(promote(lc_, hc_)..., n_)
end

"""
    grid1d(g::CartesianGrid[, dim::Integer])

Return a `LinRange` of the coordinates along the given dimension `dim`.
If `dim` is not provided, return a tuple of `LinRange`s for all dimensions.
"""
grid1d(g::CartesianGrid{N}) where {N} = ntuple(i -> grid1d(g, i), N)
grid1d(g::CartesianGrid, dim) = LinRange(g.lc[dim], g.hc[dim], g.n[dim])

Base.ndims(g::CartesianGrid{N}) where {N} = N
Base.size(g::CartesianGrid) = g.n
Base.length(g::CartesianGrid) = prod(size(g))
Base.eachindex(g::CartesianGrid) = nodeindices(g)

xgrid(g::CartesianGrid) = grid1d(g, 1)
ygrid(g::CartesianGrid) = grid1d(g, 2)
zgrid(g::CartesianGrid) = grid1d(g, 3)

"""
    meshsize(g::CartesianGrid[, dim::Integer])

Return the spacing between grid nodes along the given dimension `dim`.
If `dim` is not provided, return a `SVector` of spacings for all dimensions.
"""
meshsize(g::CartesianGrid) = (g.hc .- g.lc) ./ (g.n .- 1)
meshsize(g::CartesianGrid, dim) = (g.hc[dim] - g.lc[dim]) / (g.n[dim] - 1)

# Internal unchecked coordinate lookup. Used by boundary-condition code to evaluate
# coordinates at ghost indices that lie outside the grid.
function _getindex(g::CartesianGrid{N, T}, I::CartesianIndex{N}) where {N, T}
    h = meshsize(g)
    return g.lc .+ (SVector{N, T}(I.I) .- 1) .* h
end
_getindex(g::CartesianGrid, I::Int...) = _getindex(g, CartesianIndex(I...))

"""
    getnode(g::CartesianGrid, I::CartesianIndex)
    getnode(g::CartesianGrid, i1, i2, ...)

Return the coordinates of the node at multi-index `I` as an `SVector`.
`I` must be a valid node index (i.e. `I ∈ nodeindices(g)`).
"""
function getnode(g::CartesianGrid{N}, I::CartesianIndex{N}) where {N}
    I ∈ nodeindices(g) || throw(ArgumentError("$I is not a valid node index for this grid"))
    return _getindex(g, I)
end
getnode(g::CartesianGrid, I::Int...) = getnode(g, CartesianIndex(I...))

"""
    nodeindices(g::CartesianGrid)

Return a `CartesianIndices` ranging over all node indices of `g`.
Nodes are indexed `1:n[d]` in each dimension `d`.
"""
nodeindices(g::CartesianGrid) = CartesianIndices(size(g))

"""
    cellindices(g::CartesianGrid)

Return a `CartesianIndices` ranging over all cell indices of `g`.
Cell `I` is the hypercube bounded by nodes `I` and `I + 1` in each dimension.
Cells are indexed `1:n[d]-1` in each dimension `d`.
"""
cellindices(g::CartesianGrid{N}) where {N} = CartesianIndices(ntuple(d -> 1:(g.n[d] - 1), Val(N)))

"""
    struct CartesianCell{N, T}

A cell of a `CartesianGrid`: the axis-aligned hypercube bounded by nodes at `lc` (lower
corner) and `hc` (upper corner). Obtain via `getcell(grid, I)` where `I` is a cell index.
"""
struct CartesianCell{N, T}
    lc::SVector{N, T}
    hc::SVector{N, T}
end

# Internal unchecked cell lookup. Used when extrapolation or boundary handling is required.
function _getcell(g::CartesianGrid{N}, I::CartesianIndex{N}) where {N}
    lc = _getindex(g, I)
    return CartesianCell(lc, lc .+ meshsize(g))
end

"""
    getcell(g::CartesianGrid, I::CartesianIndex)

Return the `CartesianCell` with lower corner at node `I` and upper corner at node `I+1`.
`I` must be a valid cell index, i.e. `I ∈ cellindices(g)`.
"""
function getcell(g::CartesianGrid{N}, I::CartesianIndex{N}) where {N}
    I ∈ cellindices(g) || throw(ArgumentError("$I is not a valid cell index for this grid"))
    return _getcell(g, I)
end

# Display methods

"""
    _superscript(n::Int) -> String

Convert an integer to its Unicode superscript representation, e.g. `2` → `"²"`.
"""
function _superscript(n::Int)
    sups = ('⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹')
    return join(sups[d + 1] for d in reverse(digits(n; base = 10)))
end

"""
    _domain_str(g::CartesianGrid) -> String

Format the domain as `"[lo, hi] × [lo, hi] × …"`.
"""
function _domain_str(g::CartesianGrid{N}) where {N}
    ranges = ntuple(d -> "[$(g.lc[d]), $(g.hc[d])]", N)
    return join(ranges, " × ")
end

"""
    _show_fields(io, g::CartesianGrid; prefix="  ", last=true)

Print domain, nodes, and spacing of `g` as indented tree lines.
When `last=true` the spacing line uses `└─` (terminal); otherwise `├─` (continuing).
"""
function _show_fields(io::IO, g::CartesianGrid{N}; prefix = "  ", last = true) where {N}
    h = meshsize(g)
    h_str = "(" * join(round.(h; sigdigits = 4), ", ") * ")"
    println(io, "$(prefix)├─ domain:  $(_domain_str(g))")
    println(io, "$(prefix)├─ nodes:   $(join(g.n, " × "))")
    connector = last ? "└─" : "├─"
    return if last
        print(io, "$(prefix)$connector spacing: h = $h_str")
    else
        println(io, "$(prefix)$connector spacing: h = $h_str")
    end
end

function Base.show(io::IO, ::MIME"text/plain", g::CartesianGrid{N}) where {N}
    println(io, "CartesianGrid in ℝ$(_superscript(N))")
    return _show_fields(io, g)
end
