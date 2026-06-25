```@meta
CurrentModule = LevelSetMethods
```

# [Grids and mesh fields](@id grids)

Every computation in `LevelSetMethods.jl` rests on two ingredients: a *mesh* that
discretizes the domain, and *data* attached to that mesh. This page covers both — the
[`CartesianGrid`](@ref) mesh and the [`AbstractMeshField`](@ref LevelSetMethods.AbstractMeshField)
data built on it. The next page, [Level sets](@ref geometry), focuses on the real-valued
fields the level-set method actually evolves.

## The Cartesian mesh

A [`CartesianGrid`](@ref) is a uniform, axis-aligned mesh covering an `N`-dimensional box
``[lc, hc]``, described by its lower corner `lc`, upper corner `hc`, and node count `n` per
direction. The dimension `N` is inferred from the corners, so the same constructor builds
meshes in any number of dimensions:

```@example grids
using LevelSetMethods
grid = CartesianGrid((-1, -1), (1, 1), (64, 64))
```

To specify a target resolution rather than a node count, pass `meshsize` instead; the domain
is honored exactly and the cell count rounded *up*, so the spacing is never coarser than
requested:

```@example grids
CartesianGrid((0, 0), (1, 1); meshsize = 0.3)
```

### Nodes and cells

A Cartesian mesh has two kinds of geometric entities, each addressed by a `CartesianIndex`:

- **Nodes** are the grid points, indexed `1:n[d]` in each dimension `d`. Field values live here.
- **Cells** are the hyperrectangles between neighbouring nodes; cell `I` spans node `I` to
  node `I + 1`, giving `n[d] - 1` cells per dimension.

Iterate over them with [`nodeindices`](@ref) / [`cellindices`](@ref), recover the geometry of
an index with [`getnode`](@ref) / [`getcell`](@ref), and query the mesh with `size`, `ndims`,
and [`LevelSetMethods.meshsize`](@ref):

```@example grids
getnode(grid, 1, 1), size(grid), LevelSetMethods.meshsize(grid)
```

## Attaching data: mesh fields

A mesh on its own carries no data. To represent a quantity — a level-set function, a
velocity, a speed — you attach a value to each node, producing a *mesh field*. All mesh
fields are subtypes of [`AbstractMeshField`](@ref LevelSetMethods.AbstractMeshField) and
share one interface (index by `CartesianIndex`, query the underlying `mesh`, iterate over the
nodes), so code written against that interface works regardless of how the values are stored.
Two storage strategies are provided, differing *only* in where the values are kept: a
**dense** [`MeshField`](@ref) stores a value at *every* node, while a **sparse**
[`NarrowBandMeshField`](@ref) stores values only on a thin band of nodes around the interface.

### Dense fields

The most direct way to build a [`MeshField`](@ref) is to sample a function at each node:

```@example grids
ϕ = MeshField(x -> x[1]^2 + x[2]^2 - 0.5^2, grid)
```

The function receives the node coordinates (as an `SVector`) and may return any value type —
a scalar for a level set, an `SVector` for a velocity field, and so on. Values are read and
written by Cartesian index, exactly like an array (`ϕ[3, 3]`). You can also build a field
directly from an array of values, which is handy when the data comes from elsewhere:

```@example grids
using LinearAlgebra
vals = [norm(getnode(grid, I)) - 0.5 for I in nodeindices(grid)]
MeshField(vals, grid)
```

### Sparse fields

When the interface fills only a thin region of the domain, storing ``\phi`` at *every* node
is wasteful. A [`NarrowBandMeshField`](@ref) keeps values only on a band of nodes around the
interface while presenting the very same interface as a dense field — most easily built by
restricting an existing [`MeshField`](@ref) to a band:

```@example grids
nb = NarrowBandMeshField(ϕ; nlayers = 3)
@assert 300 < length(active_nodeindices(nb)) < 1000 # hide
nb
```

The display reports how many nodes are *active* (stored in the band) out of the whole mesh.
The two storage strategies are reconciled by a single distinction in the interface:
[`nodeindices`](@ref) / [`cellindices`](@ref) range over the entire mesh, while
[`active_nodeindices`](@ref) / [`active_cellindices`](@ref) range over the subset a field
actually stores. For a dense [`MeshField`](@ref) the two coincide; for a
[`NarrowBandMeshField`](@ref) the active set is just the band. Code that should work for
either storage iterates over the *active* set — the idiom used throughout the package:

```@example grids
sum(I -> nb[I], active_nodeindices(nb))   # works for dense and narrow-band fields alike
```

See [Narrow-band fields](@ref narrow-band) for when and how to use the band in detail.

## Boundary conditions

Finite-difference and WENO stencils reach beyond the mesh near its borders. To resolve those
out-of-grid reads, any mesh field can carry *boundary conditions* via the `bc` keyword;
indexing outside the mesh is then well defined, with the condition supplying the ghost value:

```@example grids
ψ = MeshField(x -> x[1]^2 + x[2]^2 - 0.5^2, grid; bc = NeumannBC())
ψ[0, 3]   # one node past the left border, filled by the Neumann condition
```

A field *without* boundary conditions throws on an out-of-grid index rather than silently
guessing. See [Boundary conditions](@ref boundary-conditions) for the full list of available
conditions and how to apply different ones per face.

## Real-valued fields are level sets

There is no separate "level set" type in this package. A mesh field whose values are real
numbers *is* a level set, by the convention that its zero contour ``\{x : \phi(x) = 0\}``
represents the interface, with ``\phi < 0`` inside and ``\phi > 0`` outside. Fields with
non-real values (such as an `SVector`-valued velocity) are ordinary mesh fields too; they
simply are not interpreted as level sets. The next page, [Level sets](@ref geometry), is
devoted to building real-valued fields and combining them into geometries.
