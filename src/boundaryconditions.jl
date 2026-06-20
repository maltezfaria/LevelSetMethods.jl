"""
    abstract type BoundaryCondition

Abstract supertype for boundary conditions. A boundary condition determines the value of a
field at *ghost* indices that lie outside the mesh, closing finite-difference stencils near
the domain boundary.

A concrete subtype `bc` must implement:

  - [`bc_stencil`](@ref)`(bc, I, ax, dim)`: return an iterable of `(weight, index)` pairs whose
    weighted sum `Σ wⱼ · ϕ[Iⱼ]` gives the ghost value at the out-of-bounds index `I` along
    dimension `dim` (with `ax` the in-bounds range). `dim` is part of the signature because an
    index can be out of bounds in several dimensions at once (e.g. a corner ghost), and each
    returned index is resolved along the remaining dimensions by the caller.

[`bc_stencil`](@ref) is a *pure* function of indices: it never touches the field, so the
field-reading and accumulation live entirely in the indexing code (see [`_getindexbc`](@ref)).
"""
abstract type BoundaryCondition end

"""
    struct PeriodicBC <: BoundaryCondition

Singleton type representing periodic boundary conditions. Ghost cells are filled by wrapping
around to the opposite side of the domain.
"""
struct PeriodicBC <: BoundaryCondition end

"""
    struct ExtrapolationBC{P} <: BoundaryCondition

Degree-P one-sided polynomial extrapolation boundary condition. Uses the `P+1`
nearest interior cells to construct a degree-P polynomial that is extrapolated
into the ghost region.

`ExtrapolationBC{0}` (aliased as [`NeumannBC`](@ref)) gives constant extension
(∂ϕ/∂n = 0 at the boundary face). `ExtrapolationBC{1}` (aliased as
[`LinearExtrapolationBC`](@ref)) gives linear extrapolation (∂²ϕ/∂n² = 0).
"""
struct ExtrapolationBC{_P} <: BoundaryCondition
    function ExtrapolationBC{P}() where {P}
        P >= 0 || throw(ArgumentError("extrapolation order P must be at least 0"))
        return new{P}()
    end
end
ExtrapolationBC(p::Int) = ExtrapolationBC{p}()

"""
    const NeumannBC = ExtrapolationBC{0}

Homogeneous Neumann boundary condition (∂ϕ/∂n = 0 at the boundary face).
Alias for [`ExtrapolationBC{0}`](@ref): ghost cells take the value of the
nearest boundary node (constant extension).
"""
const NeumannBC = ExtrapolationBC{0}

"""
    LinearExtrapolationBC = ExtrapolationBC{1}

Alias for [`ExtrapolationBC{1}`](@ref): linear extrapolation into ghost cells. Corresponds
to ∂²ϕ/∂n² = 0 at the boundary face.
"""
const LinearExtrapolationBC = ExtrapolationBC{1}

"""
    struct SymmetryBC <: BoundaryCondition

Symmetry-plane (reflective) boundary condition: the field is mirrored across the boundary
node, `ϕ[b - k] = ϕ[b + k]`. Like [`NeumannBC`](@ref) it enforces ∂ϕ/∂n = 0, but by
*reflection* rather than flat extension, so it also preserves curvature symmetrically and
makes the interface meet the boundary perpendicularly. This is the correct condition on a
symmetry axis, e.g. for axisymmetric simulations.
"""
struct SymmetryBC <: BoundaryCondition end


"""
    _lagrange_extrap_weight(j, k, P)

Lagrange weight for the `j`-th interior node (0-indexed) when extrapolating
to a ghost point at distance `k` outside the boundary, using a degree-P
polynomial fitted to P+1 nodes.

Nodes are at positions 0, 1, …, P (relative to the boundary);
the ghost is at position -k.  The weight is

# TODO: maybe write this as proper latex math?
    wⱼ = ∏_{m=0, m≠j}^{P}  (-k - m) / (j - m)
"""
function _lagrange_extrap_weight(j::Int, k::Int, P::Int)
    w = 1.0
    for m in 0:P
        m == j && continue
        w *= (-k - m) / (j - m)
    end
    return w
end

Base.show(io::IO, ::PeriodicBC) = print(io, "Periodic")
Base.show(io::IO, ::ExtrapolationBC{0}) = print(io, "Neumann")
Base.show(io::IO, ::ExtrapolationBC{1}) = print(io, "Linear extrapolation")
Base.show(io::IO, ::ExtrapolationBC{P}) where {P} = print(io, "Degree $P extrapolation")
Base.show(io::IO, ::SymmetryBC) = print(io, "Symmetry")

# Wrap an out-of-bounds index back into the grid for a periodic boundary along `dim`.
# TODO: is there no periodic wrapping
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
    bc_stencil(bc::BoundaryCondition, I, ax, dim) -> NTuple of (weight, index)

Express the ghost value at the out-of-bounds index `I` (along dimension `dim`, with in-bounds
range `ax`) as a weighted sum `Σ wⱼ · ϕ[Iⱼ]`, returned as a tuple of `(weight, index)` pairs.
Each returned index is in range along `dim` but may still be out of bounds in other
dimensions; the caller ([`_getindexbc`](@ref)) resolves those by recursion. Primary interface
method of [`BoundaryCondition`](@ref); pure function of indices, with no field access.
"""
function bc_stencil end

bc_stencil(::PeriodicBC, I, ax, dim) = ((1.0, _wrap_index_periodic(I, ax, dim)),)

function bc_stencil(::ExtrapolationBC{P}, I::CartesianIndex{N}, ax, dim) where {N, P}
    k = I[dim] < first(ax) ? (first(ax) - I[dim]) : (I[dim] - last(ax))
    b = I[dim] < first(ax) ? first(ax) : last(ax)
    # d = ±1 flips direction so both boundaries map to the same local coordinate
    d = I[dim] < first(ax) ? 1 : -1
    return ntuple(Val(P + 1)) do jp1
        j = jp1 - 1
        Ij = CartesianIndex(ntuple(s -> s == dim ? b + d * j : I[s], Val(N)))
        (_lagrange_extrap_weight(j, k, P), Ij)
    end
end

function bc_stencil(::SymmetryBC, I::CartesianIndex{N}, ax, dim) where {N}
    k = I[dim] < first(ax) ? (first(ax) - I[dim]) : (I[dim] - last(ax))
    b = I[dim] < first(ax) ? first(ax) : last(ax)
    # reflect about the boundary node: the ghost at b - k mirrors the interior node at b + k
    d = I[dim] < first(ax) ? 1 : -1
    Imirror = CartesianIndex(ntuple(s -> s == dim ? b + d * k : I[s], Val(N)))
    return ((1.0, Imirror),)
end

"""
    _normalize_bc(bc, dim)

Normalize the `bc` argument into a `dim`-tuple of `(left, right)` pairs, one per spatial
dimension, as expected by [`_add_boundary_conditions`](@ref).

- A single `BoundaryCondition` is applied to all sides.
- A length-`dim` collection applies each entry to both sides of the corresponding dimension.
- A length-`dim` collection of 2-tuples applies each entry as `(left, right)` for that
  dimension.
"""
function _normalize_bc(bc, dim)
    isa(bc, BoundaryCondition) && return ntuple(_ -> (bc, bc), dim)
    length(bc) == dim || throw(ArgumentError("invalid number of boundary conditions"))
    return ntuple(i -> _normalize_bc_pair(bc[i], i), dim)
end

"""
    _normalize_bc_pair(bc, dim) -> (left, right)

Normalize the boundary condition(s) for a single dimension `dim` into a `(left, right)`
pair. A single `BoundaryCondition` applies to both sides; a length-2 collection is taken as
`(left, right)`. Periodicity must hold on both sides of a dimension or neither.
"""
function _normalize_bc_pair(bc, dim)
    isa(bc, BoundaryCondition) && return (bc, bc)
    (length(bc) == 2 && all(b -> isa(b, BoundaryCondition), bc)) ||
        throw(ArgumentError("invalid boundary condition for dimension $dim"))
    left, right = bc
    (left isa PeriodicBC) ⊻ (right isa PeriodicBC) && throw(
        ArgumentError("periodic boundary conditions cannot be mixed with others in dimension $dim"),
    )
    return (left, right)
end

"""
    _bc_str(bcs)

Format a boundary-conditions tuple (as returned by `boundary_conditions`) into a
compact human-readable string. Each element of `bcs` is a `(left, right)` pair for
one spatial dimension.
"""
function _bc_str(bcs::Tuple)
    N = length(bcs)
    dim_names = N <= 3 ? ("x", "y", "z") : ntuple(i -> "d$i", N)
    all_bcs = [bcs[d][s] for d in 1:N for s in 1:2]
    if all(typeof(b) == typeof(all_bcs[1]) for b in all_bcs)
        return "$(sprint(show, all_bcs[1])) (all)"
    end
    parts = ntuple(N) do d
        bl, br = bcs[d]
        side = typeof(bl) == typeof(br) ? sprint(show, bl) : "$(sprint(show, bl)) ↔ $(sprint(show, br))"
        "$(dim_names[d]): $side"
    end
    return join(parts, ", ")
end
