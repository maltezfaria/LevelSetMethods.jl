"""
    abstract type BoundaryCondition

Types used to specify boundary conditions.
"""
abstract type BoundaryCondition end

"""
    struct PeriodicBC <: BoundaryCondition

Singleton type representing periodic boundary conditions. Ghost cells are filled by wrapping
around to the opposite side of the domain.

```jldoctest
using LevelSetMethods
PeriodicBC()

# output

Periodic
```

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

```jldoctest
using LevelSetMethods
ExtrapolationBC(4)

# output

Degree 4 extrapolation
```
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
    _lagrange_extrap_weight(j, k, P)

Lagrange weight for the `j`-th interior node (0-indexed) when extrapolating
to a ghost point at distance `k` outside the boundary, using a degree-P
polynomial fitted to P+1 nodes.

Nodes are at positions 0, 1, …, P (relative to the boundary);
the ghost is at position -k.  The weight is

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

"""
    struct DirichletBC{T} <: BoundaryCondition

A Dirichlet boundary condition taking values of `f(x, t)` at the boundary,
where `x` is the spatial coordinate and `t` is the current time.
"""
mutable struct DirichletBC{T} <: BoundaryCondition
    f::T
    t::Float64 # state passed to `f` for time-dependent BCs
end

function DirichletBC(f)
    isempty(methods(f)) &&
        throw(ArgumentError("DirichletBC requires a callable, got $(typeof(f))"))
    return DirichletBC(f, 0.0)
end

"""
    update_bc!(bc::BoundaryCondition, t)

Update the current time stored in a boundary condition. Only meaningful for
[`DirichletBC`](@ref); all other BC types are no-ops.
"""
update_bc!(bc::BoundaryCondition, _) = bc
update_bc!(bc::DirichletBC, t) = (bc.t = t; bc)

Base.show(io::IO, ::PeriodicBC) = print(io, "Periodic")
Base.show(io::IO, ::ExtrapolationBC{P}) where {P} = print(io, "Degree $P extrapolation")
Base.show(io::IO, ::DirichletBC) = print(io, "Dirichlet")

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
    if isa(bc, BoundaryCondition)
        return ntuple(_ -> (bc, bc), dim)
    else
        length(bc) == dim || throw(ArgumentError("invalid number of boundary conditions"))
        return ntuple(dim) do i
            if isa(bc[i], BoundaryCondition)
                return (bc[i], bc[i])
            else
                length(bc[i]) == 2 && all(isa(bc[i][n], BoundaryCondition) for n in 1:2) ||
                    throw(ArgumentError("invalid boundary condition for dimension $i"))
                # check that periodic boundary conditions are not mixed with others
                if any(isa(bc[i][n], PeriodicBC) for n in 1:2)
                    all(isa(bc[i][n], PeriodicBC) for n in 1:2) || throw(
                        ArgumentError(
                            "periodic boundary conditions cannot be mixed with others in dimension $i",
                        ),
                    )
                end
                return (bc[i][1], bc[i][2])
            end
        end
    end
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
