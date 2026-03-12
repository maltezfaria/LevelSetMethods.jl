"""
    abstract type BoundaryCondition

Types used to specify boundary conditions.
"""
abstract type BoundaryCondition end

"""
    struct PeriodicBC <: BoundaryCondition

Singleton type representing periodic boundary conditions.
"""
struct PeriodicBC <: BoundaryCondition end

"""
    struct ExtrapolationBC{P} <: BoundaryCondition

P-th order one-sided polynomial extrapolation boundary condition. Uses the `P`
nearest interior cells to construct a degree `P-1` polynomial that is
extrapolated into the ghost region.

`ExtrapolationBC{1}` (aliased as [`NeumannBC`](@ref)) gives constant extension
(∂ϕ/∂n = 0 at the boundary face). `ExtrapolationBC{2}` (aliased as
[`NeumannGradientBC`](@ref)) gives linear extrapolation (∂²ϕ/∂n² = 0).

```jldoctest
using LevelSetMethods
ExtrapolationBC(5)

# output

ExtrapolationBC{5}()
```
"""
struct ExtrapolationBC{P} <: BoundaryCondition end

"""
    NeumannBC = ExtrapolationBC{1}

Homogeneous Neumann boundary condition (∂ϕ/∂n = 0 at the boundary face).
Alias for [`ExtrapolationBC{1}`](@ref): ghost cells take the value of the
nearest boundary node (constant extension).
"""
const NeumannBC = ExtrapolationBC{1}

"""
    NeumannGradientBC = ExtrapolationBC{2}

Homogeneous Neumann gradient boundary condition (∂²ϕ/∂n² = 0).
Alias for [`ExtrapolationBC{2}`](@ref): linear extrapolation into ghost cells.
"""
const NeumannGradientBC = ExtrapolationBC{2}

ExtrapolationBC(p::Int) = ExtrapolationBC{p}()

"""
    _lagrange_extrap_weight(j, k, P)

Lagrange weight for the `j`-th interior node (0-indexed) when extrapolating
to a ghost point at distance `k` outside the boundary, using `P` nodes total.

Nodes are at positions 0, 1, …, P-1 (relative to the boundary);
the ghost is at position -k.  The weight is

    wⱼ = ∏_{m=0, m≠j}^{P-1}  (-k - m) / (j - m)
"""
function _lagrange_extrap_weight(j::Int, k::Int, P::Int)
    w = 1.0
    for m in 0:(P - 1)
        m == j && continue
        w *= (-k - m) / (j - m)
    end
    return w
end

"""
    struct DirichletBC{T} <: BoundaryCondition

A Dirichlet boundary condition taking values of `f(x)` at the boundary.
"""
struct DirichletBC{T} <: BoundaryCondition
    f::T
end

DirichletBC() = DirichletBC(0.0)
DirichletBC(a::Real) = DirichletBC(x -> float(a))
