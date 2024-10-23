"""
    abstract type BoundaryCondition

Singleton types used to specify boundary conditions. Subtypes of `BoundaryCondition` must
implement [`wrap_index`](@ref).
"""
abstract type BoundaryCondition end

function wrap_index end

"""
    struct PeriodicBC <: BoundaryCondition

Singleton type representing periodic boundary conditions.
"""
struct PeriodicBC <: BoundaryCondition end

"""
    wrap_index(ax, i, bc::PeriodicBC)

Map `i` to a valid index in the range of `ax` using periodic boundary conditions.

For example, given an `ax` going from `1` to `N`, this maps e.g. `0` to `N-1` and `N+1` to
`2`
"""
function wrap_index(ax, i, ::PeriodicBC)
    if i < first(ax)
        return last(ax) - (first(ax) - i)
    elseif i > last(ax)
        return first(ax) + (i - last(ax))
    else
        return i
    end
end

"""
    struct NeumannBC <: BoundaryCondition

Homogenous Neumann boundary condition.
"""
struct NeumannBC <: BoundaryCondition end

"""
    wrap_index(ax, i, bc::NeumannBC)

Wrap index by reflecting it at the boundary.
"""
function wrap_index(ax, i, ::NeumannBC)
    if i < first(ax)
        first(ax) + (first(ax) - i)
    elseif i > last(ax)
        last(ax) - (i - last(ax))
    else
        i
    end
end
