"""
    abstract type BoundaryCondition

Singleton types used in [`applybc!`](@ref) for dispatch purposes. Subtypes of
`BoundaryCondition` must implement [`wrap_index`](@ref).
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

Neumann boundary condition with a constant value.
"""
struct Neumann <: BoundaryCondition
    value::Float64
end
