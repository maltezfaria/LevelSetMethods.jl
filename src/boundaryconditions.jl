"""
    abstract type BoundaryCondition

Types used to specify boundary conditions.
"""
abstract type BoundaryCondition end
# TODO: define the interface for BoundaryConditions

"""
    struct PeriodicBC <: BoundaryCondition

Singleton type representing periodic boundary conditions.
"""
struct PeriodicBC <: BoundaryCondition end

"""
    struct NeumannBC <: BoundaryCondition

Homogenous Neumann boundary condition.
"""
struct NeumannBC <: BoundaryCondition end

struct DirichletBC{T} <: BoundaryCondition
    value::T
end
