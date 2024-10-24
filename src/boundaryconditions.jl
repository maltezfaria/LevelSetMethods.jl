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
    struct NeumannBC <: BoundaryCondition

Homogenous Neumann boundary condition.
"""
struct NeumannBC <: BoundaryCondition end

"""
    struct DirichletBC{T} <: BoundaryCondition

A Dirichlet boundary condition taking values of `f(x)` at the boundary.
"""
struct DirichletBC{T} <: BoundaryCondition
    f::T
end

DirichletBC() = DirichletBC(0.0)
DirichletBC(a::Real) = DirichletBC(x -> float(a))
