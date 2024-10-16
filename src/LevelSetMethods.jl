module LevelSetMethods

using LinearAlgebra
using StaticArrays
using RecipesBase

include("meshes.jl")
include("boundaryconditions.jl")
include("meshfield.jl")
include("derivatives.jl")
include("levelsetterms.jl")
include("timestepping.jl")
include("levelsetequation.jl")

export CartesianGrid,
    meshsize,
    SVector,
    MeshField,
    LevelSet,
    PeriodicBC,
    AdvectionTerm,
    CurvatureTerm,
    NormalAdvectionTerm,
    compute_terms,
    add_circle!,
    add_rectangle!,
    ForwardEuler,
    RK2,
    LevelSetEquation,
    integrate!

end # module
