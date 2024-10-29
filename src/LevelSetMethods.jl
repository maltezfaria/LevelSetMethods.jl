module LevelSetMethods

using LinearAlgebra
using StaticArrays

include("utils.jl")
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
    NeumannBC,
    DirichletBC,
    AdvectionTerm,
    CurvatureTerm,
    NormalMotionTerm,
    ReinitializationTerm,
    curvature,
    compute_terms,
    add_circle!,
    remove_circle!,
    add_rectangle!,
    remove_rectangle!,
    ForwardEuler,
    RK2,
    Upwind,
    WENO5,
    LevelSetEquation,
    integrate!,
    current_time

"""
    makie_theme()

Return a Makie theme for plots of level-set functions.
"""
function makie_theme end

"""
    set_makie_theme!()

Set the `Makie` theme to [`LevelSetMethods.makie_theme()`](@ref).
"""
function set_makie_theme! end

function export_volume_mesh end
function export_surface_mesh end

end # module
