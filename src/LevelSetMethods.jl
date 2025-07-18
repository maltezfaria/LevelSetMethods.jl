module LevelSetMethods

using LinearAlgebra
using StaticArrays

include("meshes.jl")
include("boundaryconditions.jl")
include("meshfield.jl")
include("levelset.jl")
include("derivatives.jl")
include("levelsetterms.jl")
include("timestepping.jl")
include("levelsetequation.jl")

export CartesianGrid,
    MeshField,
    LevelSet,
    PeriodicBC,
    NeumannBC,
    NeumannGradientBC,
    DirichletBC,
    AdvectionTerm,
    CurvatureTerm,
    NormalMotionTerm,
    ReinitializationTerm,
    ForwardEuler,
    RK2,
    RK3,
    Upwind,
    WENO5,
    LevelSetEquation,
    integrate!,
    current_time,
    reinitialize!

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
