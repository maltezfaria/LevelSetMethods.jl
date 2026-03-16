module LevelSetMethods

using ForwardDiff
using DiffResults
using LinearAlgebra
using SparseArrays
using StaticArrays
using NearestNeighbors

include("meshes.jl")
include("boundaryconditions.jl")
include("meshfield.jl")
include("levelset.jl")
include("derivatives.jl")
include("interpolation.jl")
include("reinitializer.jl")
include("velocityextension.jl")
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
    ExtrapolationBC,
    AdvectionTerm,
    CurvatureTerm,
    NormalMotionTerm,
    extend_along_normals!,
    EikonalReinitializationTerm,
    ForwardEuler,
    RK2,
    RK3,
    SemiImplicitI2OE,
    Upwind,
    WENO5,
    LevelSetEquation,
    NewtonReinitializer,
    integrate!,
    current_state,
    current_time,
    reinitialize!,
    interpolate


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

"""
    export_volume_mesh(eq::LevelSetEquation, filename; kwargs...)

Export a 3D volume mesh of the interior domain (where ϕ < 0) to `filename`.
Requires the `MMG` extension to be loaded.
"""
function export_volume_mesh end

"""
    export_surface_mesh(eq::LevelSetEquation, filename; kwargs...)

Export a surface mesh of the 3D interface (where ϕ = 0) to `filename`.
Requires the `MMG` extension to be loaded.
"""
function export_surface_mesh end

end # module
