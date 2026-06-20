module LevelSetMethods

using ForwardDiff
using DiffResults
using LinearAlgebra
using SparseArrays
using StaticArrays
using NearestNeighbors

include("meshes.jl")
include("boundaryconditions.jl")
include("bernstein.jl")
include("meshfield.jl")
include("levelsetops.jl")
include("derivatives.jl")
include("interpolation.jl")
include("sdf.jl")
include("reinitializer.jl")
include("velocityextension.jl")
include("levelsetterms.jl")
include("timestepping.jl")
include("levelsetequation.jl")

export
    AdvectionTerm,
    CartesianGrid,
    CurvatureTerm,
    EikonalReinitializationTerm,
    ExtrapolationBC,
    ForwardEuler,
    InterpolatedField,
    LevelSetEquation,
    LinearExtrapolationBC,
    MeshField,
    NarrowBandMeshField,
    NeumannBC,
    NormalMotionTerm,
    PeriodicBC,
    RK2,
    RK3,
    SemiImplicitI2OE,
    SymmetryBC,
    Upwind,
    WENO5,
    active_cellindices,
    active_nodeindices,
    cellindices,
    current_state,
    current_time,
    extend_along_normals!,
    getcell,
    getnode,
    integrate!,
    nodeindices,
    reinitialize!,
    update_band!


"""
    makie_theme()

Return a Makie theme for plots of level-set functions.

!!! note
    Requires loading `Makie.jl` to activate the extension.
"""
makie_theme(args...) =
    error("Makie extension not loaded. Load Makie to use this functionality.")

"""
    set_makie_theme!()

Set the `Makie` theme to [`LevelSetMethods.makie_theme()`](@ref).

!!! note
    Requires loading `Makie.jl` to activate the extension.
"""
set_makie_theme!(args...) =
    error("Makie extension not loaded. Load Makie to use this functionality.")

"""
    export_volume_mesh(eq::LevelSetEquation, filename; kwargs...)

Export a 3D volume mesh of the interior domain (where ϕ < 0) to `filename`.

!!! note
    Requires loading `MMG_jll.jl` to activate the extension.
"""
export_volume_mesh(args...; kwargs...) =
    error("MMG extension not loaded. Load MMG_jll to use this functionality.")

"""
    export_surface_mesh(eq::LevelSetEquation, filename; kwargs...)

Export a surface mesh of the 3D interface (where ϕ = 0) to `filename`.

!!! note
    Requires loading `MMG_jll.jl` to activate the extension.
"""
export_surface_mesh(args...; kwargs...) =
    error("MMG extension not loaded. Load MMG_jll to use this functionality.")

end # module
