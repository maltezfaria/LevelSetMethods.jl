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
include("levelset.jl")
include("derivatives.jl")
include("interpolation.jl")
include("reinitializer.jl")
include("narrowband.jl")
include("velocityextension.jl")
include("levelsetterms.jl")
include("timestepping.jl")
include("logging.jl")
include("levelsetequation.jl")

export AbstractMeshField,
    AdvectionTerm,
    CartesianGrid,
    CurvatureTerm,
    DirichletBC,
    EikonalReinitializationTerm,
    ExtrapolationBC,
    ForwardEuler,
    LevelSetEquation,
    MeshField,
    NarrowBandMeshField,
    NeumannBC,
    LinearExtrapolationBC,
    NewtonReinitializer,
    NormalMotionTerm,
    PeriodicBC,
    RK2,
    RK3,
    SemiImplicitI2OE,
    Upwind,
    WENO5,
    cellindices,
    check_real_valued,
    current_state,
    current_time,
    extend_along_normals!,
    getcell,
    getnode,
    integrate!,
    nodeindices,
    quadrature,
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

"""
    quadrature(ϕ::AbstractMeshField; order, surface=false, min_mass_fraction=0.0)

Generate a quadrature for the implicit domain defined by `ϕ`.
If `surface=true`, generate a quadrature for the interface `ϕ=0`;
otherwise for the interior `ϕ < 0`.

If `min_mass_fraction > 0`, small cut cells are merged into rectangular supercells
until every integration domain has a mass of at least `min_mass_fraction * M`,
where `M` is the maximum mass across all active leaf cells.

Returns a `Vector` of `(region, quadrature)` pairs, where `region` is a
`CartesianIndices` covering one or more contiguous cells (a single cut cell is
represented as a 1-element `CartesianIndices`).

!!! note
    Requires loading `ImplicitIntegration.jl` to activate the extension.
"""
function quadrature(ϕ; order, surface = false, min_mass_fraction = 0.0)
    error("ImplicitIntegration extension not loaded. Load ImplicitIntegration to use this functionality.")
end

end # module
