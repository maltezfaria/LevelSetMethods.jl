module LevelSetMethods

using LinearAlgebra
using SparseArrays
using StaticArrays

include("meshes.jl")
include("boundaryconditions.jl")
include("meshfield.jl")
include("levelset.jl")
include("derivatives.jl")
include("velocityextension.jl")
include("levelsetterms.jl")
include("timestepping.jl")
include("reinitializer.jl")
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
    extend_along_normals!,
    ReinitializationTerm,
    ForwardEuler,
    RK2,
    RK3,
    SemiImplicitI2OE,
    Upwind,
    WENO5,
    LevelSetEquation,
    Reinitializer,
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

"""
    reinitialize!(ϕ::LevelSet, reinitializer = Reinitializer())

Reinitializes the level set `ϕ` to a signed distance, modifying it in place.

The method works by first sampling the zero-level set of the interface, and then for each
grid point, finding the closest point on the interface using a Newton-based method. The
distance to the closest point is then used as the new value of the level set at that grid
point, with the sign determined by the original level set value. See [saye2014high](@cite)
for more details.

## Arguments

  - `ϕ`: The level set to reinitialize.
  - `reinitializer`: Configuration for the reinitialization. Defaults to `Reinitializer()`.
    See [`Reinitializer`](@ref) for details.

!!! note
    This functionality is provided by the `ReinitializationExt` module, which
    requires loading `Interpolations.jl` and `NearestNeighbors.jl`.
"""
function reinitialize!(ϕ, reinitializer)
    error("Reinitialization extension not loaded. Please load the ReinitializationExt module to use this functionality.")
end

# tomatoes tomatos ...
reinitialise!(args...; kwargs...) = reinitialize!(args...; kwargs...)

end # module
