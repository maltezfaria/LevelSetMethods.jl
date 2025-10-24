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

"""
    reinitialize!(ϕ::LevelSet; upsample=2, maxiters=10, xtol=1e-8)

Reinitializes the level set `ϕ` to a signed distance, modifying it in place.

The method works by first sampling the zero-level set of the interface, and then for each
grid point, finding the closest point on the interface using a Newton-based method. The
distance to the closest point is then used as the new value of the level set at that grid
point, with the sign determined by the original level set value. See [saye2014high](@cite)
for more details.

## Arguments

  - `ϕ`: The level set to reinitialize.

## Keyword Arguments

  - `upsample`: number of samples to take in each cell when sampling the interface.
    Higher values yield better initial guesses for the closest point search, but increase
    the computational cost.
  - `maxiters`: maximum number of iterations to use in the Newton's method
    to find the closest point on the interface.
  - `xtol`: convergence tolerance for the Newton's method. The iterations stop when
    the change in position is less than `xtol`.

!!! note
    This functionality is provided by the `ReinitializationExt` module, which
    requires loading `Interpolations.jl` and `NearestNeighbors.jl`.
"""
function reinitialize! end

# tomatoes tomatos ...
reinitialise!(args...; kwargs...) = reinitialize!(args...; kwargs...)


end # module
