using LevelSetMethods
using Documenter
using CairoMakie
using GLMakie
# CairoMakie renders almost everything (2D figures and gifs, headless-friendly); the 3D pages
# call GLMakie.activate!() themselves, since only GLMakie can draw the `volume` isosurfaces.
CairoMakie.activate!()
using MMG_jll
using MarchingCubes
using NearestNeighbors
using StaticArrays
using DocumenterCitations

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :numeric)

modules = [LevelSetMethods]
for extension in [:MakieExt, :MMGSurfaceExt, :MMGVolumeExt]
    ext = Base.get_extension(LevelSetMethods, extension)
    if isnothing(ext)
        @warn "extension $extension not loaded"
    else
        push!(modules, ext)
    end
end


makedocs(;
    modules,
    authors = "Luiz M. Faria and Nicolas Lebbe",
    repo = "https://github.com/maltezfaria/LevelSetMethods.jl/blob/{commit}{path}#{line}",
    sitename = "LevelSetMethods.jl",
    format = Documenter.HTML(;
        canonical = "https://maltezfaria.github.io/LevelSetMethods.jl",
        collapselevel = 2,
        assets = String["assets/citations.css"],
    ), pages = [
        "Home" => "index.md",
        "Building & solving" => [
            "grids.md",                 # Grids & mesh fields
            "geometry.md",              # Creating level sets
            "levelset-equation.md",     # The level-set equation
            "terms.md",                 # Level-set terms
            "time-integrators.md",      # Time integration
            "boundary-conditions.md",   # Boundary conditions
        ],
        "Advanced topics" => [
            "signed-distance.md",       # Closest-point reinitialization
            "velocity-extension.md",    # Velocity extension
            "narrow-band.md",           # Narrow-band fields
            "interpolation.md",         # Interpolation
            "geometry-queries.md",      # Geometric quantities
        ],
        "Extensions" => [
            "extension-makie.md",
            "extension-mmg.md",
            "extension-implicit-integration.md",
        ],
        "Examples" => [
            "example-zalesak.md",
            "example-shape-optim.md",
        ],
        "Reference" => "reference.md",
    ],
    pagesonly = true, # ignore .md files not in the pages list
    warnonly = true,
    plugins = [bib],
    draft = false,
)

deploydocs(; repo = "github.com/maltezfaria/LevelSetMethods.jl", devbranch = "main", push_preview = true)

GLMakie.closeall()
