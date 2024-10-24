using LevelSetMethods
using Documenter
using GLMakie
using MMG_jll
using MarchingCubes

DocMeta.setdocmeta!(
    LevelSetMethods,
    :DocTestSetup,
    :(using LevelSetMethods);
    recursive = true,
)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
    file != "index.md" && splitext(file)[2] == ".md" && occursin(r"^\d", file)
]

modules = [LevelSetMethods]
for extension in [:MakieExt, :MMGSurfaceExt, :MMGVolumeExt]
    ext = Base.get_extension(LevelSetMethods, extension)
    isnothing(ext) && "error loading $ext"
    push!(modules, ext)
end

makedocs(;
    modules,
    authors = "Luiz M. Faria and Nicolas Lebbe",
    repo = "https://github.com/maltezfaria/LevelSetMethods.jl/blob/{commit}{path}#{line}",
    sitename = "LevelSetMethods.jl",
    format = Documenter.HTML(;
        canonical = "https://maltezfaria.github.io/LevelSetMethods.jl",
    ),
    pages = ["index.md"; "advection_example.md"; numbered_pages],
    pagesonly = true, # ignore .md files not in the pages list
)

deploydocs(; repo = "github.com/maltezfaria/LevelSetMethods.jl", push_preview = true)
