module MakieExt

using Makie
import LevelSetMethods as LSM

function __init__()
    return @info "Loading Makie extension for LevelSetMethods.jl"
end

# NOTE: Makie recipes currently can't modify the Axis (https://discourse.julialang.org/t/makie-plot-recipe-collections/86434)
# so we have to set the theme manually
function LSM.makie_theme()
    return Theme(;
        Axis = (
            xlabel = "x",
            ylabel = "y",
            zlabel = "z",
            # autolimitaspect = 1,
            aspect = AxisAspect(1),
            xgridvisible = false,
            ygridvisible = false,
        ),
        fontsize = 20,
    )
end

function LSM.set_makie_theme!()
    return Makie.set_theme!(LSM.makie_theme())
end

# function Makie.plottype(ϕ::LSM.LevelSet)
#     N = ndims(ϕ)
#     if N == 2
#         return Contour
#     elseif N == 3
#         return Volume
#     else
#         throw(ArgumentError("Plot of $N dimensional level-set not supported."))
#     end
# end

function Makie.convert_arguments(
        P::Union{Type{<:Contour}, Type{<:Contourf}, Type{<:Heatmap}},
        ϕ::LSM.LevelSet,
    )
    ndims(ϕ) == 2 ||
        throw(ArgumentError("Contour plot only supported for 2D level-sets."))
    return Makie.convert_arguments(P, _contour_plot(ϕ)...)
end

function Makie.convert_arguments(P::Type{<:Volume}, ϕ::LSM.LevelSet)
    ndims(ϕ) == 3 ||
        throw(ArgumentError("Volume plot only supported for 3D level-sets."))
    x, y, z, v = _volume_plot(ϕ)
    return Makie.convert_arguments(P, x, y, z, v)
end

function Makie.convert_arguments(
        P::Union{Type{<:Contour}, Type{<:Contourf}, Type{<:Heatmap}},
        nb::LSM.NarrowBandLevelSet,
    )
    ndims(nb) == 2 ||
        throw(ArgumentError("Contour plot only supported for 2D level-sets."))
    msh = LSM.mesh(nb)
    x, y = LSM.xgrid(msh), LSM.ygrid(msh)
    v = _nb_to_dense(nb)
    return Makie.convert_arguments(P, x, y, v)
end

function Makie.convert_arguments(P::Type{<:Volume}, nb::LSM.NarrowBandLevelSet)
    ndims(nb) == 3 ||
        throw(ArgumentError("Volume plot only supported for 3D level-sets."))
    msh = LSM.mesh(nb)
    xlims = extrema(LSM.xgrid(msh))
    ylims = extrema(LSM.ygrid(msh))
    zlims = extrema(LSM.zgrid(msh))
    v = _nb_to_dense(nb)
    return Makie.convert_arguments(P, xlims, ylims, zlims, v)
end

function _contour_plot(ϕ::LSM.LevelSet)
    msh = LSM.mesh(ϕ)
    x, y = LSM.xgrid(msh), LSM.ygrid(msh)
    v = LSM.values(ϕ)
    return x, y, v
end

function _volume_plot(ϕ::LSM.LevelSet)
    msh = LSM.mesh(ϕ)
    x, y, z = LSM.xgrid(msh), LSM.ygrid(msh), LSM.zgrid(msh)
    v = LSM.values(ϕ)
    xlims, ylims, zlims = extrema(x), extrema(y), extrema(z)
    return xlims, ylims, zlims, v
end

# Build a dense array from a NarrowBandLevelSet, with NaN outside the active band.
function _nb_to_dense(nb::LSM.NarrowBandLevelSet{N, T}) where {N, T}
    msh = LSM.mesh(nb)
    arr = fill(T(NaN), size(msh))
    for (I, v) in LSM.values(nb)
        arr[I] = v
    end
    return arr
end

# Collect active cell rectangles for 2D narrow band.
# A cell (lower-corner index I) is active if any of its 4 corners is an active node.
function _active_cell_rects(nb::LSM.NarrowBandLevelSet{2})
    grid = LSM.mesh(nb)
    h = LSM.meshsize(grid)
    cell_axes = LSM.cellindices(grid)
    rects = Rect2f[]
    seen = Set{CartesianIndex{2}}()
    for J in LSM.active_indices(nb)
        for di in 0:1, dj in 0:1
            I = CartesianIndex(J[1] - di, J[2] - dj)
            (I ∈ cell_axes && I ∉ seen) || continue
            push!(seen, I)
            x, y = grid[I]
            push!(rects, Rect2f(x, y, h[1], h[2]))
        end
    end
    return rects
end

# Build grid line segments for a 2D mesh, bounded to the domain.
function _grid_linesegments(msh::LSM.CartesianGrid{2})
    x = LSM.xgrid(msh)
    y = LSM.ygrid(msh)
    segs = Point2f[]
    for xi in x
        push!(segs, Point2f(xi, first(y)))
        push!(segs, Point2f(xi, last(y)))
    end
    for yi in y
        push!(segs, Point2f(first(x), yi))
        push!(segs, Point2f(last(x), yi))
    end
    return segs
end

# Collect active node coordinates as a vector of Points (3D only).
function _active_node_coords(nb::LSM.NarrowBandLevelSet{3})
    msh = LSM.mesh(nb)
    return [Point3f(msh[I]) for I in LSM.active_indices(nb)]
end


Makie.@recipe(LevelSetPlot, eq) do scene
    return Attributes(; showgrid = true)
end

function Makie.plot!(p::LevelSetPlot)
    eq = p.eq
    ϕ = @lift LSM.current_state($eq)
    N = @lift ndims($ϕ)
    is_nb = to_value(ϕ) isa LSM.NarrowBandLevelSet
    if to_value(N) == 2
        if to_value(p.showgrid)
            segs = _grid_linesegments(LSM.mesh(to_value(ϕ)))
            linesegments!(p, segs; color = (:black, 0.15), linewidth = 0.5)
        end
        if is_nb
            rects = @lift _active_cell_rects($ϕ)
            poly!(p, rects; color = (:steelblue, 0.2), strokewidth = 0.5, strokecolor = (:steelblue, 0.5))
        else
            contourf!(p, ϕ; levels = [0], extendlow = (:lightgray, 0.5))
        end
        contour!(p, ϕ; levels = [0], linewidth = 2, color = :black, overdraw = true)
    elseif to_value(N) == 3
        if is_nb
            pts = @lift _active_node_coords($ϕ)
            scatter!(p, pts; color = (:steelblue, 0.3), markersize = 4)
        end
        volume!(p, ϕ; algorithm = :iso, isovalue = 0, alpha = 0.5)
    else
        throw(ArgumentError("Plot of $N dimensional level-set not supported."))
    end
    return p
end

Makie.plottype(::LSM.LevelSetEquation) = LevelSetPlot
Makie.plottype(::LSM.LevelSet) = LevelSetPlot
Makie.plottype(::LSM.NarrowBandLevelSet) = LevelSetPlot

## Pick correct axis type based on dimension of the level-set
function Makie.preferred_axis_type(p::LevelSetPlot)
    eq = p.eq
    ϕ = @lift LSM.current_state($eq)
    dim = @lift ndims($ϕ)
    if to_value(dim) == 2
        return Axis
    elseif to_value(dim) == 3
        return LScene
    else
        throw(ArgumentError("Plot of $dim dimensional level-set not supported."))
    end
end

end # module
