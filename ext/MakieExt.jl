module MakieExt

using Makie
import LevelSetMethods as LSM

function __init__()
    @info "Loading Makie extension for LevelSetMethods.jl"
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
        ),
        fontsize = 20,
    )
end

function LSM.set_makie_theme!()
    return Makie.set_theme!(LSM.makie_theme())
end

# function Makie.plottype(ϕ::LSM.LevelSet)
#     N = LSM.dimension(ϕ)
#     if N == 2
#         return Contour
#     elseif N == 3
#         return Volume
#     else
#         throw(ArgumentError("Plot of $N dimensional level-set not supported."))
#     end
# end

function Makie.convert_arguments(::Union{Type{<:Contour},Type{<:Contourf}}, ϕ::LSM.LevelSet)
    LSM.dimension(ϕ) == 2 ||
        throw(ArgumentError("Contour plot only supported for 2D level-sets."))
    return _contour_plot(ϕ)
end

function Makie.convert_arguments(::Type{<:Volume}, ϕ::LSM.LevelSet)
    LSM.dimension(ϕ) == 3 ||
        throw(ArgumentError("Volume plot only supported for 3D level-sets."))
    return _volume_plot(ϕ)
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
    # NOTE: volume gives a warning when passed an AbstractVector for x,y,z,
    # and asks for a tuple instead
    return extrema(x), extrema(y), extrema(z), v
end

Makie.@recipe(LevelSetPlot, eq) do scene
    return Theme()
end

function Makie.plot!(p::LevelSetPlot)
    eq = p.eq
    ϕ = @lift LSM.current_state($eq)
    N = @lift LSM.dimension($ϕ)
    if to_value(N) == 2
        contourf!(p, ϕ; levels = [0], extendlow = (:lightgray, 0.5))
        contour!(p, ϕ; levels = [0], linewidth = 2, color = :black)
    elseif to_value(N) == 3
        volume!(p, ϕ; algorithm = :iso, isovalue = 0, alpha = 0.5)
    else
        throw(ArgumentError("Plot of $N dimensional level-set not supported."))
    end
    return p
end

Makie.plottype(::LSM.LevelSetEquation) = LevelSetPlot
Makie.plottype(::LSM.LevelSet) = LevelSetPlot

## Pick correct axis type based on dimension of the level-set
function Makie.preferred_axis_type(p::LevelSetPlot)
    eq = p.eq
    ϕ = @lift LSM.current_state($eq)
    dim = @lift LSM.dimension($ϕ)
    if to_value(dim) == 2
        return Axis
    elseif to_value(dim) == 3
        @info "here"
        return LScene
    else
        throw(ArgumentError("Plot of $dim dimensional level-set not supported."))
    end
end

end # module
