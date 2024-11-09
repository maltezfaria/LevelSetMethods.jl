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

function Makie.plottype(ϕ::LSM.LevelSet)
    N = LSM.dimension(ϕ)
    if N == 2
        return Contour
    elseif N == 3
        return Volume
    else
        throw(ArgumentError("Plot of $N dimensional level-set not supported."))
    end
end

function Makie.convert_arguments(::Type{<:AbstractPlot}, ϕ::LSM.LevelSet)
    N = LSM.dimension(ϕ)
    if N == 2
        _contour_plot(ϕ)
    elseif N == 3
        _volume_plot(ϕ)
    else
        throw(ArgumentError("Plot of $N dimensional level-set not supported."))
    end
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
    if N == 2
        plot!(p, ϕ; levels = [0], linewidth = 2, color = :black)
    elseif N == 3
        plot!(p, ϕ; algorithm = :iso, isolevel = 0, alpha = 0.5)
    else
        throw(ArgumentError("Plot of $N dimensional level-set not supported."))
    end
    return p
end

Makie.plottype(::LSM.LevelSetEquation) = LevelSetPlot

## Pick correct axis type based on dimension of the level-set
function Makie.args_preferred_axis(p::LevelSetPlot)
    eq = p.eq
    ϕ = @lift LSM.current_state($eq)
    dim = @lift LSM.dimension($ϕ)
    if dim == 2
        return Axis
    elseif dim == 3
        return LScene
    else
        throw(ArgumentError("Plot of $dim dimensional level-set not supported."))
    end
end

end # module
