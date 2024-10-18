module MakieExt

using Makie
import LevelSetMethods as LSM

function __init__()
    @info "Loading Makie extension for LevelSetMethods.jl"
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

function Makie.convert_arguments(P::Type{<:AbstractPlot}, ϕ::LSM.LevelSet)
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

end # module
