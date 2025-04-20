module InterpolationsExt

import Interpolations as Itp
import LevelSetMethods as LSM

function __init__()
    @info "Loading Interpolations extension for LevelSetMethods.jl"
end

Itp.interpolate(ϕ::LSM.LevelSet) = Itp.interpolate(ϕ, Itp.BSpline(Itp.Cubic()))

function Itp.interpolate(ϕ::LSM.LevelSet, mode)
    itp = Itp.interpolate(ϕ.vals, mode)
    grids = LSM.grid1d(LSM.mesh(ϕ))
    return Itp.scale(itp, grids...)
end

function Itp.interpolate(eq::LSM.LevelSetEquation, args...; kwargs...)
    return Itp.interpolate(LSM.current_state(eq), args...; kwargs...)
end

end # module
