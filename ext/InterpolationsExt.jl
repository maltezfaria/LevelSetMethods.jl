module InterpolationsExt

import Interpolations as Itp
import LevelSetMethods as LSM

function __init__()
    return @info "Loading Interpolations extension for LevelSetMethods.jl"
end

function Itp.interpolate(ϕ::LSM.LevelSet)
    bc = Itp.Free(Itp.OnGrid())
    mode = Itp.BSpline(Itp.Cubic(bc))
    return Itp.interpolate(ϕ, mode)
end

function Itp.interpolate(ϕ::LSM.LevelSet, mode)
    itp = Itp.interpolate(ϕ.vals, mode)
    grids = LSM.grid1d(LSM.mesh(ϕ))
    sc_itp = Itp.scale(itp, grids...)
    return sc_itp
end

function Itp.interpolate(eq::LSM.LevelSetEquation, args...; kwargs...)
    return Itp.interpolate(LSM.current_state(eq), args...; kwargs...)
end

end # module
