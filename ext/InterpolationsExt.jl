module InterpolationsExt

import Interpolations as Itp
import LevelSetMethods as LSM

function __init__()
    return @info "Loading Interpolations extension for LevelSetMethods.jl"
end

Itp.interpolate(ϕ::LSM.LevelSet) = Itp.interpolate(ϕ, Itp.BSpline(Itp.Cubic()), Itp.Line())

function Itp.interpolate(ϕ::LSM.LevelSet, mode, bc)
    itp = Itp.interpolate(ϕ.vals, mode)
    grids = LSM.grid1d(LSM.mesh(ϕ))
    sc_itp = Itp.scale(itp, grids...)
    ext_itp = Itp.extrapolate(sc_itp, bc)
    return ext_itp
end

function Itp.interpolate(eq::LSM.LevelSetEquation, args...; kwargs...)
    return Itp.interpolate(LSM.current_state(eq), args...; kwargs...)
end

end # module
