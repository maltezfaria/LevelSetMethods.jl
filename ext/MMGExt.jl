module MMGExt

import MMG_jll as MMG
import LevelSetMethods as LSM

function __init__()
    @info "Loading MMG extension for LevelSetMethods.jl"
end

function LSM.export_mesh(eq::LSM.LevelSetEquation)
    return nothing
end

end
