module MMGSurfaceExt

import MMG_jll as MMG
import MarchingCubes
import LevelSetMethods as LSM

function __init__()
    @info "Loading MMGSurface extension for LevelSetMethods.jl"
end

"""
    export_surface_mesh(eq::LevelSetEquation, args...; kwargs...)

Call [`export_surface_mesh`](@ref LSM.export_surface_mesh(::LSM.LevelSet)) on
`current_state(eq)`.
"""
function LSM.export_surface_mesh(eq::LSM.LevelSetEquation, args...; kwargs...)
    return LSM.export_surface_mesh(LSM.current_state(eq), output; hgrad, hmin, hmax, hausd)
end

"""
    export_surface_mesh(ϕ::LevelSet, output::String;
        hgrad = nothing, hmin = nothing, hmax = nothing, hausd = nothing)

Compute a mesh of the [`LevelSet`](@ref LSM.LevelSet) `ϕ` zero contour using MMGs_O3.

`hgrad` control the growth ratio between two adjacent edges

`hmin` and `hmax` control the edge sizes to be (respectively) greater than the `hmin`
parameter and lower than the `hmax` one

`hausd` control the maximal distance between the piecewise linear representation of the
boundary and the reconstructed ideal boundary

!!! note

    Only works for 3 dimensional level-set.
"""
function LSM.export_surface_mesh(
    ϕ::LSM.LevelSet,
    output::String;
    hgrad = nothing,
    hmin = nothing,
    hmax = nothing,
    hausd = nothing,
)
    N = LSM.dimension(ϕ)
    if N != 3
        throw(ArgumentError("export_mesh of $N dimensional level-set not supported."))
    end

    mc = MarchingCubes.MC(LSM.values(ϕ))
    MarchingCubes.march(mc)

    temp_mesh_path = tempname() * ".mesh"

    try
        _write_3D_triangular_mesh(temp_mesh_path, mc.vertices, mc.triangles)

        command = MMG.mmgs_O3()
        arguments = [
            "-in",
            temp_mesh_path,
            "-out",
            output,
            "-nr", # no ridge detection
        ]
        for (name, value) in
            [("hgrad", hgrad), ("hmin", hmin), ("hmax", hmax), ("hausd", hausd)]
            if value !== nothing
                push!(arguments, '-' * name)
                push!(arguments, string(value))
            end
        end
        println(arguments)
        run(`$(command) $arguments`)

    finally
        if isfile(temp_mesh_path)
            rm(temp_mesh_path)
        end
    end
    return output
end

function _write_3D_triangular_mesh(path, vertices, triangles)
    open(path, "w") do file
        write(file, "MeshVersionFormatted 1\n")
        write(file, "Dimension 3\n")

        write(file, "\nVertices\n")
        write(file, "$(length(vertices))\n")

        for (x, y, z) in vertices
            write(file, join((x, y, z, 1), ' ') * '\n')
        end

        write(file, "\nTriangles\n")
        write(file, "$(length(triangles))\n")
        for (id1, id2, id3) in triangles
            write(file, join((id1, id2, id3, 1), ' ') * '\n')
        end

        return write(file, "\nEnd\n")
    end
end

end
