module MMGVolumeExt

import MMG_jll as MMG
import LevelSetMethods as LSM
# using DelimitedFiles # using or import ? # faster with writedlm ?

function __init__()
    @info "Loading MMGVolume extension for LevelSetMethods.jl"
end

"""
    export_volume_mesh(eq::LevelSetEquation, output; kwargs...)

Call [`export_volume_mesh`](@ref LSM.export_volume_mesh(::LSM.LevelSet)) on `current_state(eq)`.
"""
function LSM.export_volume_mesh(eq::LSM.LevelSetEquation, args...; kwargs...)
    return LSM.export_volume_mesh(LSM.current_state(eq), args...; kwargs...)
end

"""
    export_volume_mesh(ϕ::LevelSet, output::String;
        hgrad = nothing, hmin = nothing, hmax = nothing, hausd = nothing)

Compute a mesh of the domains associated with [`LevelSet`](@ref LSM.LevelSet) `eq` using
either MMG2d_O3 or MMG3d_O3.

`hgrad` control the growth ratio between two adjacent edges.

`hmin` and `hmax` control the edge sizes to be (respectively) greater
than the `hmin` parameter and lower than the `hmax` one.

`hausd` control the maximal distance between the piecewise linear
representation of the boundary and the reconstructed ideal boundary.

For more information, see the official [MMG documentation](http://www.mmgtools.org).

!!! note

    Only works for 2 and 3 dimensional level-set.
"""
function LSM.export_volume_mesh(
    ϕ::LSM.LevelSet,
    output::String;
    hgrad = nothing,
    hmin = nothing,
    hmax = nothing,
    hausd = nothing,
)
    N = LSM.dimension(ϕ)
    if N != 2 && N != 3
        throw(ArgumentError("export_mesh of $N dimensional level-set not supported."))
    end

    temp_path = tempname()
    temp_mesh_path = temp_path * ".mesh"
    temp_sol_path = temp_path * ".sol"

    try
        num_vertices = length(LSM.values(ϕ))

        open(temp_mesh_path, "w") do file
            write(file, "MeshVersionFormatted 1\n")
            write(file, "Dimension $N\n")

            write(file, "\nVertices\n")
            write(file, "$num_vertices\n")

            msh = LSM.mesh(ϕ)
            x, y = LSM.xgrid(msh), LSM.ygrid(msh)
            nx = length(x)
            ny = length(y)

            if N == 2
                for x_val in x, y_val in y
                    write(file, join((x_val, y_val, 1), ' ') * '\n') # x y ref
                end

                num_triangles = 2(nx - 1) * (ny - 1)
                write(file, "\nTriangles\n")
                write(file, "$num_triangles\n")
                for x_id in 1:nx-1, y_id in 1:ny-1
                    c00 = (y_id - 1) * nx + x_id
                    c10 = c00 + 1
                    c01 = c00 + nx
                    c11 = c01 + 1
                    write(file, join((c00, c10, c01, 1), ' ') * '\n') # id1 id2 id3 ref
                    write(file, join((c11, c10, c01, 1), ' ') * '\n')
                end
            elseif N == 3
                z = LSM.zgrid(msh)
                nz = length(z)
                for x_val in x, y_val in y, z_val in z
                    write(file, join((x_val, y_val, z_val, 1), ' ') * '\n') # x y z ref
                end

                num_tetrahedrons = 6(nx - 1) * (ny - 1) * (nz - 1)
                write(file, "\nTetrahedra\n")
                write(file, "$num_tetrahedrons\n")
                for x_id in 1:nx-1, y_id in 1:ny-1, z_id in 1:nz-1
                    c000 = (z_id - 1) * nx * ny + (y_id - 1) * nx + x_id
                    c100 = c000 + 1
                    c010 = c000 + nx
                    c110 = c010 + 1
                    c001 = c000 + nx * ny
                    c101 = c001 + 1
                    c011 = c001 + nx
                    c111 = c011 + 1
                    write(file, join((c000, c010, c011, c111, 1), ' ') * '\n') # id1 id2 id3 id4 ref
                    write(file, join((c000, c011, c001, c111, 1), ' ') * '\n')
                    write(file, join((c000, c001, c101, c111, 1), ' ') * '\n')
                    write(file, join((c000, c101, c100, c111, 1), ' ') * '\n')
                    write(file, join((c000, c100, c110, c111, 1), ' ') * '\n')
                    write(file, join((c000, c110, c010, c111, 1), ' ') * '\n')
                end
            end

            return write(file, "\nEnd\n")
        end

        open(temp_sol_path, "w") do file
            write(file, "MeshVersionFormatted 1\n")
            write(file, "Dimension $N\n")

            write(file, "\nSolAtVertices\n")
            write(file, "$num_vertices\n")
            write(file, "1 1\n\n") # 1 = #sol per node ; 1 = scalar solution

            for val in LSM.values(ϕ)[:]
                write(file, string(val) * '\n')
            end

            return write(file, "\nEnd\n")
        end

        command = N == 2 ? MMG.mmg2d_O3() : MMG.mmg3d_O3()
        arguments = [
            "-in",
            temp_mesh_path,
            "-out",
            output,
            "-ls", # levelset implicit domain meshing
            "-sol",
            temp_sol_path,
        ]
        for (name, value) in
            [("hgrad", hgrad), ("hmin", hmin), ("hmax", hmax), ("hausd", hausd)]
            if value !== nothing
                push!(arguments, name)
                push!(arguments, string(value))
            end
        end
        println(arguments)
        run(`$(command) $arguments`)

    finally
        if isfile(temp_mesh_path)
            rm(temp_mesh_path)
        end
        if isfile(temp_sol_path)
            rm(temp_sol_path)
        end
    end

    return output
end

end
