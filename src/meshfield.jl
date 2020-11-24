"""
    struct MeshField{V,M}

A field described by its discrete values on a mesh.
"""
struct MeshField{V,M}
    vals::V
    mesh::M
end

# getters
mesh(ϕ::MeshField) = ϕ.mesh
Base.values(ϕ::MeshField) = ϕ.vals
grid1d(mf::MeshField,args...) = grid1d(mesh(mf),args...)

function MeshField(f::Function,m)
    vals = map(f,m)
    MeshField(vals,m)
end

# geometric dimension
dimension(f::MeshField) = dimension(mesh(f))

meshsize(f::MeshField,args...) = meshsize(mesh(f),args...)

# overload base methods for convenience
Base.getindex(ϕ::MeshField,I...) = getindex(values(ϕ),I...)
Base.setindex!(ϕ::MeshField,vals,I...) = setindex!(values(ϕ),vals,I...)
Base.size(ϕ::MeshField) = size(values(ϕ))
Base.eltype(ϕ::MeshField) = eltype(values(ϕ))
Base.zero(ϕ::MeshField) = MeshField(zero(values(ϕ)),mesh(ϕ))

"""
    LevelSet

Alias for [`MeshField`](@ref).
"""
const LevelSet{V,M} = MeshField{V,M}

# function Contour.contour(ϕ::LevelSet)
#     N = dimension(ϕ)
#     if N == 2
#         x = xgrid(ϕ.mesh)
#         y = ygrid(ϕ.mesh)
#         c = Contour.contour(x,y,transpose(values(ϕ)),0)
#         pts = SVector{2,Float64}[]
#         for l in lines(c)
#             xs,ys = coordinates(l)
#             for i in 1:length(xs)
#                 push!(pts,SVector(xs[i],ys[i]))
#             end
#         end
#         return pts
#     else
#         notimplemented()
#     end
# end

# helpers to add geometric shapes on the a level set
function add_circle!(ϕ::MeshField, center, r)
    rsq = r*r
    circle = map(x -> sum((x.-center).^2) - r*r, mesh(ϕ))
    @. ϕ.vals = min(ϕ.vals, circle)
end

function add_rectangle!(ϕ::MeshField, center, size)
    sized2 = 0.5*size
    rectangle = map(x -> maximum(abs.(x.-center) - sized2), mesh(ϕ))
    @. ϕ.vals = min.(ϕ.vals, rectangle)
end

# recipes for Plots
@recipe function f(ϕ::MeshField)
    N = dimension(ϕ)
    if N == 2 # 2d contour plot
        seriestype := :contour
        levels --> [0,]
        aspect_ratio --> :equal
        colorbar --> false
        seriescolor --> :black
        m = mesh(ϕ)
        # Note: the values of ϕ need be transposed because contour expects the
        # matrix to have rows representing the x values and columns expecting
        # the y value.
        return xgrid(m),ygrid(m),transpose(values(ϕ))
    else
        notimplemented()
    end
end
