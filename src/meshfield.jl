"""
    struct MeshField{V,M}

A field described by its discrete values on a mesh.
"""
struct MeshField{V,M,B}
    vals::V
    mesh::M
    bc::B
end

# getters
mesh(ϕ::MeshField) = ϕ.mesh
Base.values(ϕ::MeshField) = ϕ.vals
grid1d(mf::MeshField, args...) = grid1d(mesh(mf), args...)

has_boundary_condition(mf::MeshField) = (mf.bc !== nothing)
boundary_condition(mf) = mf.bc

function MeshField(f::Function, m)
    vals = map(f, m)
    return MeshField(vals, m, nothing)
end

# geometric dimension
dimension(f::MeshField) = dimension(mesh(f))

meshsize(f::MeshField, args...) = meshsize(mesh(f), args...)

# overload base methods for convenience
Base.getindex(ϕ::MeshField, I...) = getindex(values(ϕ), I...)
Base.setindex!(ϕ::MeshField, vals, I...) = setindex!(values(ϕ), vals, I...)
Base.size(ϕ::MeshField) = size(values(ϕ))
Base.eltype(ϕ::MeshField) = eltype(values(ϕ))
Base.zero(ϕ::MeshField) = MeshField(zero(values(ϕ)), mesh(ϕ), boundary_condition(ϕ))
Base.similar(ϕ::MeshField) = MeshField(similar(values(ϕ)), mesh(ϕ), boundary_condition(ϕ))

"""
    LevelSet

Alias for [`MeshField`](@ref) with a boundary condition.
"""
const LevelSet{V,M,B<:BoundaryCondition} = MeshField{V,M,B}

function LevelSet(f::Function, m, bc::BoundaryCondition = PeriodicBC(0))
    vals = map(f, m)
    ϕ = MeshField(vals, m, bc)
    applybc!(ϕ)
    return ϕ
end

applybc!(ϕ::LevelSet) = applybc!(ϕ, boundary_condition(ϕ))

interior_indices(ϕ::LevelSet) = interior_indices(mesh(ϕ), boundary_condition(ϕ))

# helps to obtain classical shapes's signed distance function
function CircleSignedDistance(m, center, r)
    rsq = r * r
    return map(x -> sum((x .- center) .^ 2) - rsq, m)
end
function RectangleSignedDistance(m, center, size)
    sized2 = 0.5 * size
    return map(x -> maximum(abs.(x .- center) - sized2), m)
end

# helpers to add geometric shapes on the a level set
function add_circle!(ϕ::MeshField, center, r)
    circle = CircleSignedDistance(mesh(ϕ), center, r)
    return union!(values(ϕ), circle)
end
function remove_circle!(ϕ::MeshField, center, r)
    circle = CircleSignedDistance(mesh(ϕ), center, r)
    return difference!(values(ϕ), circle)
end

function add_rectangle!(ϕ::MeshField, center, size)
    rectangle = RectangleSignedDistance(mesh(ϕ), center, size)
    return union!(values(ϕ), rectangle)
end
function remove_rectangle!(ϕ::MeshField, center, size)
    rectangle = RectangleSignedDistance(mesh(ϕ), center, size)
    return difference!(values(ϕ), rectangle)
end

# helpers to merge or make the difference between two level set functions
@inline function Base.union!(ϕ1, ϕ2)
    @. ϕ1 = min(ϕ1, ϕ2)
end
@inline function difference!(ϕ1, ϕ2)
    @. ϕ1 = -min(-ϕ1, ϕ2)
end
