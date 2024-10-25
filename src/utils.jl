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
function add_circle!(ϕ, center, r)
    circle = CircleSignedDistance(mesh(ϕ), center, r)
    return union!(values(ϕ), circle)
end
function remove_circle!(ϕ, center, r)
    circle = CircleSignedDistance(mesh(ϕ), center, r)
    return difference!(values(ϕ), circle)
end

function add_rectangle!(ϕ, center, size)
    rectangle = RectangleSignedDistance(mesh(ϕ), center, size)
    return union!(values(ϕ), rectangle)
end
function remove_rectangle!(ϕ, center, size)
    rectangle = RectangleSignedDistance(mesh(ϕ), center, size)
    return difference!(values(ϕ), rectangle)
end

function dumbbell(grid; width = 1, height = 1 / 5, radius = 1 / 4)
    ϕ = LevelSet(x -> 1.0, grid)
    add_circle!(ϕ, SVector(-width / 2, 0.0), radius)
    add_circle!(ϕ, SVector(width / 2, 0.0), radius)
    add_rectangle!(ϕ, SVector(0.0, 0.0), SVector(width, height))
    return ϕ
end

# helpers to merge or make the difference between two level set functions
@inline function Base.union!(ϕ1, ϕ2)
    @. ϕ1 = min(ϕ1, ϕ2)
end
@inline function difference!(ϕ1, ϕ2)
    @. ϕ1 = -min(-ϕ1, ϕ2)
end
