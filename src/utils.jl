# helps to obtain classical shapes's signed distance function
function CircleSignedDistance(m, center, r)
    return map(x -> sqrt(sum((x .- center) .^ 2)) - r, m)
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

function sphere(
    grid;
    center = (grid.hc + grid.lc) / 2.0,
    radius = minimum(grid.hc - grid.lc) / 4.0,
)
    ϕ = LevelSet(x -> 1.0, grid)
    add_circle!(ϕ, center, radius)
    return ϕ
end

function star(grid; radius = minimum(grid.hc - grid.lc) / 4.0, deformation = 0.25, n = 5.0)
    N = dimension(grid)
    if N != 2
        throw(ArgumentError("star shape is only available in two dimensions"))
    end
    return LevelSet(grid) do (x, y)
        norm = sqrt(x^2 + y^2)
        θ = atan(y, x)
        return norm - radius * (1.0 + deformation * cos(n * θ))
    end
end

# helpers to merge or make the difference between two level set functions
@inline function Base.union!(ϕ1, ϕ2)
    @. ϕ1 = min(ϕ1, ϕ2)
end
@inline function difference!(ϕ1, ϕ2)
    @. ϕ1 = -min(-ϕ1, ϕ2)
end
