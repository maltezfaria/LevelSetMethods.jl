# Return `ϕ` unchanged if it already carries boundary conditions, otherwise wrap
# it with `LinearExtrapolationBC` on every face (needed for finite-difference stencils).
function _ensure_boundary_conditions(ϕ)
    has_boundary_conditions(ϕ) && return ϕ
    N = ndims(ϕ)
    bc = _normalize_bc(LinearExtrapolationBC(), N)
    return _add_boundary_conditions(ϕ, bc)
end

"""
    volume(ϕ::MeshField)

Compute the volume of the level-set function.

```jldoctest
using LevelSetMethods
R = 0.5
V0 = π * R^2
grid = CartesianGrid((-1, -1), (1, 1), (200, 200))
ϕ = LevelSetMethods.circle(grid; center = (0, 0), radius = R)
LevelSetMethods.volume(ϕ), V0

# output

(0.7854362890190668, 0.7853981633974483)
```
"""
function volume(ϕ::MeshField)
    δ = meshsize(mesh(ϕ))
    δmin = minimum(δ)
    vol = prod(δ)
    sign = smooth_heaviside.(-values(ϕ), δmin)
    return vol * sum(sign .* trapezoidal_coefficients(ϕ))
end

"""
    perimeter(ϕ::MeshField)

Compute the perimeter area of the level-set function.

Note: this function does not compute the perimeter on the borders of the domain.

```jldoctest
using LevelSetMethods
R = 0.5
S0 = 2π * R
grid = CartesianGrid((-1, -1), (1, 1), (200, 200))
ϕ = LevelSetMethods.circle(grid; center = (0, 0), radius = R)
LevelSetMethods.perimeter(ϕ), S0

# output

(3.142641549143036, 3.141592653589793)
```
"""
function perimeter(ϕ::MeshField)
    ϕ = _ensure_boundary_conditions(ϕ)
    δ = meshsize(mesh(ϕ))
    δmin = minimum(δ)
    vol = prod(δ)
    nrm = norm.(gradient(ϕ))
    integrand = smooth_delta.(values(ϕ), δmin) .* nrm
    return vol * sum(integrand .* trapezoidal_coefficients(ϕ))
end

# from "A Variational Level Set Approach to Multiphase Motion"
function smooth_heaviside(x, α)
    if x > α
        return 1.0
    elseif x < -α
        return 0.0
    else
        return 0.5 * (1.0 + x / α + 1.0 / π * sin(π * x / α))
    end
end

function smooth_delta(x, α)
    return abs(x) > α ? 0.0 : 0.5 / α * (1.0 + cos(π * x / α))
end

# Method used to numerically integrate N-th dimensional matrices.
# return a matrix with coefficients equal to 2^(-n) where n is
# the number of times this index if on the borders of a dimension.
function trapezoidal_coefficients(ϕ::MeshField)
    N = ndims(ϕ)
    ax = axes(ϕ)
    coeffs = zeros(size(values(ϕ)))
    for I in CartesianIndices(coeffs)
        n = sum(I[dim] in (first(ax[dim]), last(ax[dim])) for dim in 1:N)
        coeffs[I] = 1.0 / (1 << n)
    end
    return coeffs
end

"""
    curvature(ϕ::MeshField, I)

Compute the mean curvature of ϕ at I using κ = ∇ ⋅ (∇ϕ / |∇ϕ|).
We use the formula κ = (Δϕ |∇ϕ|^2 - ∇ϕ^T Hϕ ∇ϕ) / |∇ϕ|^3 with
first order finite differences.
https://en.wikipedia.org/wiki/Mean_curvature#Implicit_form_of_mean_curvature
"""
function curvature(ϕ, I)
    N = ndims(ϕ)
    if N == 2
        ϕx = D⁰(ϕ, I, 1)
        ϕy = D⁰(ϕ, I, 2)
        ϕxx = D2⁰(ϕ, I, 1)
        ϕyy = D2⁰(ϕ, I, 2)
        ϕxy = D2(ϕ, I, (2, 1))
        κ = (ϕxx * (ϕy)^2 - 2 * ϕy * ϕx * ϕxy + ϕyy * ϕx^2) / (ϕx^2 + ϕy^2)^(3 / 2)
        return κ
    elseif N == 3
        ϕx = D⁰(ϕ, I, 1)
        ϕy = D⁰(ϕ, I, 2)
        ϕz = D⁰(ϕ, I, 3)
        ϕxx = D2⁰(ϕ, I, 1)
        ϕyy = D2⁰(ϕ, I, 2)
        ϕzz = D2⁰(ϕ, I, 3)
        ϕxy = D2(ϕ, I, (2, 1))
        ϕxz = D2(ϕ, I, (3, 1))
        ϕyz = D2(ϕ, I, (3, 2))
        κ = (ϕxx * ϕy^2 + ϕyy * ϕx^2 + ϕxx * ϕz^2 + ϕzz * ϕx^2 + ϕyy * ϕz^2 + ϕzz * ϕy^2 - 2 * ϕx * ϕz * ϕxz - 2 * ϕy * ϕz * ϕyz - 2 * ϕy * ϕx * ϕxy) / (ϕx^2 + ϕy^2 + ϕz^2)^(3 / 2)
        return κ
    else
        # generic method
        ∇ϕ = gradient(ϕ, I)
        nrmsq = dot(∇ϕ, ∇ϕ)
        nrmsq == 0.0 && return 0.0
        Hϕ = hessian(ϕ, I)
        Δϕ = tr(Hϕ)
        κ = (Δϕ * nrmsq - ∇ϕ' * Hϕ * ∇ϕ) / nrmsq^(3 / 2)
        return κ
    end
end

"""
    curvature(ϕ::MeshField)

Compute the mean curvature of ϕ at I using κ = ∇ ⋅ (∇ϕ / |∇ϕ|).
See [`curvature(ϕ::MeshField, I)`](@ref) for more details.

```julia
using LevelSetMethods
N = 50
grid = CartesianGrid((-1, -1), (1, 1), (N, N))
ϕ = LevelSetMethods.star(grid)
using GLMakie
coeff = exp.(-40.0 * values(ϕ) .^ 2)
κ = curvature(ϕ) .* coeff
xs = LevelSetMethods.grid1d(grid, 1)
ys = LevelSetMethods.grid1d(grid, 2)
fig, ax, hm = heatmap(xs, ys, κ)
Colorbar(fig[:, end+1], hm)
contour!(xs, ys, values(ϕ); levels = [0.0])
```
"""
function curvature(ϕ::MeshField)
    _check_bc(ϕ)
    return [curvature(ϕ, I) for I in eachindex(ϕ)]
end

"""
    gradient(ϕ::MeshField, I::CartesianIndex)

Compute the gradient vector ``∇ϕ`` at grid index `I` using centered finite differences.
Returns an `SVector` (or `Vector`) of derivatives.
"""
function gradient(ϕ, I::CartesianIndex)
    N = ndims(ϕ)
    return [D⁰(ϕ, I, dim) for dim in 1:N]
end

"""
    gradient(ϕ::MeshField)

Compute the gradient vector ``∇ϕ`` for all grid points.
"""
function gradient(ϕ::MeshField)
    return [gradient(ϕ, I) for I in eachindex(ϕ)]
end

"""
    normal(ϕ::MeshField, I::CartesianIndex)

Compute the unit exterior normal vector ``\\mathbf{n} = \\frac{∇ϕ}{\\|∇ϕ\\|}`` at grid index `I`.
"""
function normal(ϕ, I)
    ∇ϕ = gradient(ϕ, I)
    return ∇ϕ ./ norm(∇ϕ)
end

"""
    normal(ϕ::MeshField)

Compute the unit exterior normal vector ``\\mathbf{n} = \\frac{∇ϕ}{\\|∇ϕ\\|}`` for all grid points.

```julia
using LevelSetMethods
N = 50
grid = CartesianGrid((-1, -1), (1, 1), (N, N))
ϕ = LevelSetMethods.star(grid)
using GLMakie
n = normal(ϕ)
xs = LevelSetMethods.grid1d(grid, 1)
ys = LevelSetMethods.grid1d(grid, 2)
coeff = exp.(-40.0 * values(ϕ) .^ 2)
us = getindex.(n, 1) .* coeff
vs = getindex.(n, 2) .* coeff
arrows(xs, ys, us, vs; arrowsize = 10 * vec(coeff), lengthscale = 2.0 / (N - 1))
contour!(xs, ys, values(ϕ); levels = [0.0])
```
"""
function normal(ϕ::MeshField)
    _check_bc(ϕ)
    return [normal(ϕ, I) for I in eachindex(ϕ)]
end

"""
    hessian(ϕ::MeshField, I::CartesianIndex)

Compute the Hessian matrix ``\\mathbf{H}ϕ = ∇∇ϕ`` at grid index `I` using second-order
finite differences. Returns a `Symmetric` matrix.
"""
function hessian(ϕ, I::CartesianIndex)
    N = ndims(ϕ)
    return Symmetric([i == j ? D2⁰(ϕ, I, i) : D2(ϕ, I, (i, j)) for i in 1:N, j in 1:N])
end

"""
    hessian(ϕ::MeshField)

Compute the Hessian matrix for all grid points.
"""
function hessian(ϕ::MeshField)
    return [hessian(ϕ, I) for I in eachindex(ϕ)]
end

"""
    grad_norm(ϕ::MeshField)

Compute the norm of the gradient of ϕ, i.e. `|∇ϕ|`, at all grid points.
"""
function grad_norm(ϕ::MeshField)
    _check_bc(ϕ)
    return map(i -> _compute_∇_norm(sign(ϕ[i]), ϕ, i), eachindex(ϕ))
end

#=

Predefined implicit shapes for creating level set functions.

All methods have the signature `f(grid; kwargs...)` where `grid` is the mesh and `kwargs`
are optional keyword arguments (e.g. the `center` or `radius` of a circle).

=#

"""
    circle(grid; center = (0, 0), radius = 1, kwargs...)

Create a 2D circle with the specified `center` and `radius` on a `grid`.
Returns a [`MeshField`](@ref) field.
"""
function circle(grid; center = (0, 0), radius = 1, kwargs...)
    ndims(grid) == 2 ||
        throw(ArgumentError("circle shape is only available in two dimensions"))
    return MeshField(x -> sqrt(sum((x .- center) .^ 2)) - radius, grid; kwargs...)
end

"""
    rectangle(grid; center = (0, 0), width = (1, 1), kwargs...)

Create a rectangle (or N-dimensional box) with the specified `center` and `width` on a `grid`.
Returns a [`MeshField`](@ref) field.
"""
function rectangle(grid; center = zero(grid.lc), width = (1, 1), kwargs...)
    return MeshField(x -> maximum(abs.(x .- center) .- width ./ 2), grid; kwargs...)
end

"""
    sphere(grid; center = (0, 0, 0), radius, kwargs...)

Create a 3D sphere with the specified `center` and `radius` on a `grid`.
Returns a [`MeshField`](@ref) field.
"""
function sphere(grid; center = (0, 0, 0), radius, kwargs...)
    ndims(grid) == 3 ||
        throw(ArgumentError("sphere shape is only available in three dimensions"))
    return MeshField(x -> sqrt(sum((x .- center) .^ 2)) - radius, grid; kwargs...)
end

"""
    star(grid; radius = 1, deformation = 0.25, n = 5.0, kwargs...)

Create a 2D star shape defined in polar coordinates by ``r = R(1 + d \\cos(nθ))``.
Returns a [`MeshField`](@ref) field.
"""
function star(grid; radius = 1, deformation = 0.25, n = 5.0, kwargs...)
    # ndims(grid) == 2 ||
    #     throw(ArgumentError("star shape is only available in two dimensions"))
    return MeshField(grid; kwargs...) do x
        r = norm(x)
        θ = atan(x[2], x[1])
        return r - radius * (1.0 + deformation * cos(n * θ))
    end
end

"""
    dumbbell(grid; width = 1, height = 0.2, radius = 0.25, center = (0, 0), kwargs...)

Create a 2D dumbbell shape consisting of two circles connected by a rectangle.
Returns a [`MeshField`](@ref) field.
"""
function dumbbell(grid; width = 1, height = 1 / 5, radius = 1 / 4, center = (0, 0), kwargs...)
    cl = circle(grid; center = center .- (width / 2, 0), radius, kwargs...)
    cr = circle(grid; center = center .+ (width / 2, 0), radius, kwargs...)
    rec = rectangle(grid; center, width = (width, height), kwargs...)
    return cl ∪ cr ∪ rec
end

"""
    zalesak_disk(grid; center = (0, 0), radius = 0.5, width = 0.25, height = 1, kwargs...)

Create a Zalesak disk (a circle with a rectangular slot cut out).
Used for testing advection schemes. Returns a [`MeshField`](@ref) field.
"""
function zalesak_disk(grid; center = (0, 0), radius = 0.5, width = 0.25, height = 1, kwargs...)
    ndims(grid) == 2 ||
        throw(ArgumentError("zalesak disk shape is only available in two dimensions"))
    disk = circle(grid; center = center, radius = radius, kwargs...)
    rec = rectangle(grid; center = center .- (0, radius), width = (width, height), kwargs...)
    return setdiff(disk, rec)
end

#=

Set operations for level set functions.

=#

"""
    union!(ϕ1::MeshField, ϕ2::MeshField)

In-place union of two level sets: ``ϕ_1 = \\min(ϕ_1, ϕ_2)``.
"""
function Base.union!(ϕ1::MeshField, ϕ2::MeshField)
    v1, v2 = values(ϕ1), values(ϕ2)
    v1 .= min.(v1, v2)
    return ϕ1
end

"""
    union(ϕ1::MeshField, ϕ2::MeshField)

Return the union of two level sets: ``\\min(ϕ_1, ϕ_2)``.
"""
Base.union(ϕ1::MeshField, ϕ2::MeshField) = union!(deepcopy(ϕ1), ϕ2)

"""
    intersect!(ϕ1::MeshField, ϕ2::MeshField)

In-place intersection of two level sets: ``ϕ_1 = \\max(ϕ_1, ϕ_2)``.
"""
function Base.intersect!(ϕ1::MeshField, ϕ2::MeshField)
    v1, v2 = values(ϕ1), values(ϕ2)
    v1 .= max.(v1, v2)
    return ϕ1
end

"""
    intersect(ϕ1::MeshField, ϕ2::MeshField)

Return the intersection of two level sets: ``\\max(ϕ_1, ϕ_2)``.
"""
Base.intersect(ϕ1::MeshField, ϕ2::MeshField) = intersect!(deepcopy(ϕ1), ϕ2)

"""
    complement!(ϕ::MeshField)

In-place complement of a level set (negates the values).
"""
function complement!(ϕ::MeshField)
    v = values(ϕ)
    v .= -v
    return ϕ
end

"""
    complement(ϕ::MeshField)

Return the complement of a level set (negates the values).
"""
complement(ϕ::MeshField) = complement!(deepcopy(ϕ))

"""
    setdiff!(ϕ1::MeshField, ϕ2::MeshField)

In-place set difference: ``ϕ_1 = \\max(ϕ_1, -ϕ_2)``.
"""
function Base.setdiff!(ϕ1::MeshField, ϕ2::MeshField)
    v1, v2 = values(ϕ1), values(ϕ2)
    v1 .= max.(v1, -v2)
    return ϕ1
end

"""
    setdiff(ϕ1::MeshField, ϕ2::MeshField)

Return the set difference: ``\\max(ϕ_1, -ϕ_2)``.
"""
Base.setdiff(ϕ1::MeshField, ϕ2::MeshField) = setdiff!(deepcopy(ϕ1), ϕ2)
