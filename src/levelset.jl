"""
    LevelSet

Alias for [`MeshField`](@ref) with `vals` as an `AbstractArray` of `Real`s.
"""
const LevelSet{V<:AbstractArray{<:Real},M,B} = MeshField{V,M,B}

function LevelSet(f::Function, m)
    vals = map(f, m)
    return MeshField(vals, m, nothing)
end

"""
    volume(ϕ::LevelSet)

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
function volume(ϕ::LevelSet)
    δ = meshsize(mesh(ϕ))
    δmin = minimum(δ)
    vol = prod(δ)
    sign = smooth_heaviside.(-values(ϕ), δmin)
    return vol * sum(sign .* trapezoidal_coefficients(ϕ))
end

"""
    perimeter(ϕ::LevelSet)

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

(3.1426415491430366, 3.141592653589793)
```
"""
function perimeter(ϕ::LevelSet)
    # if no boundary conditions then we use homogenous Neumann
    if !has_boundary_conditions(ϕ)
        N = dimension(ϕ)
        bc = _normalize_bc(NeumannGradientBC(), N)
        ϕ = add_boundary_conditions(ϕ, bc)
    end
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
function trapezoidal_coefficients(ϕ::LevelSet)
    N = dimension(ϕ)
    ax = axes(ϕ)
    coeffs = zeros(size(values(ϕ)))
    for I in CartesianIndices(coeffs)
        n = sum(I[dim] in (first(ax[dim]), last(ax[dim])) for dim in 1:N)
        coeffs[I] = 1.0 / (1 << n)
    end
    return coeffs
end

"""
    curvature(ϕ::LevelSet, I)

Compute the mean curvature of ϕ at I using κ = ∇ ⋅ (∇ϕ / |∇ϕ|).
We use the formula κ = (Δϕ |∇ϕ|^2 - ∇ϕ^T Hϕ ∇ϕ) / |∇ϕ|^3 with
first order finite differences.
https://en.wikipedia.org/wiki/Mean_curvature#Implicit_form_of_mean_curvature
"""
function curvature(ϕ::LevelSet, I)
    N = dimension(ϕ)
    if N == 2
        ϕx  = D⁰(ϕ, I, 1)
        ϕy  = D⁰(ϕ, I, 2)
        ϕxx = D2⁰(ϕ, I, 1)
        ϕyy = D2⁰(ϕ, I, 2)
        ϕxy = D2(ϕ, I, (2, 1))
        κ   = (ϕxx * (ϕy)^2 - 2 * ϕy * ϕx * ϕxy + ϕyy * ϕx^2) / (ϕx^2 + ϕy^2)^(3 / 2)
        return κ
    elseif N == 3
        ϕx  = D⁰(ϕ, I, 1)
        ϕy  = D⁰(ϕ, I, 2)
        ϕz  = D⁰(ϕ, I, 3)
        ϕxx = D2⁰(ϕ, I, 1)
        ϕyy = D2⁰(ϕ, I, 2)
        ϕzz = D2⁰(ϕ, I, 3)
        ϕxy = D2(ϕ, I, (2, 1))
        ϕxz = D2(ϕ, I, (3, 1))
        κ   = (ϕxx * (ϕy)^2 - 2 * ϕy * ϕx * ϕxy + ϕyy * ϕx^2 + ϕx^2 * ϕzz - 2 * ϕx * ϕz * ϕxz + ϕz^2 * ϕxx + ϕy^2 * ϕzz - 2 * ϕy * ϕz * ϕyz + ϕz^2 * ϕyy) / (ϕx^2 + ϕy^2)^3 / 2
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
    curvature(ϕ::LevelSet)

Compute the mean curvature of ϕ at I using κ = ∇ ⋅ (∇ϕ / |∇ϕ|).
See [`curvature(ϕ::LevelSet, I)`](@ref) for more details.

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
function curvature(ϕ::LevelSet)
    if !has_boundary_conditions(ϕ)
        N = dimension(ϕ)
        bc = _normalize_bc(NeumannGradientBC(), N)
        ϕ = add_boundary_conditions(ϕ, bc)
    end
    return [curvature(ϕ, I) for I in eachindex(ϕ)]
end

"""
    gradient(ϕ::LevelSet, I)

Return the gradient vector ∇ϕ of ϕ at I
"""
function gradient(ϕ::LevelSet, I)
    N = dimension(ϕ)
    return [D⁰(ϕ, I, dim) for dim in 1:N]
end

function gradient(ϕ::LevelSet)
    return [gradient(ϕ, I) for I in eachindex(ϕ)]
end

"""
    normal(ϕ::LevelSet, I)

Compute the unit exterior normal vector of ϕ at I using n = ∇ϕ/|∇ϕ|
"""
function normal(ϕ::LevelSet, I)
    ∇ϕ = gradient(ϕ, I)
    return ∇ϕ ./ norm(∇ϕ)
end

"""
    normal(ϕ::LevelSet)

Compute the unit exterior normal vector of ϕ using n = ∇ϕ/|∇ϕ|

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
function normal(ϕ::LevelSet)
    # if no boundary conditions then we use homogenous Neumann
    if !has_boundary_conditions(ϕ)
        N = dimension(ϕ)
        bc = _normalize_bc(NeumannGradientBC(), N)
        ϕ = add_boundary_conditions(ϕ, bc)
    end
    return [normal(ϕ, I) for I in eachindex(ϕ)]
end

"""
    hessian(ϕ::LevelSet, I)

Return the Hessian matrix Hϕ of ϕ at I
"""
function hessian(ϕ::LevelSet, I)
    N = dimension(ϕ)
    return Symmetric([i == j ? D2⁰(ϕ, I, i) : D2(ϕ, I, (i, j)) for i in 1:N, j in 1:N])
end

function hessian(ϕ::LevelSet)
    return [hessian(ϕ, I) for I in eachindex(ϕ)]
end

#=

Predefined implicit shapes for creating level set functions.

All methods have the signature `f(grid; kwargs...)` where `grid` is the mesh and `kwargs`
are optional keyword arguments (e.g. the `center` or `radius` of a circle).

=#

function circle(grid; center = (0, 0), radius = 1)
    dimension(grid) == 2 ||
        throw(ArgumentError("circle shape is only available in two dimensions"))
    return LevelSet(x -> sqrt(sum((x .- center) .^ 2)) - radius, grid)
end

function rectangle(grid; center = zero(0, 0), width = (1, 1))
    return LevelSet(x -> maximum(abs.(x .- center) .- width ./ 2), grid)
end

function sphere(grid; center = (0, 0, 0), radius)
    dimension(grid) == 3 ||
        throw(ArgumentError("sphere shape is only available in three dimensions"))
    return LevelSet(x -> sqrt(sum((x .- center) .^ 2)) - radius, grid)
end

function star(grid; radius = 1, deformation = 0.25, n = 5.0)
    dimension(grid) == 2 ||
        throw(ArgumentError("star shape is only available in two dimensions"))
    return LevelSet(grid) do (x, y)
        norm = sqrt(x^2 + y^2)
        θ = atan(y, x)
        return norm - radius * (1.0 + deformation * cos(n * θ))
    end
end

function dumbbell(grid; width = 1, height = 1 / 5, radius = 1 / 4, center = (0, 0))
    cl = circle(grid; center = center .- (width / 2, 0), radius)
    cr = circle(grid; center = center .+ (width / 2, 0), radius)
    rec = rectangle(grid; center, width = (width, height))
    return cl ∪ cr ∪ rec
end

function zalesak_disk(grid; center = (0, 0), radius = 0.5, width = 0.25, height = 1)
    dimension(grid) == 2 ||
        throw(ArgumentError("zalesak disk shape is only available in two dimensions"))
    disk = circle(grid; center = center, radius = radius)
    rec = rectangle(grid; center = center .- (0, radius), width = (width, height))
    # return union(disk, rec)
    return setdiff(disk, rec)
end

#=

Set operations for level set functions.

=#

function Base.union!(ϕ1::LevelSet, ϕ2::LevelSet)
    v1, v2 = values(ϕ1), values(ϕ2)
    v1 .= min.(v1, v2)
    return ϕ1
end
Base.union(ϕ1::LevelSet, ϕ2::LevelSet) = union!(deepcopy(ϕ1), ϕ2)

function Base.intersect!(ϕ1::LevelSet, ϕ2::LevelSet)
    v1, v2 = values(ϕ1), values(ϕ2)
    v1 .= max.(v1, v2)
    return ϕ1
end
Base.intersect(ϕ1::LevelSet, ϕ2::LevelSet) = intersect!(deepcopy(ϕ1), ϕ2)

function complement!(ϕ::LevelSet)
    v = values(ϕ)
    v .= -v
    return ϕ
end
complement(ϕ::LevelSet) = complement!(deepcopy(ϕ))

function Base.setdiff!(ϕ1::LevelSet, ϕ2::LevelSet)
    v1, v2 = values(ϕ1), values(ϕ2)
    v1 .= max.(v1, -v2)
    return ϕ1
end
Base.setdiff(ϕ1::LevelSet, ϕ2::LevelSet) = setdiff!(deepcopy(ϕ1), ϕ2)
