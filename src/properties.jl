"""
    Volume(ϕ::LevelSet)

Compute the volume of the level-set function.

```julia
using LevelSetMethods
R = 0.5
V0 = π * R^2
V = []
Ns = 8:8:256
nb = length(Ns)
for N in Ns
    grid = CartesianGrid((-1, -1), (1, 1), (N, N))
    ϕ = LevelSetMethods.sphere(grid; radius = R)
    push!(V, Volume(ϕ))
end
lines(Ns, V; color = "blue")
lines!([minimum(Ns); maximum(Ns)], [V0; V0]; color = "red")
```
"""
function Volume(ϕ::LevelSet)
    δ = meshsize(LevelSetMethods.mesh(ϕ))
    δmin = minimum(δ)
    vol = prod(δ)
    sign = SmoothHeaviside.(-values(ϕ), δmin)
    return vol * sum(sign .* TrapezoidalCoefficients(ϕ))
end

"""
    Perimeter(ϕ::LevelSet)

Compute the perimeter area of the level-set function.
Note: this function does not compute the perimeter on the borders of the domain.

```julia
using LevelSetMethods
R = 0.5
S0 = 2π * R
S = []
Ns = 8:8:256
nb = length(Ns)
for N in Ns
    grid = CartesianGrid((-1, -1), (1, 1), (N, N))
    ϕ = LevelSetMethods.sphere(grid; radius = R)
    push!(S, Perimeter(ϕ))
end
lines(Ns, S; color = "blue")
lines!([minimum(Ns); maximum(Ns)], [S0; S0]; color = "red")
```
"""
function Perimeter(ϕ::LevelSet)
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
    integrand = SmoothDelta.(values(ϕ), δmin) .* nrm
    return vol * sum(integrand .* TrapezoidalCoefficients(ϕ))
end

# from "A Variational Level Set Approach to Multiphase Motion"
function SmoothHeaviside(x, α)
    if x > α
        return 1.0
    elseif x < -α
        return 0.0
    else
        return 0.5 * (1.0 + x / α + 1.0 / π * sin(π * x / α))
    end
end

function SmoothDelta(x, α)
    return abs(x) > α ? 0.0 : 0.5 / α * (1.0 + cos(π * x / α))
end

# Method used to numerically integrate N-th dimensional matrices.
# return a matrix with coefficients equal to 2^(-n) where n is
# the number of times this index if on the borders of a dimension.
function TrapezoidalCoefficients(ϕ::LevelSet)
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
    ∇ϕ = gradient(ϕ, I)
    nrmsq = dot(∇ϕ, ∇ϕ)
    nrmsq == 0.0 && return 0.0
    Hϕ = hessian(ϕ, I)
    Δϕ = tr(Hϕ)
    κ = (Δϕ * nrmsq - ∇ϕ' * Hϕ * ∇ϕ) / nrmsq^(3 / 2)
    return κ
end

"""
    curvature(ϕ::LevelSet)

Compute the mean curvature of ϕ at I using κ = ∇ ⋅ (∇ϕ / |∇ϕ|).
See [`curvature(ϕ::LevelSet, I)`](@ref)@ for more details.

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
