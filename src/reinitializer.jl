"""
    AbstractSignedDistanceFunction <: Function

Abstract type for signed distance functions.

Subtypes should be callable with `sdf(x)`, and must implement [`update!`](@ref) to rebuild
from a new level set.
"""
abstract type AbstractSignedDistanceFunction <: Function end

"""
    update!(sdf::AbstractSignedDistanceFunction, ϕ::LevelSet)

Rebuild `sdf` in place from the new level set `ϕ`.
"""
function update! end

"""
    mutable struct NewtonSDF{I,Tr,P} <: AbstractSignedDistanceFunction

A signed distance function to the zero level set of an underlying level set function,
computed using a Newton-based closest point method.

Evaluating `sdf(x)` returns the signed distance from point `x` to the interface.

!!! warning
    This implementation is **not thread-safe** because the underlying interpolant uses
    internal mutable buffers. For multi-threaded evaluation, use `deepcopy(sdf)` for
    each thread.
"""
mutable struct NewtonSDF{I, Tr, P} <: AbstractSignedDistanceFunction
    itp::I
    tree::Tr
    pts::P
    upsample::Int
    maxiters::Int
    xtol::Float64
    ftol::Float64
end

"""
    get_sample_points(sdf::NewtonSDF)

Return the interface sample points used to build the KDTree of `sdf`.
"""
get_sample_points(sdf::NewtonSDF) = sdf.pts

"""
    NewtonSDF(itp; upsample=8, maxiters=20, xtol=1e-8, ftol=1e-8)

Construct a [`NewtonSDF`](@ref) from a `PiecewisePolynomialInterpolation`.

The interface is sampled by projecting uniformly-spaced points in each cell onto the zero
level set. A KDTree is built from these samples for fast nearest-neighbor queries.

# Keyword arguments
- `upsample`: number of sample subdivisions per cell side (default: 8)
- `maxiters`: maximum Newton iterations (default: 20)
- `xtol`: tolerance on the KKT residual (default: 1e-8)
- `ftol`: tolerance on the function value (default: 1e-8)
"""
function NewtonSDF(itp::PiecewisePolynomialInterpolation; upsample = 8, maxiters = 20, xtol = 1.0e-8, ftol = 1.0e-8)
    grid = mesh(itp.ϕ)
    safeguard_dist = maximum(meshsize(grid))
    pts = _sample_interface(grid, itp, upsample, maxiters, ftol, safeguard_dist)
    tree = KDTree(pts)
    return NewtonSDF(itp, tree, pts, upsample, maxiters, xtol, ftol)
end

"""
    NewtonSDF(ϕ::LevelSet; order=3, kwargs...)

Construct a [`NewtonSDF`](@ref) from a `LevelSet` by first creating a piecewise polynomial
interpolant of the given `order`. Additional keyword arguments are forwarded to
`NewtonSDF(itp; ...)`.
"""
function NewtonSDF(ϕ::LevelSet; order = 3, kwargs...)
    itp = interpolate(ϕ, order)
    return NewtonSDF(itp; kwargs...)
end

"""
    update!(sdf::NewtonSDF, ϕ::LevelSet)

Rebuild `sdf` in place from the new level set `ϕ`, reusing the same interpolation order,
upsample density, and solver tolerances.
"""
function update!(sdf::NewtonSDF, ϕ::LevelSet)
    order = size(sdf.itp.mat, 1) - 1  # recover polynomial order from the matrix size
    sdf.itp = interpolate(ϕ, order)
    grid = mesh(sdf.itp.ϕ)
    safeguard_dist = maximum(meshsize(grid))
    sdf.pts = _sample_interface(grid, sdf.itp, sdf.upsample, sdf.maxiters, sdf.ftol, safeguard_dist)
    sdf.tree = KDTree(sdf.pts)
    return sdf
end

function (sdf::NewtonSDF)(x)
    cp = _closest_point_on_interface(sdf, x)
    return sign(sdf.itp(x)) * norm(x - cp)
end

function _closest_point_on_interface(sdf::NewtonSDF, x)
    safeguard_dist = maximum(meshsize(mesh(sdf.itp.ϕ)))
    idx, _ = nn(sdf.tree, x)
    x0 = sdf.pts[idx]
    base_idxs = compute_index(sdf.itp, x0)
    p = make_interpolant(sdf.itp, base_idxs)
    return _closest_point(p, x, x0, sdf.maxiters, sdf.xtol, sdf.ftol, safeguard_dist)
end

function _sample_interface(grid, itp, upsample, maxiters, ftol, safeguard_dist)
    N = dimension(grid)
    T = float(eltype(eltype(grid)))  # scalar floating-point type of grid coordinates
    P = SVector{N, T}
    pts = Vector{P}()
    ξ_ranges = ntuple(_ -> 0:upsample, N)
    for I in CartesianIndices(size(grid) .- 1)
        # Robust screening using Bernstein convex hull
        if proven_empty(itp, I; surface = true)
            continue
        end

        Ip = CartesianIndex(Tuple(I) .+ 1)
        lc, hc = grid[I], grid[Ip]
        samples = (
            lc .+ (hc .- lc) .* SVector{N, T}(Tuple(ξi)) ./ upsample for
                ξi in Iterators.product(ξ_ranges...)
        )
        p = make_interpolant(itp, I)
        for x in samples
            pt = _project_to_interface(itp, x, maxiters, ftol, safeguard_dist)
            isnothing(pt) || push!(pts, pt)
        end
    end
    return pts
end

function _project_to_interface(p, x_start, maxiters, ftol, safeguard_dist)
    x = x_start
    for _ in 1:maxiters
        val, grad = value_and_gradient(p, x)
        abs(val) < ftol && return x
        norm_grad2 = dot(grad, grad)
        norm_grad2 < 1.0e-14 && break
        x = x - val * grad / norm_grad2
        norm(x - x_start) > safeguard_dist && break
    end
    return nothing
end

"""
    _closest_point(p, xq, x0, maxiters, xtol, ftol, safeguard_dist)

Find the point on the zero level-set of `p` closest to `xq`, starting from `x0`.
Uses a Newton-Lagrange solver on the KKT conditions of `min ||x - xq||² s.t. p(x) = 0`.
"""
function _closest_point(p::F, xq::SVector{N, T}, x0::SVector{N, T}, maxiters, xtol, ftol, safeguard_dist) where {F, N, T}
    x = x0
    # Lagrangian: L(x, λ) = 0.5*|x - xq|^2 + λ*p(x)
    # ∇L = [ x - xq + λ∇p ] = 0
    #      [      p(x)     ]

    ∇p_x0 = gradient(p, x0)
    # Initialize λ from the stationarity condition at x0: λ ≈ <xq-x0, ∇p> / ||∇p||²
    λ = dot(xq - x0, ∇p_x0) / (dot(∇p_x0, ∇p_x0) + 1.0e-14)

    best_x = x0
    best_res = Inf

    for _ in 1:maxiters
        px, ∇p, ∇²p = value_gradient_hessian(p, x)

        # Residual of the KKT system
        res_vec = vcat(x - xq + λ * ∇p, px)
        res_norm = norm(res_vec)

        # Track the best candidate found so far
        if res_norm < best_res
            best_res = res_norm
            best_x = x
        end

        # Check for convergence
        if abs(px) < ftol && res_norm < xtol
            return x
        end

        # Newton step: H_L * δ = -grad_L
        hess_L = hcat(vcat(I + λ * ∇²p, ∇p'), vcat(∇p, 0))

        # Regularization for stability near singularities
        δ = -(hess_L + 1.0e-10 * I) \ res_vec
        δx = δ[SOneTo(N)]
        δλ = δ[end]

        # Limit initial step size to one cell width
        α = 1.0
        step_norm = norm(δx)
        if step_norm > safeguard_dist
            α = safeguard_dist / step_norm
        end

        x, λ = x + α * δx, λ + α * δλ

        # If we drift too far from the patch, return best so far
        if norm(x - x0) > 3 * safeguard_dist
            return best_x
        end
    end

    return best_x
end

"""
    struct NewtonReinitializer

Reinitializes a level set to a signed distance function using a Newton closest-point method.
At each reinitialization step, a piecewise polynomial interpolant is built from the current
level set, the interface is sampled, and the level set values are overwritten with the signed
distances.

# Keyword arguments
- `reinit_freq`: reinitialization frequency in time steps (default: `1`)
- `order`: polynomial interpolation order (default: `3`)
- `upsample`: interface sampling density per cell side (default: `8`)
- `maxiters`: maximum Newton iterations (default: `20`)
- `xtol`: tolerance on the KKT residual (default: `1e-8`)
- `ftol`: tolerance on the function value (default: `1e-8`)

# Examples
```julia
eq = LevelSetEquation(; terms, levelset = ϕ, bc, reinit = NewtonReinitializer(; reinit_freq = 5))
```
"""
struct NewtonReinitializer
    reinit_freq::Int
    order::Int
    upsample::Int
    maxiters::Int
    xtol::Float64
    ftol::Float64
end

function NewtonReinitializer(;
        reinit_freq = 1,
        order = 3,
        upsample = 8,
        maxiters = 20,
        xtol = 1.0e-8,
        ftol = 1.0e-8,
    )
    return NewtonReinitializer(reinit_freq, order, upsample, maxiters, xtol, ftol)
end

"""
    reinitialize!(ϕ::LevelSet, r::NewtonReinitializer)

Reinitialize the level set `ϕ` to a signed distance function in place using the Newton
closest-point method.
"""
function reinitialize!(ϕ::LevelSet, r::NewtonReinitializer)
    sdf = NewtonSDF(ϕ; order = r.order, upsample = r.upsample, maxiters = r.maxiters, xtol = r.xtol, ftol = r.ftol)
    for I in eachindex(ϕ)
        ϕ[I] = sdf(mesh(ϕ)[I])
    end
    return ϕ
end

# Called at each time step with the current step count. Reinitializes only when
# nsteps is a multiple of reinit_freq; the nothing method is a no-op.
reinitialize!(ϕ::LevelSet, ::Nothing, nsteps::Int) = ϕ
function reinitialize!(ϕ::LevelSet, r::NewtonReinitializer, nsteps::Int)
    mod(nsteps, r.reinit_freq) == 0 || return ϕ
    return reinitialize!(ϕ, r)
end

# tomatoes tomatos ...
reinitialise!(ϕ::LevelSet, r::NewtonReinitializer) = reinitialize!(ϕ, r)
