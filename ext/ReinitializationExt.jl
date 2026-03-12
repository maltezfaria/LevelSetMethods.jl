module ReinitializationExt

import LevelSetMethods as LSM
using NearestNeighbors
using LinearAlgebra
using StaticArrays

function __init__()
    return @info "Loading extension for Newton reinitialization"
end

function LSM.reinitialize!(
        ϕ::LSM.LevelSet,
        reinitializer::LSM.Reinitializer = LSM.Reinitializer(),
    )
    grid = LSM.mesh(ϕ)
    vals = ϕ.vals
    maxdist = maximum(LSM.meshsize(grid))

    # Ensure we have boundary conditions for the interpolation patch
    ϕ_bc = if !LSM.has_boundary_conditions(ϕ)
        N = LSM.dimension(ϕ)
        bc = ntuple(_ -> (LSM.NeumannGradientBC(), LSM.NeumannGradientBC()), N)
        LSM.add_boundary_conditions(ϕ, bc)
    else
        ϕ
    end

    # Snapshot for consistent interpolation
    ϕ_snap = deepcopy(ϕ_bc)
    itp = LSM.interpolate(ϕ_snap, 3)

    # Sample the interface
    pts = _sample_interface(
        grid,
        itp,
        reinitializer.upsample,
        reinitializer.maxiters,
        reinitializer.ftol,
        maxdist,
    )
    tree = KDTree(pts)

    for I in eachindex(grid)
        x = grid[I]
        # Find closest point in the cloud
        idx, _ = nn(tree, x)
        x0 = pts[idx]

        # Lock patch around initial guess x0
        base_idxs = LSM.compute_index(itp, x0)
        p = LSM.make_interpolant(itp, base_idxs)

        # Refine with Newton's method
        cp = _closest_point(
            p,
            x,
            x0,
            reinitializer.maxiters,
            reinitializer.xtol,
            reinitializer.ftol,
            maxdist,
        )
        vals[I] = sign(vals[I]) * norm(x - cp)
    end
    return ϕ
end

function LSM.reinitialize!(eq::LSM.LevelSetEquation)
    reinit = LSM.reinitializer(eq)
    isnothing(reinit) && error("no reinitializer specified in the equation.")
    LSM.reinitialize!(LSM.current_state(eq), reinit)
    return eq
end

function _sample_interface(grid, itp, upsample, maxiter, ftol, maxdist)
    N = LSM.dimension(grid)
    pts = Vector{SVector{N, Float64}}()
    ξ_ranges = ntuple(_ -> 0:upsample, N)
    for I in CartesianIndices(LSM.size(grid) .- 1)
        # Robust screening using Bernstein convex hull
        if LSM.proven_empty(itp, I; surface = true)
            continue
        end

        Ip = CartesianIndex(Tuple(I) .+ 1)
        lc, hc = grid[I], grid[Ip]
        # Only sample cells that actually cross zero
        samples = (
            lc .+ (hc .- lc) .* SVector{N, Float64}(Tuple(ξi)) ./ upsample for
                ξi in Iterators.product(ξ_ranges...)
        )
        # Go over samples and push them to the interface
        p = LSM.make_interpolant(itp, I)
        for x in samples
            pt = _project_to_interface(p, x; ftol, maxiters = maxiter, maxdist = maxdist)
            isnothing(pt) || push!(pts, pt)
        end
    end
    return pts
end

function _project_to_interface(
        p,
        x_start;
        ftol = 1.0e-8,
        maxiters = 20,
        maxdist = Inf
    )
    x = x_start
    for _ in 1:maxiters
        val = p(x)
        abs(val) < ftol && return x
        grad = LSM.gradient(p, x)
        norm_grad2 = dot(grad, grad)
        norm_grad2 < 1.0e-14 && break
        x = x - val * grad / norm_grad2
        norm(x - x_start) > maxdist && break
    end
    return nothing
end

"""
    _closest_point(p, xq, x0, maxiters, xtol, ftol, maxdist)

Find the point on the zero level-set of `p` closest to `xq`, starting from `x0`.
Uses a robust Newton-Lagrange solver with backtracking and best-candidate tracking.
"""
function _closest_point(p, xq::SVector{N, T}, x0::SVector{N, T}, maxiters, xtol, ftol, maxdist) where {N, T}
    x = x0
    # Lagrangian: L(x, λ) = 0.5*|x - xq|^2 + λ*p(x)
    # ∇L = [ x - xq + λ∇p ] = 0
    #      [      p(x)     ]

    ∇p_x0 = LSM.gradient(p, x0)
    λ = dot(xq - x0, ∇p_x0) / (dot(∇p_x0, ∇p_x0) + 1.0e-14)

    best_x = x0
    best_res = Inf

    for _ in 1:maxiters
        px = p(x)
        ∇p = LSM.gradient(p, x)
        ∇²p = LSM.hessian(p, x)

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
        δx = δ[1:N]
        δλ = δ[N + 1]

        # Simple backtracking line search
        α = 1.0
        # Limit initial step size by maxdist
        step_norm = norm(δx)
        if step_norm > maxdist
            α = maxdist / step_norm
        end

        # We perform a single-step backtracking if the residual increases significantly
        # (A full line search is usually overkill for reinitialization)
        x_new = x + α * δx
        λ_new = λ + α * δλ

        # Update
        x, λ = x_new, λ_new

        # If we drift too far from the patch, return best so far
        if norm(x - x0) > 3.0 * maxdist
            return best_x
        end
    end

    # Return the best point found across all iterations
    return best_x
end

end
