module ReinitializationExt

import LevelSetMethods as LSM
using Interpolations
using NearestNeighbors
using LinearAlgebra
using StaticArrays

function __init__()
    return @info "Loading extension for Newton reinitialization"
end

function LSM.reinitialize!(eq::LSM.LevelSetEquation; kwargs...)
    LSM.reinitialize!(LSM.current_state(eq); kwargs...)
    return eq
end

function LSM.reinitialize!(
        ϕ::LSM.LevelSet;
        upsample::Int = 4,
        maxiters::Int = 10,
        xtol::Float64 = 1.0e-8,
        ftol::Float64 = 1.0e-8,
    )
    grid = LSM.mesh(ϕ)
    vals = ϕ.vals
    maxdist = maximum(LSM.meshsize(grid))
    itp = Interpolations.interpolate(ϕ)
    f(x) = itp(x...)
    ∇f(x) = Interpolations.gradient(itp, x...)
    ∇²f(x) = Interpolations.hessian(itp, x...)

    # Sample the interface
    pts = _sample_interface(grid, f, ∇f, upsample, maxiters, ftol, maxdist)
    tree = KDTree(pts)

    for I in eachindex(grid)
        x = grid[I]
        # Find closest point in the cloud
        idx, dist = nn(tree, x)
        x0 = pts[idx]
        # Refine with Newton's method
        cp = _closest_point(f, ∇f, ∇²f, x, x0, maxiters, xtol, ftol, maxdist)
        vals[I] = sign(vals[I]) * norm(x - cp)
    end
    return ϕ
end

function _sample_interface(grid, f, ∇f, upsample, maxiter, ftol, maxdist)
    pts = Vector{SVector{LSM.dimension(grid), Float64}}()
    for I in CartesianIndices(LSM.size(grid) .- 1)
        Ip = CartesianIndex(Tuple(I) .+ 1)
        lc, hc = grid[I], grid[Ip]
        # Use an (upsample + 1) x (upsample + 1) grid in the cell, always including endpoints
        samples = (lc .+ (hc .- lc) .* SVector(j, k) ./ upsample for j in 0:upsample, k in 0:upsample)
        s = samples |> first |> f |> sign
        any(x -> f(x) * s < 0, samples) || continue
        # Go over samples and push them to the interface
        for x in samples
            pt = _project_to_interface(f, ∇f, x, maxiter, ftol, maxdist)
            isnothing(pt) || push!(pts, pt)
        end
    end
    return pts
end

function _project_to_interface(f, ∇f, x0, maxiter, ftol, maxdist)
    x = x0
    for _ in 1:maxiter
        val = f(x)
        abs(val) < ftol && break # close enough to the interface
        grad = ∇f(x)
        norm_grad = norm(grad)
        δx = val * grad / norm_grad^2
        norm(δx) > maxdist && (return nothing) # too far from interface
        x = x - δx
    end
    f(x) > ftol && (return nothing) # did not converge
    return x
end

function _closest_point(f, ∇f, ∇²f, xq::SVector, x0::SVector, maxiters, xtol, ftol, maxdist)
    x = x0
    ∇p_x0 = ∇f(x0)
    λ = dot(xq - x0, ∇p_x0) / dot(∇p_x0, ∇p_x0)

    converged = false
    for _ in 1:maxiters
        px = f(x)
        ∇p = ∇f(x)
        ∇²p = ∇²f(x)

        # System for Newton's method
        # ∇L = [ x - xq + λ∇p ] = 0
        #      [      p(x)     ]
        grad_L = vcat(x - xq + λ * ∇p, px)

        # Hessian of the Lagrangian
        # H_L = [ I + λ∇²p   ∇p ]
        #       [    ∇p'      0 ]
        hess_L = hcat(vcat(I + λ * ∇²p, ∇p'), vcat(∇p, 0))

        # Solve for the update
        δ = -hess_L \ grad_L
        δx = δ[1:(end - 1)]
        δλ = δ[end]

        # Update variables
        α = 1.0 # TODO: reduce step size if diverging?
        x = x + α * δx
        λ = λ + α * δλ

        # Check for convergence
        if norm(δx) < xtol && norm(f(x)) < ftol
            converged = true
            break
        end
    end

    converged || @warn "closest point search did not converge at xq=$xq"

    return x
end

end
