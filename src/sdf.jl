"""
    struct NewtonSDF{Φ,Tr,P,T}

A signed distance function to the zero level set of an underlying level set function,
computed using a Newton-based closest point method.

Evaluating `sdf(x)` returns the signed distance from point `x` to the interface. An
optional second argument `sdf(x, s)` can be used to supply the sign directly (e.g.
`sign(ϕ(x))`) when it is already known, avoiding an extra interpolant evaluation.

!!! note "Thread safety"
    Evaluating a `NewtonSDF` concurrently from multiple tasks is safe: the underlying
    [`InterpolatedField`](@ref) keeps one scratch buffer per task, and the KDTree and
    sample points are read-only during evaluation.
"""
struct NewtonSDF{Φ, Tr, P, T}
    meshfield::Φ
    tree::Tr
    pts::P           # interface sample points in the original order (tree reorders them)
    upsample::Int
    maxiters::Int
    xtol::T
    ftol::T
end

"""
    get_sample_points(sdf::NewtonSDF)

Return the interface sample points, lying on the surface, used to build the KDTree of `sdf`.
"""
get_sample_points(sdf::NewtonSDF) = sdf.pts

function Base.show(io::IO, ::MIME"text/plain", sdf::NewtonSDF)
    println(io, "NewtonSDF")
    println(io, "  ├─ interface points: $(length(sdf.pts))")
    println(io, "  ├─ upsample:         $(sdf.upsample)×")
    println(io, "  ├─ maxiters:         $(sdf.maxiters)")
    println(io, "  ├─ xtol:             $(sdf.xtol)")
    return print(io, "  └─ ftol:             $(sdf.ftol)")
end

"""
    NewtonSDF(ϕ::AbstractMeshField; order=3, upsample=2, maxiters=10, xtol=nothing, ftol=nothing)

Construct a [`NewtonSDF`](@ref) from an [`AbstractMeshField`]. The interface is sampled by
projecting uniformly-spaced points in each cell onto the zero level set, building a KDTree
for fast nearest-neighbour queries. Given a query point, the closest interface point is
found by seeding a Newton-Lagrange solve from the nearest sample.

# Keyword arguments
- `order`: polynomial interpolation order
- `upsample`: sampling density per cell side
- `maxiters`: maximum Newton iterations
- `xtol`: tolerance on the KKT residual; defaults to `sqrt(eps(T))` with `T = valtype(ϕ)`
- `ftol`: tolerance on the function value; defaults to `sqrt(eps(T))` with `T = valtype(ϕ)`
"""
function NewtonSDF(
        ϕ::AbstractMeshField;
        order = 3,
        upsample = 2,
        maxiters = 10,
        xtol = nothing,
        ftol = nothing,
    )
    check_real_valued(ϕ)
    T = float(valtype(ϕ))
    xtol = convert(T, something(xtol, sqrt(eps(T))))
    ftol = convert(T, something(ftol, sqrt(eps(T))))
    cf = InterpolatedField(deepcopy(ϕ), order)
    grid = mesh(cf)
    pts = _sample_interface(grid, cf, active_cellindices(cf.field), upsample, maxiters, ftol)
    tree = KDTree(pts)
    return NewtonSDF(cf, tree, pts, upsample, maxiters, xtol, ftol)
end

function (sdf::NewtonSDF)(x, s = nothing)
    cp, _, grad = _closest_point_on_interface(sdf, x)
    s === nothing && (s = sign(dot(x - cp, grad)))
    return s * norm(x - cp)
end

"""
    _closest_point_from_seed(sdf, x, seed)

Run the Newton-Lagrange closest-point solver on the Bernstein patch of the cell containing
`seed` (a point already on or near the interface). Returns `(closest_point, converged,
∇φ(closest_point))`, where the gradient gives the (unnormalized, outward) interface normal
at the closest point.
"""
function _closest_point_from_seed(sdf::NewtonSDF, x::SVector, seed::SVector)
    # allow the closest point to reach 1.5 cells out, since the seed is only approximately
    # co-cellular with the true closest point
    safeguard_dist = 3 * maximum(meshsize(mesh(sdf.meshfield))) / 2
    cell = compute_index(sdf.meshfield, seed)
    p = make_interpolant(sdf.meshfield, cell)
    cp, converged = _closest_point(p, x, seed, sdf.maxiters, sdf.xtol, sdf.ftol, safeguard_dist)
    return cp, converged, gradient(p, cp)
end

# Maximum number of nearest interface samples tried as seeds before giving up (see
# `_closest_point_on_interface`).
const _MAX_SEEDS = 10

"""
    _closest_point_on_interface(sdf, x)

Find the point on the interface closest to `x` by seeding a local Newton-Lagrange solve from
the nearest interface sample. Returns `(closest_point, converged, ∇φ(closest_point))`.
"""
function _closest_point_on_interface(sdf::NewtonSDF, x)
    idx, _ = nn(sdf.tree, x)
    cp, converged, g = _closest_point_from_seed(sdf, x, sdf.pts[idx])
    converged && return cp, converged, g
    # widen the seed search only on the rare non-converged nearest seed
    nseeds = min(_MAX_SEEDS, length(sdf.pts))
    idxs, _ = knn(sdf.tree, x, nseeds, true)
    best_cp, best_g, best_d = cp, g, norm(x - cp)
    for j in idxs
        j == idx && continue
        cp, converged, g = _closest_point_from_seed(sdf, x, sdf.pts[j])
        converged && return cp, converged, g
        d = norm(x - cp)
        d < best_d && ((best_cp, best_g, best_d) = (cp, g, d))
    end
    return best_cp, false, best_g
end

"""
    hausdorff_distance(sdf₁::NewtonSDF, sdf₂::NewtonSDF) -> Float64

Approximate the Hausdorff distance between the zero level sets Γ₁ and Γ₂ of `sdf₁` and
`sdf₂`. The symmetric Hausdorff distance is

    d_H(Γ₁, Γ₂) = max(max_{p ∈ Γ₁} d(p, Γ₂),  max_{p ∈ Γ₂} d(p, Γ₁))

Each one-sided maximum is approximated by iterating over the interface sample points of one
`NewtonSDF` and finding the closest point on the other interface via the Newton-Lagrange
solver.
"""
function hausdorff_distance(sdf₁::NewtonSDF, sdf₂::NewtonSDF)
    d₁₂ = maximum(sdf₁.pts) do p
        cp, _ = _closest_point_on_interface(sdf₂, p)
        norm(p - cp)
    end
    d₂₁ = maximum(sdf₂.pts) do p
        cp, _ = _closest_point_on_interface(sdf₁, p)
        norm(p - cp)
    end
    return max(d₁₂, d₂₁)
end

"""
    _sample_cell!(pts, lk, grid, itp, I, upsample, maxiters, ftol, safeguard_dist)

Project uniformly-spaced sample points in cell `I` onto the interface, appending to the
shared `pts` (under lock `lk`) the converged projections that land back inside `I`. Cells
proven to contain no interface are skipped.
"""
function _sample_cell!(
        pts, lk, grid::CartesianGrid{N, T}, itp::InterpolatedField, I, upsample, maxiters,
        ftol, safeguard_dist,
    ) where {N, T}
    proven_empty(itp, I; surface = true) && return pts
    cell = getcell(grid, I)
    ξ_ranges = ntuple(_ -> 0:upsample, N)
    for ξi in Iterators.product(ξ_ranges...)
        x = cell.lc .+ (cell.hc .- cell.lc) .* SVector{N, T}(Tuple(ξi)) ./ upsample
        pt = _project_to_interface(itp, x, maxiters, ftol, safeguard_dist)
        isnothing(pt) && continue
        compute_index(itp, pt) == I || continue
        @lock lk push!(pts, pt)
    end
    return pts
end

"""
    _sample_interface(grid, itp, cells, upsample, maxiters, ftol)

Project uniformly-spaced sample points in each candidate cell onto the interface.
Returns all converged projections; cells proven to be empty are skipped. Cells are
processed in parallel, appending to a single shared vector guarded by a lock.
"""
function _sample_interface(grid::CartesianGrid{N, T}, itp::InterpolatedField, cells, upsample, maxiters, ftol) where {N, T}
    safeguard_dist = maximum(meshsize(grid))
    cellv = collect(cells)
    pts = SVector{N, T}[]
    lk = ReentrantLock()
    Threads.@threads for I in cellv
        _sample_cell!(pts, lk, grid, itp, I, upsample, maxiters, ftol, safeguard_dist)
    end
    return pts
end

"""
    _project_to_interface(itp::InterpolatedField, x_start, maxiters, ftol, safeguard_dist)

Use Newton's method to project a starting point onto the zero level set of `itp`.
Returns the converged point or `nothing` if Newton fails to converge or if the
iterate moves more than `safeguard_dist` from `x_start`. The full [`InterpolatedField`](@ref)
(not a single-cell patch) is used so the projection follows the iterate across cells.
"""
function _project_to_interface(itp::InterpolatedField, x_start, maxiters, ftol, safeguard_dist)
    x = x_start
    for _ in 1:maxiters
        val, grad = value_and_gradient(itp, x)
        abs(val) < ftol && return x
        norm_grad2 = dot(grad, grad)
        iszero(norm_grad2) && break  # guard against 0/0
        x = x - val * grad / norm_grad2
        norm(x - x_start) > safeguard_dist && break
    end
    return nothing
end

"""
    _closest_point(p::BernsteinPolynomial, xq, x0, maxiters, xtol, ftol, safeguard_dist) -> (x_closest, converged)

Find the point on the zero level-set of the single-cell patch `p` closest to `xq`, starting
from `x0`. Uses a Newton-Lagrange solver on the KKT conditions of
`min ||x - xq||² s.t. p(x) = 0`. The patch is fixed: unlike [`_project_to_interface`](@ref),
the iterate is not re-dispatched to neighbouring cells.
"""
function _closest_point(p::BernsteinPolynomial, xq::SVector{N, T}, x0::SVector{N, T}, maxiters, xtol, ftol, safeguard_dist) where {N, T}
    # KKT system of L(x, λ) = ½‖x - xq‖² + λ p(x): the residual stacks stationarity
    # `x - xq + λ∇p` and feasibility `p(x)`, with Jacobian `[I + λ∇²p  ∇p; ∇pᵀ  0]`.
    ∇p₀ = gradient(p, x0)
    g2 = dot(∇p₀, ∇p₀)
    # seed λ so stationarity holds at x0; iszero guards against a vanishing gradient (0/0)
    λ = iszero(g2) ? zero(T) : dot(xq - x0, ∇p₀) / g2
    x = x0
    best_x, best_res = x0, T(Inf)
    for _ in 1:maxiters
        px, ∇p, ∇²p = value_gradient_hessian(p, x)
        res = vcat(x - xq + λ * ∇p, px)
        res_norm = norm(res)
        res_norm < best_res && ((best_res, best_x) = (res_norm, x))
        abs(px) < ftol && res_norm < xtol && return x, true
        # regularize so a near-singular patch (e.g. at a boundary minimum) stays solvable
        hess = hcat(vcat(I + λ * ∇²p, ∇p'), vcat(∇p, 0))
        δ = -(hess + sqrt(eps(T)) * I) \ res
        δx, δλ = δ[SOneTo(N)], δ[end]
        # damp the step so one iterate can't jump out of the patch's neighbourhood
        α = min(one(T), safeguard_dist / norm(δx))
        x, λ = x + α * δx, λ + α * δλ
        # the patch is only accurate locally; once we leave its neighbourhood, give up
        norm(x - x0) > safeguard_dist && return best_x, false
    end
    return best_x, false
end
