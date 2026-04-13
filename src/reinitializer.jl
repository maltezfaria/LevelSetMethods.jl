"""
    AbstractSignedDistanceFunction <: Function

Abstract type for signed distance functions.

Subtypes should be callable with `sdf(x)`, and must implement [`update!`](@ref) to rebuild
from a new level set.
"""
abstract type AbstractSignedDistanceFunction <: Function end

"""
    update!(sdf::AbstractSignedDistanceFunction, ϕ::MeshField)

Rebuild `sdf` in place from the new level set `ϕ`.
"""
function update! end

"""
    mutable struct NewtonSDF{I,Tr,P} <: AbstractSignedDistanceFunction

A signed distance function to the zero level set of an underlying level set function,
computed using a Newton-based closest point method.

Evaluating `sdf(x)` returns the signed distance from point `x` to the interface. An
optional second argument `sdf(x, s)` can be used to supply the sign directly (e.g.
`sign(ϕ(x))`) when it is already known, avoiding an extra interpolant evaluation.

!!! warning
    This implementation is **not thread-safe** because the underlying interpolant uses
    internal mutable buffers. For multi-threaded evaluation, use `deepcopy(sdf)` for
    each thread.
"""
mutable struct NewtonSDF{Φ, Tr, P} <: AbstractSignedDistanceFunction
    meshfield::Φ
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

function Base.show(io::IO, ::MIME"text/plain", sdf::NewtonSDF)
    println(io, "NewtonSDF")
    println(io, "  ├─ interface points: $(length(sdf.pts))")
    println(io, "  ├─ upsample:         $(sdf.upsample)×")
    println(io, "  ├─ maxiters:         $(sdf.maxiters)")
    println(io, "  ├─ xtol:             $(sdf.xtol)")
    return print(io, "  └─ ftol:             $(sdf.ftol)")
end

"""
    NewtonSDF(ϕ; order=3, upsample=2, maxiters=10, xtol=1e-8, ftol=1e-8)

Construct a [`NewtonSDF`](@ref) from a level set `ϕ`. If `ϕ` already carries
`InterpolationData` (i.e. was constructed with an `interp_order` keyword), it is used
directly; otherwise a deep copy is made with `ExtrapolationBC{2}` added and
`InterpolationData` of the given `order` attached.

The interface is sampled by projecting uniformly-spaced points in each cell onto the
zero level set, building a KDTree for fast nearest-neighbour queries.

Works for both [`MeshField`](@ref) and narrow-band fields.

# Keyword arguments
- `order`: polynomial interpolation order (only used when `ϕ` has no interpolation)
- `upsample`: sampling density per cell side
- `maxiters`: maximum Newton iterations
- `xtol`: tolerance on the KKT residual
- `ftol`: tolerance on the function value
"""
function NewtonSDF(
        ϕ;
        order = 3,
        upsample = 2,
        maxiters = 10,
        xtol = 1.0e-8,
        ftol = 1.0e-8,
    )
    if has_interpolation(ϕ)
        mf = ϕ
    else
        ϕ_copy = deepcopy(ϕ)
        if !has_boundary_conditions(ϕ_copy)
            N = ndims(mesh(ϕ_copy))
            ϕ_copy = _add_boundary_conditions(ϕ_copy, ntuple(_ -> (ExtrapolationBC{2}(), ExtrapolationBC{2}()), N))
        end
        N, T = ndims(mesh(ϕ_copy)), eltype(ϕ_copy)
        if ϕ_copy isa NarrowBandMeshField
            mf = NarrowBandMeshField(
                values(ϕ_copy), mesh(ϕ_copy), boundary_conditions(ϕ_copy),
                halfwidth(ϕ_copy), InterpolationData(N, order, T),
            )
        else
            mf = MeshField(values(ϕ_copy), mesh(ϕ_copy), boundary_conditions(ϕ_copy), InterpolationData(N, order, T))
        end
    end
    grid = mesh(mf)
    pts = _sample_interface(grid, mf, cellindices(mf), upsample, maxiters, ftol)
    tree = KDTree(pts)
    return NewtonSDF(mf, tree, pts, upsample, maxiters, xtol, ftol)
end

"""
    update!(sdf::NewtonSDF, ϕ)

Rebuild `sdf` in place from the new level set `ϕ`, reusing the existing interpolant
buffers, upsample density, and solver tolerances.
"""
function update!(sdf::NewtonSDF, ϕ)
    copy!(sdf.meshfield, ϕ)
    grid = mesh(sdf.meshfield)
    sdf.pts = _sample_interface(grid, sdf.meshfield, cellindices(sdf.meshfield), sdf.upsample, sdf.maxiters, sdf.ftol)
    sdf.tree = KDTree(sdf.pts)
    return sdf
end

function (sdf::NewtonSDF)(x, s = sign(sdf.meshfield(x)))
    cp, _ = _closest_point_on_interface(sdf, x)
    return s * norm(x - cp)
end

"""
    _closest_point_on_interface(sdf, x)

Find the point on the interface closest to `x` by nearest-neighbor seeding into a
local Newton-Lagrange solve. Returns `(closest_point, converged)`.

If the first solve does not converge (e.g. because the closest point lies on a
neighbouring polynomial patch), a single retry is attempted using the best iterate
from the failed solve as a new seed on its own patch.
"""
function _closest_point_on_interface(sdf::NewtonSDF, x, max_retries = 3)
    safeguard_dist = 1.5 * maximum(meshsize(mesh(sdf.meshfield)))
    idx, _ = nn(sdf.tree, x)
    cp = sdf.pts[idx]
    cell = compute_index(sdf.meshfield, cp)
    converged = false
    for _ in 1:max_retries
        p = make_interpolant(sdf.meshfield, cell)
        cp, converged = _closest_point(p, x, cp, sdf.maxiters, sdf.xtol, sdf.ftol, safeguard_dist)
        new_cell = compute_index(sdf.meshfield, cp)
        (converged || new_cell == cell) && break
        cell = new_cell
    end
    return cp, converged
end


"""
    _sample_interface(grid, itp, cells, upsample, maxiters, ftol)

Project uniformly-spaced sample points in each candidate cell onto the interface.
Returns all converged projections; cells proven to be empty are skipped.
"""
function _sample_interface(grid::CartesianGrid{N, T}, itp, cells, upsample, maxiters, ftol) where {N, T}
    pts = SVector{N, T}[]
    ξ_ranges = ntuple(_ -> 0:upsample, N)
    safeguard_dist = maximum(meshsize(grid))
    for I in cells
        proven_empty(itp, I; surface = true) && continue
        cell = getcell(grid, I)
        samples = (
            cell.lc .+ (cell.hc .- cell.lc) .* SVector{N, T}(Tuple(ξi)) ./ upsample for
                ξi in Iterators.product(ξ_ranges...)
        )
        for x in samples
            pt = _project_to_interface(itp, x, maxiters, ftol, safeguard_dist)
            isnothing(pt) && continue
            I_pt = compute_index(itp, pt)
            I_pt in cells || continue
            push!(pts, pt)
        end
    end
    return pts
end

"""
    _project_to_interface(p, x_start, maxiters, ftol, safeguard_dist)

Use Newton's method to project a starting point onto the zero level set of `p`.
Returns the converged point or `nothing` if Newton fails to converge or if the
iterate moves more than `safeguard_dist` from `x_start`.
"""
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
    _closest_point(p, xq, x0, maxiters, xtol, ftol, safeguard_dist) -> (x_closest, converged)

Find the point on the zero level-set of `p` closest to `xq`, starting from `x0`.  Uses a
Newton-Lagrange solver on the KKT conditions of `min ||x - xq||² s.t. p(x) = 0`.
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
            return x, true
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
        if norm(x - x0) > safeguard_dist
            return best_x, false
        end
    end

    return best_x, false
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

```jldoctest; output = true
using LevelSetMethods
NewtonReinitializer()

# output

NewtonReinitializer
  ├─ frequency: every step
  ├─ order:     3
  ├─ upsample:  8×
  ├─ maxiters:  20
  ├─ xtol:      1.0e-8
  └─ ftol:      1.0e-8
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

function Base.show(io::IO, ::MIME"text/plain", r::NewtonReinitializer)
    freq_str = r.reinit_freq == 1 ? "every step" : "every $(r.reinit_freq) steps"
    println(io, "NewtonReinitializer")
    println(io, "  ├─ frequency: $freq_str")
    println(io, "  ├─ order:     $(r.order)")
    println(io, "  ├─ upsample:  $(r.upsample)×")
    println(io, "  ├─ maxiters:  $(r.maxiters)")
    println(io, "  ├─ xtol:      $(r.xtol)")
    return print(io, "  └─ ftol:      $(r.ftol)")
end

"""
    reinitialize!(ϕ::MeshField, r::NewtonReinitializer)

Reinitialize the level set `ϕ` to a signed distance function in place using the Newton
closest-point method.
"""
function reinitialize!(ϕ::MeshField, r::NewtonReinitializer)
    sdf = NewtonSDF(ϕ; order = r.order, upsample = r.upsample, maxiters = r.maxiters, xtol = r.xtol, ftol = r.ftol)
    nfail = 0
    for I in eachindex(ϕ)
        x = getnode(mesh(ϕ), I)
        cp, converged = _closest_point_on_interface(sdf, x)
        converged || (nfail += 1)
        ϕ[I] = sign(ϕ[I]) * norm(x - cp)
    end
    if nfail > 0
        @warn "NewtonReinitializer: closest-point solver did not converge for $nfail / $(length(ϕ)) grid points"
    end
    return ϕ
end

# Called at each time step with the current step count. Reinitializes only when
# nsteps is a multiple of reinit_freq; the nothing method is a no-op.
reinitialize!(ϕ::MeshField, ::Nothing, _) = ϕ
function reinitialize!(ϕ::MeshField, r::NewtonReinitializer, nsteps::Int)
    mod(nsteps, r.reinit_freq) == 0 || return ϕ
    return reinitialize!(ϕ, r)
end

# tomatoes tomatos ...
reinitialise!(ϕ::MeshField, r::NewtonReinitializer) = reinitialize!(ϕ, r)
