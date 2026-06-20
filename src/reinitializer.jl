"""
    reinitialize!(ϕ::AbstractMeshField; order=3, upsample=2, maxiters=20, xtol=nothing, ftol=nothing)

Reinitialize the level set `ϕ` to a signed distance function in place using the Newton
closest-point method: a [`NewtonSDF`](@ref) is built from `ϕ` and each active grid node is
overwritten with its signed distance to the interface (a Newton-Lagrange solve seeded from
the nearest interface sample). The keyword arguments are forwarded to [`NewtonSDF`](@ref).

Reinitialization is not built into the time integrator; drive it from a `prehook`, e.g.
`prehook = eq -> reinitialize!(current_state(eq); order = 5)`. See [`integrate!`](@ref).
"""
function reinitialize!(
        ϕ::AbstractMeshField;
        order = 3, upsample = 2, maxiters = 20, xtol = nothing, ftol = nothing,
    )
    sdf = NewtonSDF(ϕ; order, upsample, maxiters, xtol, ftol)
    return _apply_sdf!(ϕ, sdf)
end

# Overwrite the active values of `ϕ` with signed distances from `sdf`. Nodes are evaluated in
# parallel into a buffer, then written back serially, keeping the evaluation phase free of
# writes to `ϕ` (`sdf` interpolates its own private copy of the level set; only the sign is
# read from `ϕ`). Only the values are touched; the active set is left untouched.
function _apply_sdf!(ϕ::AbstractMeshField, sdf::NewtonSDF)
    grid = mesh(ϕ)
    idxs = vec(collect(active_nodeindices(ϕ)))
    T = float(valtype(ϕ))
    vals = Vector{T}(undef, length(idxs))
    nfail = Threads.Atomic{Int}(0)
    Threads.@threads for k in eachindex(idxs)
        x = getnode(grid, idxs[k])
        cp, converged, _ = _closest_point_on_interface(sdf, x)
        converged || Threads.atomic_add!(nfail, 1)
        vals[k] = sign(ϕ[idxs[k]]) * norm(x - cp)
    end
    for (k, I) in enumerate(idxs)
        ϕ[I] = vals[k]
    end
    nfail[] > 0 &&
        @warn "reinitialize!: closest-point solver did not converge for $(nfail[]) / $(length(idxs)) points"
    return ϕ
end
