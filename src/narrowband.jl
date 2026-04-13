# Build the active-index dict by evaluating `f_at_idx(I)` at every grid node and
# keeping only those where `|v| < γ`.
function _nb_dict(f_at_idx, grid::CartesianGrid{N}, γ::T) where {N, T}
    vals = Dict{CartesianIndex{N}, T}()
    for I in nodeindices(grid)
        v = T(f_at_idx(I))
        abs(v) < γ && (vals[I] = v)
    end
    return vals
end

"""
    NarrowBandMeshField(ϕ::MeshField, halfwidth::Real; reinitialize=true)

Construct a narrow-band field from a full-grid [`MeshField`](@ref).
Active nodes are those where `|ϕ[I]| < halfwidth`. Boundary conditions are
inherited from `ϕ`.

If `reinitialize` is `true` (the default), `ϕ` is first reinitialized to a signed
distance function using [`NewtonReinitializer`](@ref).
"""
function NarrowBandMeshField(ϕ::MeshField, halfwidth::Real; reinitialize::Bool = true)
    bcs = boundary_conditions(ϕ)   # preserve the caller's BCs (may be nothing)
    itp = interp_data(ϕ)           # preserve interpolation data
    if reinitialize
        ϕ = deepcopy(ϕ)
        # reinit needs BCs for gradient computation; add temporary ones if missing
        if !has_boundary_conditions(ϕ)
            ϕ = _add_boundary_conditions(ϕ, ExtrapolationBC(2))
        end
        reinitialize!(ϕ, NewtonReinitializer())
    end
    grid = mesh(ϕ)
    T = float(eltype(ϕ))
    γ = T(halfwidth)
    vals = _nb_dict(I -> ϕ[I], grid, γ)
    return NarrowBandMeshField(vals, grid, bcs, γ, isnothing(itp) ? nothing : copy(itp))
end

"""
    NarrowBandMeshField(ϕ::MeshField; nlayers=3, reinitialize=true)

Construct a narrow-band field with halfwidth automatically computed as
`nlayers * minimum(meshsize(ϕ))`.
"""
function NarrowBandMeshField(ϕ::MeshField; nlayers::Int = 3, reinitialize::Bool = true)
    return NarrowBandMeshField(ϕ, nlayers * minimum(meshsize(ϕ)); reinitialize)
end

# ---- Reinitialization ------------------------------------------------------------

reinitialize!(nb::NarrowBandMeshField, ::Nothing, _) = nb

function reinitialize!(nb::NarrowBandMeshField, r::NewtonReinitializer)
    sdf = NewtonSDF(nb; order = r.order, upsample = r.upsample, maxiters = r.maxiters, xtol = r.xtol, ftol = r.ftol)
    rebuild_band!(nb, sdf)
    return nb
end

function reinitialize!(nb::NarrowBandMeshField, r::NewtonReinitializer, nsteps::Int)
    mod(nsteps, r.reinit_freq) == 0 || return nb
    return reinitialize!(nb, r)
end

# ---- Band rebuild ----------------------------------------------------------------

"""
    rebuild_band!(nb::NarrowBandMeshField, sdf)

Rebuild the active node set from the signed distance function `sdf` using a breadth-first
search seeded from all previously active nodes. The BFS expands axis-aligned neighbors and
adds a node whenever `|sdf(x)| < halfwidth`, stopping a branch when a node falls outside the
band.
"""
function rebuild_band!(nb::NarrowBandMeshField{<:Any, <:AbstractMesh{N}}, sdf) where {N}
    grid = mesh(nb)
    γ = halfwidth(nb)
    grid_axes = axes(nb)
    vals = values(nb)

    queue = collect(keys(vals))
    queue_set = Set{CartesianIndex{N}}(queue)
    empty!(vals)

    head = 1
    while head <= length(queue)
        I = queue[head]
        head += 1
        v = sdf(getnode(grid, I))
        abs(v) >= γ && continue
        vals[I] = v
        for d in 1:N, s in (-1, 1)
            J = _increment_index(I, d, s)
            J ∈ queue_set && continue
            all(d -> J[d] in grid_axes[d], 1:N) || continue
            push!(queue_set, J)
            push!(queue, J)
        end
    end
    return nb
end
