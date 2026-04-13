module ImplicitIntegrationExt

using LevelSetMethods
import ImplicitIntegration as II
using StaticArrays

function _bounding_box(indices::AbstractVector{CartesianIndex{N}}) where {N}
    isempty(indices) && return CartesianIndices(ntuple(d -> 1:0, N))
    mins = MVector(Tuple(first(indices)))
    maxs = MVector(Tuple(first(indices)))
    for I in indices
        for d in 1:N
            mins[d] = min(mins[d], I[d])
            maxs[d] = max(maxs[d], I[d])
        end
    end
    return CartesianIndices(ntuple(d -> mins[d]:maxs[d], N))
end

function _rcb_partition(active_indices::AbstractVector{CartesianIndex{N}}, masses::AbstractVector{T}, M::T, c::T) where {N, T}
    if length(active_indices) <= 1
        return [(_bounding_box(active_indices), active_indices, masses)]
    end

    region = _bounding_box(active_indices)
    if length(region) == 1
        return [(region, active_indices, masses)]
    end

    sizes = ntuple(d -> length(region.indices[d]), N)
    sorted_dims = sort(1:N, by = d -> sizes[d], rev = true)

    for d in sorted_dims
        dim_start = region.indices[d][1]
        dim_len = sizes[d]

        # Marginal mass prefix sum
        marginal_mass = zeros(T, dim_len)
        for i in eachindex(active_indices)
            I = active_indices[i]
            idx = I[d] - dim_start + 1
            marginal_mass[idx] += masses[i]
        end

        prefix_sum = copy(marginal_mass)
        for i in 2:dim_len
            prefix_sum[i] += prefix_sum[i - 1]
        end

        total_mass = prefix_sum[end]

        best_split_idx = nothing
        min_diff = typemax(T)

        for k in 1:(dim_len - 1)
            m1 = prefix_sum[k]
            m2 = total_mass - m1

            if m1 >= c * M && m2 >= c * M
                diff = abs(m1 - m2)
                if diff < min_diff
                    min_diff = diff
                    best_split_idx = dim_start + k - 1
                end
            end
        end

        if best_split_idx !== nothing
            indices1 = CartesianIndex{N}[]
            masses1 = T[]
            indices2 = CartesianIndex{N}[]
            masses2 = T[]

            for i in eachindex(active_indices)
                I = active_indices[i]
                if I[d] <= best_split_idx
                    push!(indices1, I)
                    push!(masses1, masses[i])
                else
                    push!(indices2, I)
                    push!(masses2, masses[i])
                end
            end
            return [_rcb_partition(indices1, masses1, M, c)..., _rcb_partition(indices2, masses2, M, c)...]
        end
    end

    return [(region, active_indices, masses)]
end

_single_cell(I::CartesianIndex{N}) where {N} = CartesianIndices(ntuple(d -> I[d]:I[d], Val(N)))

function LevelSetMethods.quadrature(ϕ::LevelSetMethods.AbstractMeshField; order, surface = false, min_mass_fraction = 0.0)
    LevelSetMethods.has_interpolation(ϕ) ||
        error(
        "quadrature requires a MeshField constructed with `interp_order`. " *
            "Pass `interp_order=k` to the MeshField or MeshField constructor."
    )

    grid = LevelSetMethods.mesh(ϕ)
    N = ndims(grid)

    initial_results = Dict{CartesianIndex{N}, II.Quadrature}()
    masses = Dict{CartesianIndex{N}, Float64}()

    for I in LevelSetMethods.cellindices(ϕ)
        LevelSetMethods.proven_empty(ϕ, I; surface) && continue
        bp_lsm = LevelSetMethods.make_interpolant(ϕ, I)
        bp = II.BernsteinPolynomial(copy(bp_lsm.coeffs), bp_lsm.low_corner, bp_lsm.high_corner)
        out = II.quadgen(bp, bp.low_corner, bp.high_corner; order, surface)
        if !isempty(out.quad.coords)
            initial_results[I] = out.quad
            masses[I] = sum(out.quad.weights)
        end
    end

    results = Tuple{CartesianIndices{N, NTuple{N, UnitRange{Int}}}, II.Quadrature}[]

    if min_mass_fraction <= 0.0 || isempty(initial_results)
        for (I, q) in initial_results
            push!(results, (_single_cell(I), q))
        end
        return results
    end

    M = maximum(values(masses))
    active_indices = collect(keys(initial_results))
    masses_vec = [masses[I] for I in active_indices]
    supercells = _rcb_partition(active_indices, masses_vec, M, Float64(min_mass_fraction))

    for (R, cells_in_R, masses_in_R) in supercells
        if isempty(cells_in_R)
            continue
        elseif length(cells_in_R) == 1
            I = first(cells_in_R)
            push!(results, (_single_cell(I), initial_results[I]))
        else
            I_max = cells_in_R[argmax(masses_in_R)]

            bp_lsm = LevelSetMethods.make_interpolant(ϕ, I_max)
            low_corner = SVector(LevelSetMethods.getnode(grid, first(R)))
            high_corner = SVector(LevelSetMethods.getnode(grid, last(R) + CartesianIndex(ntuple(d -> 1, N))))

            # Reconstruct the polynomial over the supercell
            n = size(bp_lsm.coeffs)
            bp_super = II.berninterp(x -> bp_lsm(x), n, low_corner, high_corner)

            out = II.quadgen(bp_super, low_corner, high_corner; order, surface)

            mass_children = sum(masses_in_R)
            mass_super = sum(out.quad.weights)

            # Reject the supercell if it is geometrically bloated (bbox > 3× cell count) or
            # if its integrated mass deviates more than 10% from the sum of its children —
            # either signals a poor interpolant over the merged region.
            if length(R) > 3 * length(cells_in_R) || abs(mass_super - mass_children) > 0.1 * mass_children
                for I in cells_in_R
                    push!(results, (_single_cell(I), initial_results[I]))
                end
            else
                if !isempty(out.quad.coords)
                    push!(results, (R, out.quad))
                end
            end
        end
    end

    return results
end

end # module
