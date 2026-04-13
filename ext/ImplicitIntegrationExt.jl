module ImplicitIntegrationExt

using LevelSetMethods
import ImplicitIntegration as II
using StaticArrays

_single_cell(I::CartesianIndex{N}) where {N} = CartesianIndices(ntuple(d -> I[d]:I[d], Val(N)))

function LevelSetMethods.quadrature(ϕ::LevelSetMethods.AbstractMeshField; order, surface = false)
    LevelSetMethods.has_interpolation(ϕ) ||
        error(
        "quadrature requires a MeshField constructed with `interp_order`. " *
            "Pass `interp_order=k` to the MeshField or NarrowBandMeshField constructor."
    )

    grid = LevelSetMethods.mesh(ϕ)
    N = ndims(grid)

    results = Tuple{CartesianIndices{N, NTuple{N, UnitRange{Int}}}, II.Quadrature}[]

    for I in LevelSetMethods.cellindices(ϕ)
        LevelSetMethods.proven_empty(ϕ, I; surface) && continue
        bp_lsm = LevelSetMethods.make_interpolant(ϕ, I)
        bp = II.BernsteinPolynomial(copy(bp_lsm.coeffs), bp_lsm.low_corner, bp_lsm.high_corner)
        out = II.quadgen(bp, bp.low_corner, bp.high_corner; order, surface)
        if !isempty(out.quad.coords)
            push!(results, (_single_cell(I), out.quad))
        end
    end

    return results
end

end # module
