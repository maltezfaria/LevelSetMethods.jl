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

    # FIXME: volume integrals (surface=false) on a NarrowBandMeshField are not supported.
    # cellindices(nb) only returns cells whose corners are all stored in the band dict,
    # so interior cells deep inside the zero level set are never visited and their volume
    # is silently omitted.  The fix requires splitting cellindices into two functions:
    # one returning all mesh cells (for volume integrals) and one returning only band cells
    # (for surface integrals and interface sampling).  See TODO.md.
    if ϕ isa LevelSetMethods.NarrowBandMeshField && !surface
        error(
            "volume integrals (surface=false) are not supported on NarrowBandMeshField. " *
                "Use a full MeshField for volume integrals, or pass surface=true for surface integrals."
        )
    end

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
