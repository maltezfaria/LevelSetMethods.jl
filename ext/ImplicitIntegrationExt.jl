module ImplicitIntegrationExt

using LevelSetMethods
import ImplicitIntegration as II
using StaticArrays

const LSM = LevelSetMethods
const BP = LevelSetMethods.BernsteinPolynomial

# ImplicitIntegration's `quadgen` consumes any function implementing its interface (`bound`,
# `gradient`, `project`, `split`); see `ImplicitIntegration/src/interface.jl`. We implement
# that interface directly on LSM's own `BernsteinPolynomial` so the quadrature depends on II's
# documented contract, not on its concrete Bernstein type. The operations below are the
# standard Bernstein-basis formulas (same coefficient convention as II), returning LSM
# polynomials so the interface closes recursively under `project`/`split`.

# Sharp value bounds via the convex-hull property: a Bernstein polynomial is bounded by the
# extrema of its coefficients over its defining box.
_bound(p::BP) = extrema(LSM.coefficients(p))

# Derivative along dimension `d`, as a degree-(k-1) Bernstein polynomial on the same box.
function _derivative(p::BP{N}, d::Int) where {N}
    c = LSM.coefficients(p)
    k = LSM.degree(p)[d]
    l, u = LSM.low_corner(p), LSM.high_corner(p)
    scale = k / (u[d] - l[d])
    c′ = mapslices(b -> (b[2:end] .- b[1:(end - 1)]) .* scale, c; dims = d)
    return LSM.BernsteinPolynomial(c′, l, u)
end

# De Casteljau subdivision along dimension `d` at the box midpoint (α = 1/2): returns the two
# sub-polynomials on the left/right halves, exact reparametrizations of `p`.
function _split(p::BP{D, T}, d::Integer, α = 0.5) where {D, T}
    c = LSM.coefficients(p)
    k = LSM.degree(p)[d]
    k == 0 && return p, p
    n = k + 1
    coeffs = mapslices(c; dims = d) do col
        b = collect(col)
        c1 = T[]
        c2 = T[]
        for i in k:-1:1
            push!(c1, b[1])
            pushfirst!(c2, b[i + 1])
            @. b[1:i] = b[1:i] * (1 - α) + b[2:(i + 1)] * α
        end
        push!(c1, b[1])
        append!(c1, c2)
        return c1
    end
    lc, hc = LSM.low_corner(p), LSM.high_corner(p)
    pos = lc[d] + (hc[d] - lc[d]) * α
    p1 = LSM.BernsteinPolynomial(collect(selectdim(coeffs, d, 1:n)), lc, setindex(hc, pos, d))
    p2 = LSM.BernsteinPolynomial(collect(selectdim(coeffs, d, n:(2n - 1))), setindex(lc, pos, d), hc)
    return p1, p2
end

# Restriction to the lower/upper face along dimension `d`, as an (N-1)-dim polynomial.
_lower_restrict(p::BP{D}, d::Integer) where {D} =
    LSM.BernsteinPolynomial(
    collect(selectdim(LSM.coefficients(p), d, 1)),
    deleteat(LSM.low_corner(p), d), deleteat(LSM.high_corner(p), d)
)
_upper_restrict(p::BP{D}, d::Integer) where {D} =
    LSM.BernsteinPolynomial(
    collect(selectdim(LSM.coefficients(p), d, size(LSM.coefficients(p), d))),
    deleteat(LSM.low_corner(p), d), deleteat(LSM.high_corner(p), d)
)

# --- ImplicitIntegration interface (the box args are always the polynomial's own box) ---

II.bound(p::BP, _lc, _hc) = _bound(p)

II.gradient(p::BP{N}) where {N} = ntuple(d -> _derivative(p, d), N)

(∇p::NTuple{N, <:BP})(x::SVector{N}) where {N} = SVector(ntuple(i -> ∇p[i](x), N))

II.bound(∇p::NTuple{N, <:BP}, _lc, _hc) where {N} = SVector(ntuple(i -> _bound(∇p[i]), N))

function II.project(p::BP{N}, k::Int, v) where {N}
    lc, hc = LSM.low_corner(p), LSM.high_corner(p)
    v ≈ lc[k] && return _lower_restrict(p, k)
    v ≈ hc[k] && return _upper_restrict(p, k)
    return error("projection value $v does not match a face of the polynomial")
end

II.split(p::BP, _lb, _ub, dir) = _split(p, dir, 0.5)

# --- quadrature ---

function LevelSetMethods.quadrature(ϕ::LSM.InterpolatedField; quadrature_order, surface = false)
    # FIXME: volume integrals (surface=false) on a NarrowBandMeshField are not supported.
    # We iterate active_cellindices (band cells only), so interior cells deep inside the zero
    # level set are never visited and their volume is silently omitted. Volume support could
    # now iterate cellindices (all mesh cells) instead; wiring that up is a follow-up.
    if ϕ.field isa LSM.NarrowBandMeshField && !surface
        error(
            "volume integrals (surface=false) are not supported on NarrowBandMeshField. " *
                "Use a full MeshField for volume integrals, or pass surface=true for surface integrals."
        )
    end
    N = ndims(LSM.mesh(ϕ))
    quads = Dict{CartesianIndex{N}, II.Quadrature}()
    for I in LSM.active_cellindices(ϕ.field)
        LSM.proven_empty(ϕ, I; surface) && continue
        # bp's coeffs alias the task scratch buffer (see make_interpolant); quadgen only reads
        # them (its sub-polynomials are fresh arrays) and finishes before the next cell, so no
        # copy is needed.
        bp = LSM.make_interpolant(ϕ, I)
        out = II.quadgen(bp, LSM.low_corner(bp), LSM.high_corner(bp); order = quadrature_order, surface)
        isempty(out.quad.coords) || (quads[I] = out.quad)
    end
    return quads
end

function LevelSetMethods.quadrature(mf::LSM.AbstractMeshField; interpolation_order, quadrature_order, surface = false)
    return LevelSetMethods.quadrature(LSM.InterpolatedField(mf, interpolation_order); quadrature_order, surface)
end

end # module
