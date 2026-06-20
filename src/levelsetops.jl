#=
A level set is a real-valued `AbstractMeshField` whose zero contour represents an interface.
This file collects operations that interpret a field that way: geometric measures,
differential geometry, and set (CSG) operations on the enclosed domains.
=#

"""
    volume(ϕ::MeshField)

Measure of the region `{x : ϕ(x) ≤ 0}` enclosed by the zero level set of `ϕ`. In `N`
dimensions this is the `N`-dimensional measure — a length in 1D, an area in 2D, a volume in
3D — approximated as `∫ H(-ϕ) dx` with a smoothed Heaviside `H`.

```jldoctest
using LevelSetMethods
R = 0.5
V₀ = π * R^2
grid = CartesianGrid((-1, -1), (1, 1), (200, 200))
ϕ = MeshField(x -> sqrt(sum(abs2, x)) - R, grid)
LevelSetMethods.volume(ϕ), V₀

# output

(0.7854362890190668, 0.7853981633974483)
```
"""
function volume(ϕ::MeshField)
    check_real_valued(ϕ)
    δ = meshsize(mesh(ϕ))
    δmin = minimum(δ)
    vol = prod(δ)
    return vol * sum(v -> smooth_heaviside(-v, δmin), values(ϕ))
end

"""
    volume(nb::NarrowBandMeshField)

Measure of the region `{x : ϕ(x) ≤ 0}` enclosed by the zero level set, matching
[`volume(ϕ::MeshField)`](@ref) but computed from the band alone, without materialising values
on the full grid.

The smoothed Heaviside transition is supported within `δmin` of the interface — entirely
inside the band — so every off-band node contributes exactly `0` (outside) or `1` (inside),
and only its *sign* is needed. Those signs are recovered by a scanline sweep: the interface
never crosses outside the band, so along each grid line the off-band interior shows up as
same-sign gaps between consecutive band nodes (and the tails beyond the outermost ones),
counted by arithmetic without ever visiting an interior node. A line with no band node carries
no crossing and is classified by the sign of its nearest band node.
"""
function volume(nb::NarrowBandMeshField{N}) where {N}
    check_real_valued(nb)
    d = values(nb)
    δ = meshsize(mesh(nb))
    T = float(valtype(nb))
    isempty(d) && return zero(T)   # no interface captured by the band
    δmin = minimum(δ)
    # near-interface contribution: smoothed Heaviside over the band nodes themselves
    interface = sum(v -> smooth_heaviside(-v, δmin), Base.values(d))
    return prod(δ) * (interface + _count_offband_interior(nb))
end

# Transverse key of a node: its indices in every dimension but the first, i.e. the grid line
# (along dim 1) it lies on.
_transverse_key(I::CartesianIndex{N}) where {N} = ntuple(k -> I[k + 1], N - 1)

# Count the off-band interior (`ϕ < 0`) nodes of `nb` without materialising the grid, by
# sweeping each scanline along dimension 1. See [`volume`](@ref).
function _count_offband_interior(nb::NarrowBandMeshField{N}) where {N}
    d = values(nb)
    n1 = size(mesh(nb))[1]
    # group band nodes by scanline (transverse key), ordered along dim 1 within each line
    ks = collect(keys(d))
    sort!(ks; by = I -> (_transverse_key(I)..., I[1]))
    count = 0
    band_lines = Set{NTuple{N - 1, Int}}()
    i, M = 1, length(ks)
    while i <= M
        j = i
        while j < M && _transverse_key(ks[j + 1]) == _transverse_key(ks[i])
            j += 1
        end
        push!(band_lines, _transverse_key(ks[i]))
        first_node, last_node = ks[i], ks[j]
        # tails beyond the outermost band nodes are crossing-free, hence uniformly signed
        d[first_node] < 0 && (count += first_node[1] - 1)
        d[last_node] < 0 && (count += n1 - last_node[1])
        # each gap between consecutive band nodes is likewise crossing-free, hence same-signed
        for t in i:(j - 1)
            gap = ks[t + 1][1] - ks[t][1] - 1
            (gap > 0 && d[ks[t]] < 0 && d[ks[t + 1]] < 0) && (count += gap)
        end
        i = j + 1
    end
    return count + _count_bandfree_interior(nb, band_lines, n1)
end

# Classify each band-free scanline (no band node ⇒ no crossing ⇒ uniform sign) by the sign of
# its nearest band node, counting the whole line when interior. A KDTree over the band-node
# index coordinates keeps this scalable when many lines are band-free.
function _count_bandfree_interior(nb::NarrowBandMeshField{N}, band_lines, n1) where {N}
    sz = size(mesh(nb))
    transverse = CartesianIndices(ntuple(k -> sz[k + 1], N - 1))
    length(band_lines) == length(transverse) && return 0   # every line carries a band node
    pairs = collect(values(nb))
    pts = [SVector{N, Float64}(Tuple(p.first)) for p in pairs]
    neg = [p.second < 0 for p in pairs]
    tree = KDTree(pts)
    count = 0
    for tt in transverse
        t = Tuple(tt)
        t in band_lines && continue
        idx, _ = nn(tree, SVector{N, Float64}((n1 ÷ 2, t...)))
        neg[idx] && (count += n1)
    end
    return count
end

"""
    perimeter(ϕ::MeshField)

Measure of the interface `{x : ϕ(x) = 0}` described by the zero level set of `ϕ`. In `N`
dimensions this is the `(N-1)`-dimensional measure — a perimeter (arc length) in 2D, a surface
area in 3D — approximated as `∫ δ(ϕ) ‖∇ϕ‖ dx` with a smoothed Dirac delta `δ`. Contributions
from the domain border are neglected.

```jldoctest
using LevelSetMethods
R = 0.5
S₀ = 2π * R
grid = CartesianGrid((-1, -1), (1, 1), (200, 200))
ϕ = MeshField(x -> sqrt(sum(abs2, x)) - R, grid)
LevelSetMethods.perimeter(ϕ), S₀

# output

(3.1426415491430384, 3.141592653589793)
```
"""
function perimeter(ϕ::MeshField)
    check_real_valued(ϕ)
    # the centered gradient stencil reaches off-grid at the border; supply a default BC if none
    has_boundary_conditions(ϕ) || (ϕ = _add_boundary_conditions(ϕ, LinearExtrapolationBC()))
    δ = meshsize(mesh(ϕ))
    δmin = minimum(δ)
    vol = prod(δ)
    return vol * sum(nodeindices(ϕ)) do I
        smooth_delta(ϕ[I], δmin) * norm(gradient(ϕ, I))
    end
end

"""
    perimeter(nb::NarrowBandMeshField)

Measure of the interface `{x : ϕ(x) = 0}` described by the zero level set, matching
[`perimeter(ϕ::MeshField)`](@ref). The smoothed Dirac delta is supported only within `δmin` of
the interface — entirely inside the band — so the sum runs over the active nodes alone.
"""
function perimeter(nb::NarrowBandMeshField)
    check_real_valued(nb)
    # the centered gradient stencil reaches off-grid at the border; supply a default BC if none
    has_boundary_conditions(nb) || (nb = _add_boundary_conditions(nb, LinearExtrapolationBC()))
    δ = meshsize(mesh(nb))
    δmin = minimum(δ)
    vol = prod(δ)
    return vol * sum(active_nodeindices(nb)) do I
        smooth_delta(nb[I], δmin) * norm(gradient(nb, I))
    end
end

# from "A Variational Level Set Approach to Multiphase Motion"
function smooth_heaviside(x, α)
    if x > α
        return 1.0
    elseif x < -α
        return 0.0
    else
        return 0.5 * (1.0 + x / α + 1.0 / π * sin(π * x / α))
    end
end

function smooth_delta(x, α)
    return abs(x) > α ? 0.0 : 0.5 / α * (1.0 + cos(π * x / α))
end

"""
    curvature(ϕ::AbstractMeshField, I::CartesianIndex)

Mean curvature `κ = ∇ ⋅ (∇ϕ / ‖∇ϕ‖)` of the level set of `ϕ` at grid index `I`, computed from
the [`gradient`](@ref) and [`hessian`](@ref) (centered finite differences) via

    κ = (Δϕ ‖∇ϕ‖² - ∇ϕᵀ Hϕ ∇ϕ) / ‖∇ϕ‖³ .

Returns zero where `∇ϕ` vanishes, since the curvature is undefined there. See the [Wikipedia
article](https://en.wikipedia.org/wiki/Mean_curvature#Implicit_form_of_mean_curvature) on the
implicit form of the mean curvature.
"""
function curvature(ϕ::AbstractMeshField, I::CartesianIndex)
    check_real_valued(ϕ)
    ∇ϕ = gradient(ϕ, I)
    nrmsq = dot(∇ϕ, ∇ϕ)
    nrmsq < eps(valtype(ϕ)) && return zero(nrmsq)
    Hϕ = hessian(ϕ, I)
    Δϕ = tr(Hϕ)
    return (Δϕ * nrmsq - ∇ϕ' * Hϕ * ∇ϕ) / nrmsq^(3 / 2)
end

"""
    gradient(ϕ::AbstractMeshField, I::CartesianIndex)

Gradient `∇ϕ` at grid index `I` from centered finite differences, returned as an `SVector`.
"""
function gradient(ϕ::AbstractMeshField{N}, I::CartesianIndex) where {N}
    check_real_valued(ϕ)
    return SVector{N}(ntuple(dim -> D⁰(ϕ, I, dim), Val(N)))
end

"""
    normal(ϕ::AbstractMeshField, I::CartesianIndex)

Unit exterior normal `n = ∇ϕ / ‖∇ϕ‖` of the level set of `ϕ` at grid index `I`.
"""
function normal(ϕ::AbstractMeshField, I::CartesianIndex)
    check_real_valued(ϕ)
    ∇ϕ = gradient(ϕ, I)
    return ∇ϕ ./ norm(∇ϕ)
end

"""
    hessian(ϕ::AbstractMeshField, I::CartesianIndex)

Hessian `Hϕ = ∇∇ϕ` at grid index `I` from second-order centered finite differences, returned
as a `Symmetric` `SMatrix`.
"""
function hessian(ϕ::AbstractMeshField{N}, I::CartesianIndex) where {N}
    check_real_valued(ϕ)
    H = SMatrix{N, N}(
        ntuple(Val(N * N)) do k
            i = (k - 1) % N + 1
            j = (k - 1) ÷ N + 1
            return i == j ? D2⁰(ϕ, I, i) : D2(ϕ, I, (i, j))
        end,
    )
    return Symmetric(H)
end

# Set operations for level set functions.

"""
    union!(ϕ₁::MeshField, ϕ₂::MeshField)

In-place union of two level sets: `ϕ₁ = min(ϕ₁, ϕ₂)`.
"""
function Base.union!(ϕ₁::MeshField, ϕ₂::MeshField)
    check_real_valued(ϕ₁)
    check_real_valued(ϕ₂)
    v₁, v₂ = values(ϕ₁), values(ϕ₂)
    v₁ .= min.(v₁, v₂)
    return ϕ₁
end

"""
    union(ϕ₁::MeshField, ϕ₂::MeshField)

Union of two level sets: `min(ϕ₁, ϕ₂)`.
"""
Base.union(ϕ₁::MeshField, ϕ₂::MeshField) = union!(copy(ϕ₁), ϕ₂)

"""
    intersect!(ϕ₁::MeshField, ϕ₂::MeshField)

In-place intersection of two level sets: `ϕ₁ = max(ϕ₁, ϕ₂)`.
"""
function Base.intersect!(ϕ₁::MeshField, ϕ₂::MeshField)
    check_real_valued(ϕ₁)
    check_real_valued(ϕ₂)
    v₁, v₂ = values(ϕ₁), values(ϕ₂)
    v₁ .= max.(v₁, v₂)
    return ϕ₁
end

"""
    intersect(ϕ₁::MeshField, ϕ₂::MeshField)

Intersection of two level sets: `max(ϕ₁, ϕ₂)`.
"""
Base.intersect(ϕ₁::MeshField, ϕ₂::MeshField) = intersect!(copy(ϕ₁), ϕ₂)

"""
    complement!(ϕ::MeshField)

In-place complement of a level set: `ϕ = -ϕ`.
"""
function complement!(ϕ::MeshField)
    check_real_valued(ϕ)
    v = values(ϕ)
    v .= -v
    return ϕ
end

"""
    complement(ϕ::MeshField)

Complement of a level set: `-ϕ`.
"""
complement(ϕ::MeshField) = complement!(copy(ϕ))

"""
    setdiff!(ϕ₁::MeshField, ϕ₂::MeshField)

In-place set difference: `ϕ₁ = max(ϕ₁, -ϕ₂)`.
"""
function Base.setdiff!(ϕ₁::MeshField, ϕ₂::MeshField)
    check_real_valued(ϕ₁)
    check_real_valued(ϕ₂)
    v₁, v₂ = values(ϕ₁), values(ϕ₂)
    v₁ .= max.(v₁, -v₂)
    return ϕ₁
end

"""
    setdiff(ϕ₁::MeshField, ϕ₂::MeshField)

Set difference of two level sets: `max(ϕ₁, -ϕ₂)`.
"""
Base.setdiff(ϕ₁::MeshField, ϕ₂::MeshField) = setdiff!(copy(ϕ₁), ϕ₂)
