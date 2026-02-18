"""
    extend_along_normals!(F, ϕ::LevelSet;
                          nb_iters = 50,
                          cfl = 0.45,
                          frozen = nothing,
                          interface_band = 1.5,
                          min_norm = 1e-14)

Extend a scalar speed field `F` away from the interface of `ϕ` by solving in pseudo-time
`∂τF + sign(ϕ) n⋅∇F = 0`, with `n = ∇ϕ / |∇ϕ|`.

The equation is discretized with first-order upwind derivatives. The update preserves
`frozen` nodes (Dirichlet constraint). If `frozen` is not provided, a mask is built from
the interface band `abs(ϕ) <= interface_band * Δ`, where `Δ = minimum(meshsize(ϕ))`.

`F` can be an `AbstractArray` (same size as `ϕ`) or a `MeshField` on the same mesh.

Reference from [Peng et al. 1999]
"""
function extend_along_normals!(
        F::AbstractArray{T, N},
        ϕ::LevelSet;
        nb_iters::Integer = 50,
        cfl::Real = 0.45,
        frozen = nothing,
        interface_band::Real = 1.5,
        min_norm::Real = 1.0e-14,
    ) where {T <: Real, N}
    size(F) == size(values(ϕ)) ||
        throw(ArgumentError("F and ϕ must have the same size"))
    nb_iters >= 0 || throw(ArgumentError("nb_iters must be non-negative"))
    cfl > 0 || throw(ArgumentError("cfl must be strictly positive"))
    interface_band >= 0 || throw(ArgumentError("interface_band must be non-negative"))
    min_norm >= 0 || throw(ArgumentError("min_norm must be non-negative"))
    T <: AbstractFloat ||
        throw(ArgumentError("F must have floating-point element type"))

    bc = if has_boundary_conditions(ϕ)
        boundary_conditions(ϕ)
    else
        ntuple(_ -> (NeumannGradientBC(), NeumannGradientBC()), N)
    end
    ϕw = has_boundary_conditions(ϕ) ? ϕ : add_boundary_conditions(ϕ, bc)
    Fw = MeshField(F, mesh(ϕ), bc)

    Δ = minimum(meshsize(ϕ))
    τ = cfl * Δ
    frozen_mask = _normalize_frozen_mask(frozen, ϕw, interface_band, Δ)
    signed_normals = _signed_normal_components(ϕw, Δ, min_norm)

    F_new = similar(F)
    for _ in 1:nb_iters
        for I in eachindex(ϕw)
            if frozen_mask[I]
                F_new[I] = Fw[I]
                continue
            end
            advection = zero(eltype(F_new))
            for dim in 1:N
                a = signed_normals[dim][I]
                dF = a > 0 ? D⁻(Fw, I, dim) : D⁺(Fw, I, dim)
                advection += a * dF
            end
            F_new[I] = Fw[I] - τ * advection
        end
        copy!(F, F_new)
    end
    return F
end

function extend_along_normals!(F::MeshField, ϕ::LevelSet; kwargs...)
    mesh(F) == mesh(ϕ) ||
        throw(ArgumentError("F and ϕ must be defined on the same mesh"))
    extend_along_normals!(values(F), ϕ; kwargs...)
    return F
end

function _normalize_frozen_mask(frozen, ϕ::LevelSet, interface_band, Δ)
    vals = values(ϕ)
    if isnothing(frozen)
        return abs.(vals) .<= interface_band * Δ
    end
    if frozen isa MeshField
        mesh(frozen) == mesh(ϕ) ||
            throw(ArgumentError("frozen mask must be on the same mesh as ϕ"))
        frozen = values(frozen)
    end
    size(frozen) == size(vals) ||
        throw(ArgumentError("frozen mask must have the same size as ϕ"))
    eltype(frozen) <: Bool ||
        throw(ArgumentError("frozen mask must contain Bool values"))
    return BitArray(frozen)
end

function _signed_normal_components(ϕ::LevelSet, Δ, min_norm)
    N = dimension(ϕ)
    T = float(eltype(values(ϕ)))
    components = ntuple(_ -> Array{T}(undef, size(values(ϕ))), N)
    min_norm² = min_norm^2
    for I in eachindex(ϕ)
        ∇ϕ = ntuple(dim -> T(D⁰(ϕ, I, dim)), N)
        norm² = sum(abs2, ∇ϕ)
        if norm² <= min_norm²
            for dim in 1:N
                components[dim][I] = zero(T)
            end
            continue
        end
        invnorm = inv(sqrt(norm²))
        S = ϕ[I] / sqrt(ϕ[I]^2 + Δ^2)
        for dim in 1:N
            components[dim][I] = S * ∇ϕ[dim] * invnorm
        end
    end
    return components
end
