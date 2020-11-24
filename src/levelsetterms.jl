"""
    abstract type LevelSetTerm

A typical term in a level-set evolution equation.
"""
abstract type LevelSetTerm end

"""
    compute_terms(terms,ϕ,bc)
    compute_terms!(buffer,terms,ϕ,bc)

Given a tuple `terms` containing `LevSetTerm`s, compute the contribution of all
these terms to the level set equation. A `buffer` can be passed for allocation
purposes, so that `compute_terms!` is does not allocate any (dynamic) memory.
"""
function compute_terms!(buffer::MeshField,terms::Tuple,ϕ::MeshField,bc::BoundaryCondition)
    @assert mesh(ϕ) == mesh(buffer)
    grid = mesh(ϕ)
    # update ϕ with prescribed bc before entering the loop
    applybc!(ϕ,bc)
    for I in interior_indices(grid,bc)
        map(terms) do term
            _update_term!(buffer,term,ϕ,I)    
        end    
    end   
    return buffer     
end    
compute_terms(terms::Tuple,ϕ::MeshField,bc::BoundaryCondition) = compute_terms!(zero(ϕ),terms::Tuple,ϕ::MeshField,bc::BoundaryCondition)

"""
    struct AdvectionTerm{V,M} <: LevelSetTerm

Level-set advection term representing  `𝐯 ⋅ ∇ϕ`.
"""
Base.@kwdef struct AdvectionTerm{V,M} <: LevelSetTerm
    velocity::MeshField{V,M}
    scheme::Symbol = :upwind
end
velocity(adv::AdvectionTerm) = adv.velocity
boundary_condition(adv)      = adv.bc

function _update_term!(buffer,term::AdvectionTerm,ϕ,I)
    𝐮 = velocity(term)
    N = dimension(ϕ)
    # for each dimension, compute the upwind derivative and multiply by the
    # velocity and add to buffer
    for dim in 1:N
        v = 𝐮[I][dim]
        if v > 0
            buffer[I] += v*D⁻(ϕ,I,dim)
        else
            buffer[I] += v*D⁺(ϕ,I,dim)
        end
    end
    return buffer
end

"""
    struct CurvatureTerm{V,M} <: LevelSetTerm

Level-set curvature term representing `bκ|∇ϕ|`, where `κ = ∇ ⋅ (∇ϕ/|∇ϕ|) ` is
the curvature.
"""
struct CurvatureTerm{V,M} <: LevelSetTerm
    b::MeshField{V,M}
end
coefficient(cterm::CurvatureTerm) = cterm.b

function _update_term!(buffer,term::CurvatureTerm,ϕ,I)
    b = coefficient(term)
    N = dimension(ϕ)
    κ = curvature(ϕ,I)
    # compute |∇ϕ|
    ϕ² = sum(1:N) do dim
        D⁰(ϕ,I,dim)^2
    end
    buffer[I] += b[I]*κ*sqrt(ϕ²)
    return buffer
end

function curvature(ϕ::LevelSet,I)
    N = dimension(ϕ)
    if N == 2
        ϕx  = D⁰(ϕ,I,1)
        ϕy  = D⁰(ϕ,I,2)
        ϕxx = D2⁰(ϕ,I,1)
        ϕyy = D2⁰(ϕ,I,2)
        ϕxy = D2(ϕ,I,(2,1))
        κ   = (ϕxx*(ϕy)^2 - 2*ϕy*ϕx*ϕxy + ϕyy*ϕx^2) / (ϕx^2 + ϕy^2)^(3/2)
        return κ
    elseif N == 3
        ϕx  = D⁰(ϕ,I,1)
        ϕy  = D⁰(ϕ,I,2)
        ϕz  = D⁰(ϕ,I,3)
        ϕxx = D2⁰(ϕ,I,1)
        ϕyy = D2⁰(ϕ,I,2)
        ϕzz = D2⁰(ϕ,I,3)
        ϕxy = D2(ϕ,I,(2,1))
        ϕxz = D2(ϕ,I,(3,1))
        # TODO: test + simplify this
        κ   = (ϕxx*(ϕy)^2 - 2*ϕy*ϕx*ϕxy + ϕyy*ϕx^2 + ϕx^2*ϕzz - 2*ϕx*ϕz*ϕxz + ϕz^2*ϕxx + ϕy^2*ϕzz - 2*ϕy*ϕz*ϕyz + ϕz^2*ϕyy) / (ϕx^2 + ϕy^2)^3/2
        return κ
    else
        notimplemented()
    end
end

"""
    struct NormalAdvectionTerm{V,M} <: LevelSetTerm

Level-set advection term representing  `v |∇ϕ|`. This `LevelSetTerm` should be
used for internally generated velocity fields; for externally generated
velocities you may use `AdvectionTerm` instead.
"""
@Base.kwdef struct NormalAdvectionTerm{V,M} <: LevelSetTerm
    speed::MeshField{V,M}
end
speed(adv::NormalAdvectionTerm) = adv.speed

function _update_term!(buffer,term::NormalAdvectionTerm,ϕ,I)
    u = speed(term)
    N = dimension(ϕ)
    v = u[I]
    mA0² = 0.0
    mB0² = 0.0
    for dim in 1:N
        h = meshsize(ϕ,dim)

        # eq. (6.22-6.27) generalized for any dimensions
        A = D⁻(ϕ,I,dim) + 0.5 * h * limiter(D2⁻⁻(ϕ,I,dim), D2⁰(ϕ,I,dim))
        B = D⁺(ϕ,I,dim) - 0.5 * h * limiter(D2⁺⁺(ϕ,I,dim), D2⁰(ϕ,I,dim))

        if v > 0.0
            mA0² += positive(A)^2
            mB0² += negative(B)^2
        else
            mA0² += negative(A)^2
            mB0² += positive(B)^2
        end
    end

    ∇ = sqrt(mA0² + mB0²)
    buffer[I] += ∇ * v

    return buffer
end

@inline positive(x) = x > zero(x) ? x : zero(x)
@inline negative(x) = x < zero(x) ? x : zero(x)

function _compute_cfl(buffer,term::NormalAdvectionTerm,ϕ)
    mind = minimum(meshsize(ϕ))
    norminf = maximum(abs.(speed(term)))
    return 0.5 * mind / norminf
end

# eq. (6.20-6.21)
function g(x, y)
    tmp = zero(x)
    if x > zero(x); tmp += x*x; end
    if y < zero(x); tmp += y*y; end
    return sqrt(tmp)
end

# eq. (6.28)
function limiter(x, y)
    x*y < zero(x) || return zero(x)
    return abs(x) <= abs(y) ? x : y
end
