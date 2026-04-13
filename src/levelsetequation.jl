mutable struct LevelSetEquation
    terms::Tuple{Vararg{LevelSetTerm}}
    integrator::TimeIntegrator
    state::AbstractMeshField
    t::Float64
    reinit::Union{Nothing, NewtonReinitializer}
    log::SimulationLog
end

"""
    LevelSetEquation(; terms, ic, bc, t = 0, integrator = RK2(), reinit = nothing)

Create a level-set equation of the form `ϕₜ + sum(terms) = 0`, where each `t ∈ terms`
is a [`LevelSetTerm`](@ref) and `ic` is the initial condition — either a [`MeshField`](@ref)
for a full-grid discretization or a [`NarrowBandMeshField`](@ref) for a narrow-band discretization.

Calling [`integrate!(eq, tf)`](@ref) will evolve the equation up to time `tf`, modifying
`current_state(eq)` and `current_time(eq)` in place.

Boundary conditions are specified via the required `bc` keyword. If a single
`BoundaryCondition` is provided, it will be applied uniformly to all boundaries of the
domain. To apply different boundary conditions to each boundary, pass a tuple of the form
`(bc_x, bc_y, ...)` with as many elements as dimensions in the domain. If `bc_x` is a
`BoundaryCondition`, it will be applied to both boundaries in the `x` direction. If `bc_x`
is a tuple of two `BoundaryCondition`s, the first will be applied to the left boundary and
the second to the right boundary. The same logic applies to the other dimensions.

The optional parameter `t` specifies the initial time of the simulation, and `integrator` is
the [`TimeIntegrator`](@ref) used to evolve the level-set equation.

Reinitialization is controlled by the `reinit` keyword, which accepts:
- `nothing` (default): no automatic reinitialization.
- an `Int`: reinitialization every `reinit` steps using [`NewtonReinitializer`](@ref) with default settings.
- a [`NewtonReinitializer`](@ref): full control over algorithm parameters and frequency.

```jldoctest; output = true
using LevelSetMethods, StaticArrays
grid = CartesianGrid((-1, -1), (1, 1), (50, 50))    # define the grid
ϕ = MeshField(x -> x[1]^2 + x[2]^2 - 0.5^2, grid)    # initial shape
𝐮 = MeshField(x -> SVector(1, 0), grid)             # advection velocity
terms = (AdvectionTerm(𝐮),)                          # advection term
bc = NeumannBC()                                     # zero-gradient boundary conditions
eq = LevelSetEquation(; terms, ic = ϕ, bc)          # level-set equation

# output

LevelSetEquation
  ├─ equation: ϕₜ + 𝐮 ⋅ ∇ ϕ = 0
  ├─ time:     0.0
  ├─ integrator: RK2 (2nd order TVD Runge-Kutta, Heun's method)
  │  └─ cfl: 0.5
  ├─ reinit:   none
  ├─ state: MeshField on CartesianGrid in ℝ²
  │  ├─ domain:  [-1.0, 1.0] × [-1.0, 1.0]
  │  ├─ nodes:   50 × 50
  │  ├─ spacing: h = (0.04082, 0.04082)
  │  ├─ bc:     Degree 0 extrapolation (all)
  │  ├─ eltype:  Float64
  │  └─ values:  min = -0.2492,  max = 1.75
  ├─ log: SimulationLog (empty)
  ╰─

```
"""
function LevelSetEquation(;
        terms,
        integrator = RK2(),
        ic::AbstractMeshField,
        bc,
        t = 0,
        reinit = nothing,
    )
    N = ndims(ic)
    terms = _normalize_terms(terms)
    reinit = _normalize_reinit(reinit)
    # bc is the authoritative source
    has_boundary_conditions(ic) &&
        @warn "ic already has boundary conditions; these will be overwritten by bc"
    state = _add_boundary_conditions(ic, bc)
    log = SimulationLog(t, terms)
    return LevelSetEquation(terms, integrator, state, t, reinit, log)
end

_normalize_reinit(::Nothing) = nothing
_normalize_reinit(r::NewtonReinitializer) = r
_normalize_reinit(freq::Int) = NewtonReinitializer(; reinit_freq = freq)

function _normalize_terms(terms)
    if isa(terms, LevelSetTerm) # single term
        return (terms,)
    else
        N = length(terms)
        return ntuple(N) do i
            if isa(terms[i], LevelSetTerm)
                return terms[i]
            else
                throw(ArgumentError("invalid term $(terms[i]) on entry $i"))
            end
        end
    end
end

"""
    _embed_show(io, label, obj; indent="  ")

Print the `text/plain` representation of `obj` indented under `label` as a tree branch.
The first line becomes `├─ label: <header>`, and any remaining lines are indented with `│`.
"""
function _embed_show(io, label, obj; indent = "  ")
    str = sprint(show, MIME("text/plain"), obj)
    parts = split(str, '\n')
    println(io, "$(indent)├─ $label: $(parts[1])")
    for line in @view parts[2:end]
        println(io, "$(indent)│$line")
    end
    return
end

function Base.show(io::IO, ::MIME"text/plain", eq::LevelSetEquation)
    pde = "ϕₜ + " * join(sprint.(show, eq.terms), " + ") * " = 0"
    println(io, "LevelSetEquation")
    println(io, "  ├─ equation: $pde")
    println(io, "  ├─ time:     $(eq.t)")
    _embed_show(io, "integrator", eq.integrator)
    if !isnothing(eq.reinit)
        _embed_show(io, "reinit", eq.reinit)
    else
        println(io, "  ├─ reinit:   none")
    end
    _embed_show(io, "state", eq.state)
    _embed_show(io, "log", eq.log)
    print(io, "  ╰─")
    return io
end

"""
    reset_log!(eq::LevelSetEquation)

Clear the simulation log of `eq`, resetting step count and all timing records.
"""
reset_log!(eq::LevelSetEquation) = reset_log!(eq.log, eq.t)

# keep the compact (non-MIME) method for embedding in error messages etc.
function Base.show(io::IO, eq::LevelSetEquation)
    pde = "ϕₜ + " * join(sprint.(show, eq.terms), " + ") * " = 0"
    print(io, "LevelSetEquation($pde, t=$(eq.t))")
    return io
end

"""
    current_state(eq::LevelSetEquation)

Return the current state of the level-set equation (a [`MeshField`](@ref)).
"""
current_state(ls::LevelSetEquation) = ls.state

# Allow current_state on a bare AbstractMeshField so that plotting recipes work uniformly.
current_state(ϕ::AbstractMeshField) = ϕ

"""
    current_time(eq::LevelSetEquation)

Return the current time of the simulation.
"""
current_time(ls::LevelSetEquation) = ls.t

"""
    time_integrator(eq::LevelSetEquation)

Return the [`TimeIntegrator`](@ref) of the equation.
"""
time_integrator(ls) = ls.integrator

"""
    terms(eq::LevelSetEquation)

Return the tuple of [`LevelSetTerm`](@ref)s of the equation.
"""
terms(ls) = ls.terms

"""
    boundary_conditions(eq::LevelSetEquation)

Return the boundary conditions of the equation.
"""
boundary_conditions(ls::LevelSetEquation) = boundary_conditions(ls.state)

"""
    mesh(eq::LevelSetEquation)

Return the underlying [`CartesianGrid`](@ref) of the equation.
"""
mesh(ls::LevelSetEquation) = mesh(ls.state)

"""
    reinitializer(eq::LevelSetEquation)

Return the [`NewtonReinitializer`](@ref) (if any) attached to the equation.
"""
reinitializer(ls::LevelSetEquation) = ls.reinit

# Convenience delegations to current state
volume(eq::LevelSetEquation) = volume(current_state(eq))
perimeter(eq::LevelSetEquation) = perimeter(current_state(eq))

"""
    reinitialize!(eq::LevelSetEquation)

Reinitialize the current state of the level-set equation using its attached [`NewtonReinitializer`](@ref).
"""
function reinitialize!(eq::LevelSetEquation)
    r = reinitializer(eq)
    isnothing(r) && throw(ArgumentError("no reinitializer attached to the equation"))
    reinitialize!(current_state(eq), r)
    return eq
end

"""
    integrate!(ls::LevelSetEquation,tf,Δt=Inf)

Integrate the [`LevelSetEquation`](@ref) `ls` up to time `tf`,
mutating the `levelset` and `current_time` of the object `ls` in the
process.

An optional parameter `Δt` can be passed to specify a maximum time-step
allowed for the integration. Note that the internal time-steps taken to evolve
the level-set up to `tf` may be smaller than `Δt` due to stability reasons
related to the `terms` and `integrator` employed.
"""
function integrate!(ls::LevelSetEquation, tf, Δt = Inf)
    tc = current_time(ls)
    tf >= tc || throw(ArgumentError("final time $tf must be ≥ initial time $tc: the level-set equation cannot be solved back in time"))
    # append boundary conditions for integration
    ϕ = current_state(ls)
    integrator = time_integrator(ls)
    reinit = reinitializer(ls)
    # dynamic dispatch. Should not be a problem provided enough computation is
    # done inside of the function below
    _integrate!(ϕ, integrator, ls.terms, reinit, tc, tf, Δt, ls.log)
    ls.t = tf
    return ls
end
