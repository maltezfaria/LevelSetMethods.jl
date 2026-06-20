mutable struct LevelSetEquation
    terms::Tuple{Vararg{LevelSetTerm}}
    integrator::TimeIntegrator
    state::AbstractMeshField
    t::Float64
end

"""
    LevelSetEquation(; terms, ic, bc, t = 0, integrator = RK2())

Create a level-set equation of the form `ϕₜ + sum(terms) = 0`, where each `t ∈ terms` is a
[`LevelSetTerm`](@ref) and `ic` is the initial condition — either a [`MeshField`](@ref) for
a full-grid discretization or a [`NarrowBandMeshField`](@ref) for a narrow-band
discretization.

Calling [`integrate!(eq, tf)`](@ref) will evolve the equation up to time `tf`, modifying
`current_state(eq)` and `current_time(eq)` in place.

Boundary conditions come from the `bc` keyword, or from `ic` if it already carries them
(passing both warns and `bc` wins); it is an error if neither supplies them. If a single
`BoundaryCondition` is provided, it will be applied uniformly to all boundaries of the
domain. To apply different boundary conditions to each boundary, pass a tuple of the form
`(bc_x, bc_y, ...)` with as many elements as dimensions in the domain. If `bc_x` is a
`BoundaryCondition`, it will be applied to both boundaries in the `x` direction. If `bc_x`
is a tuple of two `BoundaryCondition`s, the first will be applied to the left boundary and
the second to the right boundary. The same logic applies to the other dimensions.

The optional parameter `t` specifies the initial time of the simulation, and `integrator` is
the [`TimeIntegrator`](@ref) used to evolve the level-set equation.

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
  ├─ state: MeshField on CartesianGrid in ℝ²
  │  ├─ domain:  [-1.0, 1.0] × [-1.0, 1.0]
  │  ├─ nodes:   50 × 50
  │  ├─ spacing: h = (0.04082, 0.04082)
  │  ├─ bc:     Neumann (all)
  │  ├─ valtype: Float64
  │  └─ values:  min = -0.2492,  max = 1.75
  ╰─
```
"""
function LevelSetEquation(;
        terms,
        integrator = RK2(),
        ic::AbstractMeshField,
        bc = nothing,
        t = 0,
    )
    terms = _normalize_terms(terms)
    state = if isnothing(bc)
        has_boundary_conditions(ic) ||
            throw(ArgumentError("no boundary conditions: pass `bc` or build `ic` with one"))
        ic
    else
        has_boundary_conditions(ic) &&
            @warn "ic already has boundary conditions; these will be overwritten by bc"
        _add_boundary_conditions(ic, bc)
    end
    return LevelSetEquation(terms, integrator, state, t)
end

_normalize_terms(t::LevelSetTerm) = (t,)
_normalize_terms(t::Tuple{Vararg{LevelSetTerm}}) = t
_normalize_terms(t) =
    throw(ArgumentError("terms must be a LevelSetTerm or a tuple of them, got $(typeof(t))"))

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
    _embed_show(io, "state", eq.state)
    print(io, "  ╰─")
    return io
end

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

# Convenience delegations to current state
volume(eq::LevelSetEquation) = volume(current_state(eq))
perimeter(eq::LevelSetEquation) = perimeter(current_state(eq))

"""
    integrate!(ls::LevelSetEquation, tf, Δt = Inf; prehook = identity, posthook = identity)

Integrate the [`LevelSetEquation`](@ref) `ls` up to time `tf`,
mutating the `levelset` and `current_time` of the object `ls` in the
process.

An optional parameter `Δt` can be passed to specify a maximum time-step
allowed for the integration. Note that the internal time-steps taken to evolve
the level-set up to `tf` may be smaller than `Δt` due to stability reasons
related to the `terms` and `integrator` employed.

`prehook` and `posthook` are functions called once per *accepted* time step (as opposed to
once per Runge-Kutta stage): `prehook(ls)` runs at the start of the step, before the state is
advanced, and `posthook(ls)` runs after the step has been committed. Both receive `ls` with
`current_state(ls)` and `current_time(ls)` synced to that point, and may mutate
`current_state(ls)` (mutations are carried into the step). Their return values are ignored;
the default `identity` is a no-op.

Reinitialization of the level-set to a signed distance function is not built in by default,
but can be easily added by creating a `posthook`: pass `posthook = eq ->
reinitialize!(current_state(eq))` (see [`reinitialize!`](@ref)) to reinitialize after every
step, or gate the call on any criterion (elapsed time, `|∇ϕ|` drift, a step counter closed
over by the hook). The integrator never reinitializes on its own — keeping a signed distance
function is entirely the caller's choice.
"""
function integrate!(ls::LevelSetEquation, tf, Δt = Inf; prehook = identity, posthook = identity)
    tc = current_time(ls)
    tf >= tc || throw(ArgumentError("final time $tf must be ≥ initial time $tc: the level-set equation cannot be solved back in time"))
    # dynamic dispatch on the integrator. Should not be a problem provided enough
    # computation is done inside of the function below. `_integrate!` advances `ls.t`
    # itself (it may stop before `tf` if a hook requests early termination). Band maintenance
    # (`update_band!`) runs inside the step loop, after the advance and before the posthook.
    _integrate!(ls, current_state(ls), time_integrator(ls), ls.terms, tc, tf, Δt, prehook, posthook)
    return ls
end
