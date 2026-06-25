```@meta
CurrentModule = LevelSetMethods
```

# [Time integration](@id time-integrators)

A [`LevelSetEquation`](@ref) is advanced in time by a *time integrator*, chosen with the
`integrator` keyword (see [Level-set equation](@ref levelset-equation)). The following
options are available:

```@example
using LevelSetMethods
using InteractiveUtils # hide
subtypes(LevelSetMethods.TimeIntegrator)
```

[`ForwardEuler`](@ref), [`RK2`](@ref), and [`RK3`](@ref) are explicit schemes, and therefore
a sufficiently small time step — dependent on the [`LevelSetTerm`](@ref)s being used — is
required to ensure stability. We recommend the second-order [`RK2`](@ref) scheme (the
default) for most applications. The package also provides [`SemiImplicitI2OE`](@ref), a
semi-implicit scheme ([mikula2010new](@cite)) for advection equations.

## The CFL number

To stay stable, an explicit scheme cannot take an arbitrarily large time step: the step is
bounded by a
[CFL condition](https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition)
set by the active terms and the mesh spacing. [`integrate!`](@ref) enforces this for you,
computing the largest stable `Δt` at each step. Each integrator scales that limit by a `cfl`
*safety factor*, passed at construction; a smaller factor takes smaller, more accurate (and
more expensive) steps. It defaults to `0.5` for the explicit schemes:

```@example
using LevelSetMethods
RK2(; cfl = 0.25)   # half the default step size
```

You can also cap the step from above by passing a maximum time step as the `Δt` argument of
[`integrate!`](@ref); the step actually taken is then the smaller of that cap and the CFL
limit.

## Comparing the integrators

To compare the schemes quantitatively, we rotate a dumbbell — built with the CSG operators
from the [geometry](@ref geometry) page — through one full revolution. Rigid rotation is
area-preserving, so the exact final area equals the initial one; the leftover change in
enclosed area then measures each scheme's accumulated error. Running every integrator at the
same (default) CFL:

```@example integrators
using LevelSetMethods, Markdown, Printf

grid = CartesianGrid((-1, -1), (1, 1), (64, 64))
disk(c) = MeshField(x -> hypot((x .- c)...) - 0.25, grid)
bar     = MeshField(x -> maximum(abs.(x) .- (1.0, 0.2) ./ 2), grid)
ϕ₀ = disk((-0.5, 0.0)) ∪ disk((0.5, 0.0)) ∪ bar   # a dumbbell
𝐮  = (x, t) -> (-x[2], x[1])
V₀ = LevelSetMethods.volume(ϕ₀)                   # exact final area (rotation preserves it)

results = map(("ForwardEuler" => ForwardEuler(), "RK2" => RK2(), "RK3" => RK3())) do (name, integrator)
    eq = LevelSetEquation(; terms = AdvectionTerm(𝐮), ic = ϕ₀, bc = NeumannBC(), integrator)
    integrate!(eq, 2π)
    V = LevelSetMethods.volume(eq)
    (; name, V, err = abs(V - V₀) / V₀)
end

errs = Dict(r.name => r.err for r in results)                                                # hide
@assert errs["ForwardEuler"] > 4 * errs["RK2"] "ForwardEuler should lose several × more area than RK2"  # hide
@assert errs["ForwardEuler"] > 4 * errs["RK3"] "ForwardEuler should lose several × more area than RK3"  # hide
@assert abs(errs["RK2"] - errs["RK3"]) < 0.1 * errs["RK2"] "RK2 and RK3 should be nearly indistinguishable"  # hide

io = IOBuffer()
println(io, "| Integrator | Final area | Area error |")
println(io, "|:--|--:|--:|")
for r in results
    @printf(io, "| %s | %.4f | %.2f%% |\n", r.name, r.V, 100 * r.err)
end
Markdown.parse(String(take!(io)))
```

The first-order [`ForwardEuler`](@ref) loses several times more area than the two higher-order
schemes. [`RK2`](@ref) and [`RK3`](@ref) are almost indistinguishable here: once the temporal
error drops below the fifth-order WENO5 spatial error, refining the time integration further
buys little. This is why [`RK2`](@ref) — the default — is the right balance for most problems;
it removes the first-order temporal error of `ForwardEuler` without the extra stage cost of
[`RK3`](@ref).

## The semi-implicit scheme

[`SemiImplicitI2OE`](@ref) is a finite-volume scheme designed specifically for advection.
Being semi-implicit, it remains stable at much larger time steps than the explicit schemes —
its `cfl` factor defaults to `2.0` — so it reaches a given final time in far fewer steps. On
the rotation above, one revolution costs only 198 steps versus 792 for the explicit schemes
at their default CFL, trading some per-step accuracy for a substantially lower step count.

It comes with restrictions: it requires a full-grid [`MeshField`](@ref) (not a narrow band)
and exactly one [`AdvectionTerm`](@ref). Using it is otherwise a one-keyword change:

```@example integrators
eq = LevelSetEquation(;
    terms      = AdvectionTerm(𝐮),
    ic         = ϕ₀,
    bc         = NeumannBC(),
    integrator = SemiImplicitI2OE(),
)
integrate!(eq, 2π)
nothing # hide
```
