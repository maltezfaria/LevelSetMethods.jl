# [Volume conservation](@id volume-conservation)

In this example, we examine the volume conservation properties of different
time-stepping methods. We advect a [Zalesak disk](@ref zalesak) under the
solid-body rotation velocity field ``\mathbf{u}(\mathbf{x}) = (-x_2, x_1)``,
which is divergence-free. The exact solution therefore conserves the enclosed
volume

```math
V(t) = \int_{\{\phi(\mathbf{x},t) \leq 0\}} d\mathbf{x}
```

exactly, and any deviation from the initial volume ``V_0`` is a measure of
numerical error.

## Setup

```@setup volume_conservation_example
using LevelSetMethods
using GLMakie
LevelSetMethods.set_makie_theme!()
```

```@example volume_conservation_example
grid = CartesianGrid((-1.5, -1.5), (1.5, 1.5), (100, 100))
center = (-0.75, 0)
radius = 0.5
h = 1.0
w = 0.2
disk = LevelSetMethods.circle(grid; center, radius)
rec = LevelSetMethods.rectangle(grid; center = center .- (0, radius), width = (w, h))
ϕ₀ = setdiff(disk, rec)
V₀ = LevelSetMethods.volume(ϕ₀)
tf = 2π   # one full revolution
nframes = 50
timestamps = range(0, tf; length = nframes + 1)

# Evolve the level set with a given integrator and record volume at each frame
function track_volume(integrator)
    ϕ = deepcopy(ϕ₀)
    eq = LevelSetEquation(;
        ic = ϕ,
        terms = AdvectionTerm((x, t) -> (-x[2], x[1])),
        bc = NeumannBC(),
        integrator,
    )
    V = zeros(length(timestamps))
    V[1] = V₀
    for (i, t) in enumerate(timestamps[2:end])
        integrate!(eq, t)
        V[i + 1] = LevelSetMethods.volume(current_state(eq))
    end
    return (V .- V₀) ./ V₀
end
```

## Comparing time-stepping methods

We run each available integrator at its default CFL number and plot the relative
volume error ``(V(t) - V_0)/V_0`` over one full revolution:

```@example volume_conservation_example
integrators = [
    "ForwardEuler (CFL=0.5)"     => ForwardEuler(),
    "RK2 (CFL=0.5)"              => RK2(),
    "RK3 (CFL=0.5)"              => RK3(),
    "SemiImplicitI2OE (CFL=2.0)" => SemiImplicitI2OE(),
]

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "t", ylabel = "(V(t) - V₀) / V₀",
          title = "Volume conservation: comparing integrators")
for (label, integrator) in integrators
    lines!(ax, collect(timestamps), track_volume(integrator); label)
end
axislegend(ax; position = :lb)
fig
```

Higher-order schemes generally conserve volume better at the same CFL number. The
semi-implicit scheme takes much larger time steps (CFL=2.0 vs 0.5) yet remains
stable, trading some accuracy per step for fewer total steps.

## Effect of time-step size

To isolate the role of the time step, we fix the integrator to `RK2` and vary the
CFL number:

```@example volume_conservation_example
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "t", ylabel = "(V(t) - V₀) / V₀",
          title = "Volume conservation: varying CFL (RK2)")
for α in [0.1, 0.25, 0.5]
    lines!(ax, collect(timestamps), track_volume(RK2(; cfl = α)); label = "CFL = $α")
end
axislegend(ax; position = :lb)
fig
```

As expected, reducing the time step improves volume conservation at the cost of
more integration steps.
