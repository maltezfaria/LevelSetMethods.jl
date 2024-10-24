# [Advection example](@id contributing)

This example illustrates how to solve the advection equation using the `LevelSetMethods` package:

```math
\phi_t + \mathbf{v} \cdot \nabla \phi = 0
```

where $\phi$ is the level set function and $\mathbf{v}$ is a given velocity field (for now it has to be constant in time).

Let's start by defining the grid

```@example advection
using LevelSetMethods
a = (-1,-1)
b = (1,1)
n = (100, 100)
grid = CartesianGrid(a, b, n)
```

Next we define the velocity field on the grid

```@example advection
𝐮 = MeshField(grid) do (x, y)
    return SVector(1, 0)
end
```

With `𝐮` defined, we can now create the advection term

```@example advection
advection_term = AdvectionTerm(; velocity = 𝐮)
```

We can now initialize the level set function

```@example advection
ϕ = LevelSet(grid) do (x, y)
    return 0.5^2 - x^2 - y^2
end
```

We now have all ingredients to create our level set equation:

```@example advection
b = deepcopy(ϕ)
integrator = ForwardEuler(0.5)
bc = PeriodicBC()
eq = LevelSetEquation(; terms = (advection_term,), integrator, levelset = ϕ, t = 0, bc)
```

Finally, we can integrate the equation in time and visualize the results. For that we will need to have a `Makie` backend installed. If you don't have it, you can install it by running e.g. `]add GLMakie`.

```@example advection
using GLMakie
theme = LevelSetMethods.makie_theme() # a custom theme for the plot
anim = with_theme(theme) do
    eq.t = 0
    obs = Observable(eq)
    fig = Figure()
    ax = Axis(fig[1, 1])
    plot!(ax, obs)
    framerate = 30
    tf = 2
    timestamps = range(0, tf, tf * framerate)
    record(fig, joinpath(@__DIR__,"advection.gif"), timestamps) do t_
        integrate!(eq, t_)
        return obs[] = eq
    end
end
```

![Advection](advection.gif)
