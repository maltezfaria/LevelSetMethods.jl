```@meta
CurrentModule = LevelSetMethods
```

# LevelSetMethods

Documentation for [LevelSetMethods](https://github.com/maltezfaria/LevelSetMethods.jl).

## Installation

LevelSetMethods.jl is not yet registered in the Julia package registry. To install it, run
the following command on a Julia REPL:

```julia
using Pkg; Pkg.add("https://github.com/maltezfaria/LevelSetMethods.jl")
```

This will install the latest tagged version of the package and its dependencies.

## Overview

This package defines a [`LevelSetEquation`](@ref) type that can be used to solve partial
differential equations of the form

```math
\phi_t + \underbrace{\boldsymbol{u} \cdot \nabla \phi}_{\substack{\text{advection} \\ \text{term}}} + \underbrace{v |\nabla \phi|}_{\substack{\text{normal} \\ \text{term}}} + \underbrace{b \kappa |\nabla \phi|}_{\substack{\text{curvature} \\ \text{term}}} + \underbrace{\text{sign}(\phi)(|\nabla \phi| - 1)}_{\substack{\text{reinitialization}\\ \text{term}}} = 0
```

where

- ``\phi : \mathbb{R}^d \to \mathbb{R}`` is the level set function
- ``\boldsymbol{u} :\mathbb{R}^d \to \mathbb{R}^d`` is a given (external) velocity field
- ``v : \mathbb{R}^d \to \mathbb{R}`` is a normal speed
- ``b : \mathbb{R}^d \to \mathbb{R}`` is a function that multiplies the curvature ``\kappa =
  \nabla \cdot (\nabla \phi / |\nabla \phi|)``

Here is how it looks in practice to create a simple `LevelSetEquation`:

```@example ls-intro
using LevelSetMethods, StaticArrays
grid = CartesianGrid((-1, -1), (1, 1), (100, 100))
# ϕ    = LevelSet(x -> sqrt(2*x[1]^2 + x[2]^2) - 1/2, grid)
ϕ    = LevelSetMethods.dumbbell(grid)
𝐮    = MeshField(x -> SVector(-x[2], x[1]), grid)
eq   = LevelSetEquation(;
  terms = (AdvectionTerm(𝐮),),
  levelset = ϕ,
  bc = PeriodicBC()
)
```

You can easily plot the current state of your level set equation using the `plot` function
from [Makie](https://docs.makie.org):

```@example ls-intro
using GLMakie # loads the MakieExt from LevelSetMethods
LevelSetMethods.set_makie_theme!() # optional theme customization
plot(eq)
```

To step it in time, we can use the [`integrate!`](@ref) function:

```@example ls-intro
integrate!(eq, 1)
```

This will advance the solution up to `t = 1`, modifying `ϕ` in the process:

```@example ls-intro
plot(eq)
```

Creating an animation can be achieved by calling `integrate!` in a loop and saving the
results to a file:

```@example ls-intro
using GLMakie
theme = LevelSetMethods.makie_theme()
anim = with_theme(theme) do
    obs = Observable(eq)
    fig = Figure()
    ax = Axis(fig[1, 1])
    plot!(ax, obs)
    framerate = 30
    t0 = current_time(eq)
    tf = t0 + π
    timestamps = range(t0, tf; step = 1 / framerate)
    record(fig, joinpath(@__DIR__, "ls_intro.gif"), timestamps) do t_
        integrate!(eq, t_)
        return obs[] = eq
    end
end
```

Here is what the `.gif` file looks like:

![Dumbbell](ls_intro.gif)

For more interesting applications and advanced usage, see the [examples section](@ref
examples)!

!!! note "Other resources"
    There is an almost one-to-one correspondance between each of the [`LevelSetTerm`](@ref)s
    described above and individual chapters of the book by Osher and Fedwick on level set
    methods [osher2003level](@cite), so users interested in digging deeper into the
    theory/algorithms are encourage to consult that refenrence. We also drew some
    inspiration from the great Matlab library `ToolboxLS` by Ian Mitchell
    [mitchell2007toolbox](@cite).

## Going further

As illustrated above, the `LevelSetEquation` type is the main structure of this package.
Becoming familiar with its fields and methods is a good starting point to use the package:

```@docs
LevelSetEquation
```

To learn more about the package, you should also check out the following sections:

- The section on [terms](@ref terms) for a detailed description of each term and their
  corresponding customizations
- The section on [time integrators](@ref time-integrators) for a description of the
  available time integrators and how to use them
- The section on [boundary conditions](@ref boundary-conditions) for a description of the
  available boundary conditions and how to use them

Finally, the [examples](@ref examples) section contains a list of examples that demonstrate some
hopefully cool applications.

## Bibliography

```@bibliography
```
