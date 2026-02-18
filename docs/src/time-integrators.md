# [Time integration](@id time-integrators)

The following options are available for time integration:

```@example
using LevelSetMethods
using InteractiveUtils # hide
subtypes(LevelSetMethods.TimeIntegrator)
```

`ForwardEuler`, `RK2`, and `RK3` are explicit schemes, and therefore a sufficiently small
time step, dependant on the `LevelSetTerm` being used, is required to ensure stability. We
recommend using the second order [`RK2`](@ref) scheme for most applications.

The package also provides [`SemiImplicitI2OE`](@ref), a semi-implicit scheme (Mikula et al.)
for advection equations.