```@meta
CurrentModule = LevelSetMethods
```

# [Interpolation](@id interpolation)

A [`AbstractMeshField`](@ref) stores values at grid nodes and nothing in between. An
[`InterpolatedField`](@ref) equips it with piecewise-polynomial interpolation, turning it into
a continuous, callable function you can evaluate — and differentiate — anywhere in the domain.
It is also the engine underneath [reinitialization](@ref signed-distance),
[`NewtonSDF`](@ref LevelSetMethods.NewtonSDF), and the [ImplicitIntegration
quadrature](@ref extension-implicit-integration), though most users meet it through the
off-grid queries below.

## Building an interpolant

Wrap a [`AbstractMeshField`](@ref) in an [`InterpolatedField`](@ref), choosing the polynomial order.
The result is callable and evaluates the local interpolant at the query point — coordinates
may be passed as separate scalars or as a single point-like object such as an `SVector`:

```@example interpolation
using LevelSetMethods
star = MeshField(CartesianGrid((-2.0, -2.0), (2.0, 2.0), (64, 64))) do x   # see the geometry page
    r, θ = hypot(x...), atan(x[2], x[1])
    return r - (1 + 0.25 * cos(5θ))
end
ϕ = InterpolatedField(star, 3)
ϕ(0.5, 0.5)   # evaluate at an arbitrary point inside the grid
```

The construction is identical in any dimension.

## Derivatives

The interpolant differentiates its local polynomial patch *exactly* (via `ForwardDiff`), so
[`LevelSetMethods.gradient`](@ref) and [`LevelSetMethods.hessian`](@ref) return analytic
derivatives rather than further finite-difference approximations. When you need the value
alongside its derivatives, the fused `value_and_gradient` and `value_gradient_hessian` compute
them together in a single pass:

```@example interpolation
using StaticArrays
val, grad = LevelSetMethods.value_and_gradient(ϕ, SVector(0.1, 0.2))
```

!!! note "Thread safety"
    An [`InterpolatedField`](@ref) is safe to *evaluate* concurrently from multiple tasks:
    each task gets its own scratch buffer and the shared interpolation operator is read-only.
    *Mutating* the field (`setindex!`, `copy!`) while other tasks evaluate it is not safe;
    mutations invalidate every task's cached cell coefficients.
