```@meta
CurrentModule = LevelSetMethods
```

# [Geometric quantities](@id geometry-queries)

Once you have a level set — whether freshly built (see [Level sets](@ref geometry)) or evolved
through a [`LevelSetEquation`](@ref) — you often want to *read geometry off it*. There are two
flavours of query: **global measures** of the shape it encodes (the size of the enclosed
region, the length or area of the interface) and **local differential geometry** at a point
(the outward normal, the curvature). Both are read-only — they compute from ``\phi`` without
modifying it — and both work on a dense [`MeshField`](@ref) and a [`NarrowBandMeshField`](@ref)
alike.

## Global measures: volume and perimeter

For a level set in ``N`` dimensions, [`LevelSetMethods.volume`](@ref) returns the
``N``-dimensional measure of the enclosed region ``\{x : \phi(x) \le 0\}`` (a length in 1D, an
area in 2D, a volume in 3D), and [`LevelSetMethods.perimeter`](@ref) the ``(N-1)``-dimensional
measure of the interface ``\{x : \phi(x) = 0\}``. They are computed by integrating a smoothed
indicator over the grid, ``\int H(-\phi)\,\mathrm{d}x`` and
``\int \delta(\phi)\,\lvert\nabla\phi\rvert\,\mathrm{d}x`` respectively. Neither is exported.
As a check, the area and circumference of a disk of radius ``0.5`` come out close to the exact
``\pi r^2`` and ``2\pi r``:

```@example geom
using LevelSetMethods
grid = CartesianGrid((-1, -1), (1, 1), (64, 64))
ϕ = MeshField(x -> hypot(x...) - 0.5, grid)
(LevelSetMethods.volume(ϕ), π * 0.5^2), (LevelSetMethods.perimeter(ϕ), 2π * 0.5)
```

(`perimeter` reads ``\nabla\phi`` with a centered stencil that reaches off-grid at the border;
if the field carries no [boundary conditions](@ref boundary-conditions) a default linear
extrapolation is supplied, so the call works without a `bc`.)

Because the smoothed Heaviside and Dirac are spread over roughly one cell width, both measures
are *low-order* approximations: convenient monitoring quantities that converge under
refinement, rather than high-accuracy integrals.

```@example geom
err(n) = abs(LevelSetMethods.volume(MeshField(x -> hypot(x...) - 0.5,
            CartesianGrid((-1, -1), (1, 1), (n, n)))) - π * 0.5^2)
@assert err(128) < err(32)   # hide  (error shrinks under refinement)
err(32), err(128)
```

When you need an accurate integral over the implicit domain or its interface — rather than an
eyeball-grade measure — use the high-order quadrature from the [ImplicitIntegration
extension](@ref extension-implicit-integration) instead. Tracking `volume` (or `perimeter`)
across a simulation is the usual way to monitor mass conservation; drive it from a `posthook`
on [`integrate!`](@ref), as shown on the [Level-set equation](@ref levelset-equation) page.

## Local differential geometry

At a grid node you can query the differential geometry of the level set directly. The gradient
``\nabla\phi`` gives the (unnormalised) direction of steepest ascent;
[`LevelSetMethods.normal`](@ref) normalises it to the outward unit normal
``\boldsymbol{n} = \nabla\phi / \lvert\nabla\phi\rvert``; and
[`LevelSetMethods.curvature`](@ref) returns the mean curvature
``\kappa = \nabla \cdot (\nabla\phi / \lvert\nabla\phi\rvert)``. All three are built from
centered finite differences (via [`LevelSetMethods.hessian`](@ref) for the second-order part)
and take a `CartesianIndex`. For the disk above, the node at ``(0.5, 0)`` has an outward normal
along ``+x`` and a curvature close to ``1/r = 2``:

```@example geom
gc = CartesianGrid((-1, -1), (1, 1), (65, 65))
ϕc = MeshField(x -> hypot(x...) - 0.5, gc)
I = CartesianIndex(49, 33)                # the node at (0.5, 0.0)
LevelSetMethods.normal(ϕc, I), LevelSetMethods.curvature(ϕc, I)
```

Because these are finite differences, they are most reliable where ``\phi`` is well resolved
and close to a signed distance function. Curvature in particular is a *second-derivative*
quantity, sensitive to noise in ``\phi``; keeping the level set reinitialized (see
[Reinitialization](@ref signed-distance)) keeps it well behaved, and it is defined to be zero
wherever ``\nabla\phi`` vanishes. To evaluate the field or its derivatives at *arbitrary*
points rather than at nodes, wrap it in an [`InterpolatedField`](@ref), which differentiates
the local polynomial patch exactly; see [Interpolation](@ref interpolation).
