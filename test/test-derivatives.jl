using Test
using LevelSetMethods
using StaticArrays

using LevelSetMethods: D⁺, D⁻, D⁰, D2⁰, D2, weno5⁻, weno5⁺

# Test on f(x,y) = x³ + xy² — non-constant second derivatives, non-zero mixed derivative
# Exact derivatives: ∂_x = 3x²+y², ∂_y = 2xy, ∂_xx = 6x, ∂_yy = 2x, ∂_xy = 2y
grid = CartesianGrid((-2.0, -2.0), (2.0, 2.0), (100, 50))
h = LevelSetMethods.meshsize(grid)
ϕ = MeshField(v -> v[1]^3 + v[1] * v[2]^2, grid)
I = CartesianIndex(9, 7)
x, y = getnode(grid, I)

@testset "First derivatives" begin
    exact = SVector(3x^2 + y^2, 2x * y)
    for dim in 1:2
        # first-order schemes: error O(h)
        @test abs(D⁺(ϕ, I, dim) - exact[dim]) < 10 * h[dim]
        @test abs(D⁻(ϕ, I, dim) - exact[dim]) < 10 * h[dim]
        # second-order scheme: exact on quadratics ⟹ error O(h²)
        @test abs(D⁰(ϕ, I, dim) - exact[dim]) < 5 * h[dim]^2
        # WENO5: fifth-order for smooth functions
        @test abs(weno5⁻(ϕ, I, dim) - exact[dim]) < 5 * h[dim]^2
        @test abs(weno5⁺(ϕ, I, dim) - exact[dim]) < 5 * h[dim]^2
    end
end

@testset "Second derivatives" begin
    exact_diag = SVector(6x, 2x)      # ∂_xx, ∂_yy
    exact_cross = 2y                   # ∂_xy = ∂_yx
    for dim in 1:2
        @test abs(D2⁰(ϕ, I, dim) - exact_diag[dim]) < 5 * h[dim]
        @test abs(D2(ϕ, I, (dim, dim)) - exact_diag[dim]) < 5 * h[dim]
    end
    for dims in ((1, 2), (2, 1))
        @test abs(D2(ϕ, I, dims) - exact_cross) < 5 * h[1] * h[2]
    end
end
