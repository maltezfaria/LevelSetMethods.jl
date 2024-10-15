using Test
using LevelSetMethods

@testset "Basic ops" begin
    nx, ny = 100, 50
    x      = LinRange(-1, 1, nx)
    y      = LinRange(0, 3, ny)
    grid   = CartesianGrid(x, y)
    @test size(grid) === (length(x), length(y))
    @test meshsize(grid)[1] ≈ step(x)
    @test meshsize(grid)[2] ≈ step(y)
    @test grid[6, 2] == (x[6], y[2])
    @test length(CartesianIndices(grid)) == length(x) * length(y)
end
