using Test
using LevelSetMethods
using StaticArrays

@testset "Basic ops" begin
    nx, ny = 100, 50
    a, b = (-1, 0), (1, 3)
    grid = CartesianGrid(a, b, (nx, ny))
    @test size(grid) === (nx, ny)
    @test length(CartesianIndices(grid)) == nx * ny
    @test grid[1, 1] == SVector(a[1], a[2])
    @test grid[nx, ny] == SVector(b[1], b[2])
end
