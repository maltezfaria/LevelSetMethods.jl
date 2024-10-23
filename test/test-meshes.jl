using Test
using LevelSetMethods

@testset "Basic ops" begin
    nx, ny = 100, 50
    a, b   = (-1, 0), (1, 3)
    grid   = CartesianGrid(a, b, (nx, ny))
    @test size(grid) === (nx, ny)
    @test length(CartesianIndices(grid)) == nx * ny
end
