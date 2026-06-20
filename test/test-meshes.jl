using Test
using LevelSetMethods
using StaticArrays

@testset "Basic ops" begin
    nx, ny = 100, 50
    a, b = (-1, 0), (1, 3)
    grid = CartesianGrid(a, b, (nx, ny))
    @test size(grid) === (nx, ny)
    @test length(nodeindices(grid)) == nx * ny
    @test getnode(grid, 1, 1) == SVector(a[1], a[2])
    @test getnode(grid, nx, ny) == SVector(b[1], b[2])
end

@testset "meshsize constructor" begin
    # exact divisor: spacing hits the requested value
    grid = CartesianGrid((-1, -1), (1, 1); meshsize = 0.5)
    @test size(grid) === (5, 5)
    @test LevelSetMethods.meshsize(grid) ≈ SVector(0.5, 0.5)
    # domain is honored exactly (corners unchanged)
    @test getnode(grid, 1, 1) == SVector(-1.0, -1.0)
    @test getnode(grid, 5, 5) == SVector(1.0, 1.0)

    # ceil rule: spacing is never coarser than requested, but may be finer
    grid = CartesianGrid((0, 0), (1, 1); meshsize = 0.3)
    @test size(grid) === (5, 5)
    @test all(LevelSetMethods.meshsize(grid) .<= 0.3)
    @test getnode(grid, 5, 5) == SVector(1.0, 1.0)

    # per-dimension meshsize
    grid = CartesianGrid((0, 0), (2, 1); meshsize = (0.4, 0.3))
    @test size(grid) === (6, 5)
    @test all(LevelSetMethods.meshsize(grid) .<= SVector(0.4, 0.3))

    # validation
    @test_throws ArgumentError CartesianGrid((0, 0), (1, 1); meshsize = -0.1)
    @test_throws ArgumentError CartesianGrid((0, 0), (1, 1); meshsize = (0.1,))
    @test_throws ArgumentError CartesianGrid((1, 1), (0, 0); meshsize = 0.1)
end
