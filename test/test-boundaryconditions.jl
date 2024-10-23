using Test
import LevelSetMethods as LSM

@testset "PeriodicBC" begin
    a = (0, 0)
    b = (1, 1)
    n = (10, 5)
    grid = LSM.CartesianGrid(a, b, n)
    vals = rand(n...)
    bcs = ((LSM.PeriodicBC(), LSM.PeriodicBC()), (LSM.PeriodicBC(), LSM.PeriodicBC()))
    mf = LSM.MeshField(vals, grid, bcs)
    @test mf[1, 1] == vals[1, 1]
    @test mf[1, 0] == vals[1, 4]
    @test mf[n.+1] == mf[(2, 2)]
    mf[0, 0] = 0
    @test vals[n[1]-1, n[2]-1] == 0
end
