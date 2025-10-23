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
    @test mf[11, 5] == mf[2, 5]
end

@testset "Normalize BC" begin
    bcs = LSM.PeriodicBC()
    @test LSM._normalize_bc(bcs, 2) ==
        ((LSM.PeriodicBC(), LSM.PeriodicBC()), (LSM.PeriodicBC(), LSM.PeriodicBC()))
    bcs = (LSM.PeriodicBC(), LSM.PeriodicBC())
    @test LSM._normalize_bc(bcs, 2) ==
        ((LSM.PeriodicBC(), LSM.PeriodicBC()), (LSM.PeriodicBC(), LSM.PeriodicBC()))
    bcs = (LSM.PeriodicBC(), LSM.NeumannBC())
    @test LSM._normalize_bc(bcs, 2) ==
        ((LSM.PeriodicBC(), LSM.PeriodicBC()), (LSM.NeumannBC(), LSM.NeumannBC()))
    bcs = [LSM.PeriodicBC(), (LSM.DirichletBC(), LSM.NeumannBC())]
    @test LSM._normalize_bc(bcs, 2) ==
        ((LSM.PeriodicBC(), LSM.PeriodicBC()), (LSM.DirichletBC(), LSM.NeumannBC()))
    bcs = [(LSM.PeriodicBC(), LSM.DirichletBC()), (LSM.DirichletBC(), LSM.NeumannBC())]
    @test_throws ArgumentError LSM._normalize_bc(bcs, 2)
end
