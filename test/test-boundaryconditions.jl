using Test
import LevelSetMethods as LSM

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
    dbc = LSM.DirichletBC((x, t) -> 0.0)
    bcs = [LSM.PeriodicBC(), (dbc, LSM.NeumannBC())]
    result = LSM._normalize_bc(bcs, 2)
    @test result[1] == (LSM.PeriodicBC(), LSM.PeriodicBC())
    @test result[2][1] === dbc
    @test result[2][2] === LSM.NeumannBC()
    bcs = [(LSM.PeriodicBC(), dbc), (dbc, LSM.NeumannBC())]
    @test_throws ArgumentError LSM._normalize_bc(bcs, 2)
end

@testset "update_bc!" begin
    bc = LSM.DirichletBC((x, t) -> t)
    @test bc.t == 0.0
    returned = LSM.update_bc!(bc, 5.0)
    @test bc.t == 5.0
    @test returned === bc
    # no-op for other BC types
    nbc = LSM.NeumannBC()
    @test LSM.update_bc!(nbc, 1.0) === nbc
end
