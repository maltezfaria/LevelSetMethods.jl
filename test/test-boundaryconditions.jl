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
    ebc = LSM.ExtrapolationBC(2)
    bcs = [LSM.PeriodicBC(), (ebc, LSM.NeumannBC())]
    result = LSM._normalize_bc(bcs, 2)
    @test result[1] == (LSM.PeriodicBC(), LSM.PeriodicBC())
    @test result[2][1] === ebc
    @test result[2][2] === LSM.NeumannBC()
    bcs = [(LSM.PeriodicBC(), ebc), (ebc, LSM.NeumannBC())]
    @test_throws ArgumentError LSM._normalize_bc(bcs, 2)
end
