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

@testset "ExtrapolationBC" begin
    # On a 1D node-based grid [0,1] with N nodes, node i is at x=(i-1)*h.
    # ExtrapolationBC{P} should reproduce degree-(P-1) polynomials exactly.
    grid = LSM.CartesianGrid((0.0,), (1.0,), (10,))
    h = LSM.meshsize(grid)[1]
    for P in 1:4
        f = x -> x[1]^(P - 1)
        ϕ = LSM.MeshField(f, grid)
        bcs = ((LSM.ExtrapolationBC(P), LSM.ExtrapolationBC(P)),)
        ϕ_bc = LSM.add_boundary_conditions(ϕ, bcs)
        # ghost at index 0 is at x = -h; ghost at index 11 is at x = 10h = 1+h
        @test ϕ_bc[0] ≈ (-h)^(P - 1)     atol = 1.0e-12
        @test ϕ_bc[11] ≈ (10h)^(P - 1)    atol = 1.0e-12
    end
    # ExtrapolationBC{1} == NeumannBC (constant extension)
    grid2 = LSM.CartesianGrid((0.0,), (1.0,), (5,))
    f = x -> 3 * x[1] - 1.0   # arbitrary function
    ϕ = LSM.MeshField(f, grid2)
    ϕ_n = LSM.add_boundary_conditions(ϕ, ((LSM.NeumannBC(), LSM.NeumannBC()),))
    ϕ_e1 = LSM.add_boundary_conditions(ϕ, ((LSM.ExtrapolationBC(1), LSM.ExtrapolationBC(1)),))
    for k in (0, -1, 6, 7)
        @test ϕ_n[k] ≈ ϕ_e1[k]
    end
    # ExtrapolationBC{2} == NeumannGradientBC on uniform node-based grids
    ϕ_ng = LSM.add_boundary_conditions(ϕ, ((LSM.NeumannGradientBC(), LSM.NeumannGradientBC()),))
    ϕ_e2 = LSM.add_boundary_conditions(ϕ, ((LSM.ExtrapolationBC(2), LSM.ExtrapolationBC(2)),))
    @test ϕ_ng[0] ≈ ϕ_e2[0]
    @test ϕ_ng[6] ≈ ϕ_e2[6]
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
