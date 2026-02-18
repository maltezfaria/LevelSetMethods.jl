using LevelSetMethods
using StaticArrays
using Test

@testset "SemiImplicitI2OE periodic transport" begin
    grid = CartesianGrid((0.0,), (1.0,), (201,))
    ϕ0 = LevelSet(grid) do (x,)
        sin(2π * x) + 0.15 * cos(6π * x)
    end
    vel = MeshField(x -> SVector(1.0), grid)
    eq = LevelSetEquation(;
        terms = (AdvectionTerm(vel, Upwind()),),
        integrator = SemiImplicitI2OE(cfl = 3.0),
        levelset = deepcopy(ϕ0),
        bc = PeriodicBC(),
    )

    tf = 0.35
    integrate!(eq, tf)

    ϕ_ref = LevelSet(grid) do (x,)
        xshift = mod(x - tf, 1.0)
        sin(2π * xshift) + 0.15 * cos(6π * xshift)
    end
    err = maximum(abs.(values(LevelSetMethods.current_state(eq)) .- values(ϕ_ref)))
    @test err < 0.12
end

@testset "SemiImplicitI2OE periodic transport 2D" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (121, 111))
    ϕ0 = LevelSet(grid) do (x, y)
        sin(2π * x) + 0.4 * cos(2π * y)
    end
    vel = MeshField(x -> SVector(0.75, -0.35), grid)
    eq = LevelSetEquation(;
        terms = (AdvectionTerm(vel, Upwind()),),
        integrator = SemiImplicitI2OE(cfl = 2.5),
        levelset = deepcopy(ϕ0),
        bc = PeriodicBC(),
    )

    tf = 0.2
    integrate!(eq, tf)
    ϕ_ref = LevelSet(grid) do (x, y)
        xshift = mod(x - 0.75 * tf, 1.0)
        yshift = mod(y + 0.35 * tf, 1.0)
        sin(2π * xshift) + 0.4 * cos(2π * yshift)
    end
    err = maximum(abs.(values(LevelSetMethods.current_state(eq)) .- values(ϕ_ref)))
    @test err < 0.2
end

@testset "SemiImplicitI2OE supports non-periodic BCs" begin
    grid = CartesianGrid((0.0,), (1.0,), (121,))
    ϕ0 = LevelSet(x -> 0.7, grid)
    term_neumann = AdvectionTerm((x, t) -> sin(2π * x[1]), Upwind())
    eq_neumann = LevelSetEquation(;
        terms = (term_neumann,),
        integrator = SemiImplicitI2OE(cfl = 4.0),
        levelset = deepcopy(ϕ0),
        bc = NeumannGradientBC(),
    )
    integrate!(eq_neumann, 0.6)
    @test maximum(abs.(values(LevelSetMethods.current_state(eq_neumann)) .- 0.7)) < 1.0e-12

    term_dirichlet = AdvectionTerm((x, t) -> 0.5, Upwind())
    eq_dirichlet = LevelSetEquation(;
        terms = (term_dirichlet,),
        integrator = SemiImplicitI2OE(cfl = 2.0),
        levelset = LevelSet(x -> 0.0, grid),
        bc = DirichletBC(0.0),
    )
    integrate!(eq_dirichlet, 0.4)
    vals = values(LevelSetMethods.current_state(eq_dirichlet))
    @test all(isfinite, vals)
    @test maximum(abs.(vals)) < 1.0e-12
end

@testset "SemiImplicitI2OE checks invalid setup" begin
    grid1d = CartesianGrid((0.0,), (1.0,), (41,))
    ϕ1d = LevelSet(x -> x[1], grid1d)
    eq_multiterm = LevelSetEquation(;
        terms = (
            AdvectionTerm((x, t) -> 1.0, Upwind()),
            CurvatureTerm((x, t) -> -0.1),
        ),
        integrator = SemiImplicitI2OE(),
        levelset = ϕ1d,
        bc = PeriodicBC(),
    )
    @test_throws ArgumentError integrate!(eq_multiterm, 0.1)

    grid_small = CartesianGrid((0.0,), (1.0,), (2,))
    ϕ_small = LevelSet(x -> x[1], grid_small)
    eq_small = LevelSetEquation(;
        terms = (AdvectionTerm((x, t) -> 1.0, Upwind()),),
        integrator = SemiImplicitI2OE(),
        levelset = ϕ_small,
        bc = NeumannBC(),
    )
    @test_throws ArgumentError integrate!(eq_small, 0.1)
end
