using Test
using LevelSetMethods
using StaticArrays

# Helper: get text/plain output of x
showstr(x) = sprint(show, MIME("text/plain"), x)

@testset "CartesianGrid" begin
    g = CartesianGrid((0, 0), (1, 1), (10, 4))
    s = showstr(g)
    @test startswith(s, "CartesianGrid in ℝ²")
    @test occursin("├─ domain:  [0.0, 1.0] × [0.0, 1.0]", s)
    @test occursin("├─ nodes:   10 × 4", s)
    @test occursin("└─ spacing: h = (0.1111, 0.3333)", s)
end

@testset "BoundaryConditions" begin
    @test sprint(show, PeriodicBC()) == "Periodic"
    @test sprint(show, NeumannBC()) == "Degree 0 extrapolation"
    @test sprint(show, LinearExtrapolationBC()) == "Degree 1 extrapolation"
    @test sprint(show, ExtrapolationBC{4}()) == "Degree 4 extrapolation"
    @test sprint(show, DirichletBC((x, t) -> 0.0)) == "Dirichlet"
end

@testset "MeshField" begin
    grid = CartesianGrid((-1, -1), (1, 1), (5, 5))

    @testset "scalar, no bc" begin
        ϕ = MeshField(x -> x[1]^2 + x[2]^2 - 0.5^2, grid)
        s = showstr(ϕ)
        @test startswith(s, "MeshField on CartesianGrid in ℝ²")
        @test occursin("├─ domain:  [-1.0, 1.0] × [-1.0, 1.0]", s)
        @test occursin("├─ nodes:   5 × 5", s)
        @test occursin("├─ spacing: h = (0.5, 0.5)", s)
        @test !occursin("bc:", s)
        @test occursin("├─ eltype:  Float64", s)
        @test occursin("└─ values:  min = -0.25,  max = 1.75", s)
    end

    @testset "scalar, with bc" begin
        ϕ = LevelSet(x -> x[1]^2 + x[2]^2 - 0.5^2, grid)
        eq = LevelSetEquation(; terms = (NormalMotionTerm(1.0),), ic = ϕ, bc = NeumannBC())
        s = showstr(current_state(eq))
        @test occursin("├─ bc:     Degree 0 extrapolation (all)", s)
    end

    @testset "vector-valued" begin
        u = MeshField(x -> SVector(x[1], x[2]), grid)
        s = showstr(u)
        @test startswith(s, "MeshField on CartesianGrid in ℝ²")
        @test occursin("└─ eltype:  SVector{2, Float64}", s)
        @test !occursin("values", s)
        @test !occursin("bc:", s)
    end
end

@testset "Time integrators" begin
    @test showstr(ForwardEuler()) == "ForwardEuler (1st order explicit)\n  └─ cfl: 0.5"
    @test showstr(RK2()) == "RK2 (2nd order TVD Runge-Kutta, Heun's method)\n  └─ cfl: 0.5"
    @test showstr(RK3()) == "RK3 (3rd order TVD Runge-Kutta)\n  └─ cfl: 0.5"
    @test showstr(SemiImplicitI2OE()) ==
        "SemiImplicitI2OE (semi-implicit advection, Mikula et al.)\n  └─ cfl: 2.0"
    @test showstr(ForwardEuler(; cfl = 0.3)) == "ForwardEuler (1st order explicit)\n  └─ cfl: 0.3"
end

@testset "NewtonReinitializer" begin
    s = showstr(NewtonReinitializer())
    @test startswith(s, "NewtonReinitializer")
    @test occursin("├─ frequency: every step", s)
    @test occursin("├─ order:     3", s)
    @test occursin("├─ upsample:  8×", s)
    @test occursin("├─ maxiters:  20", s)
    @test occursin("├─ xtol:      1.0e-8", s)
    @test occursin("└─ ftol:      1.0e-8", s)

    s5 = showstr(NewtonReinitializer(; reinit_freq = 5))
    @test occursin("├─ frequency: every 5 steps", s5)
end

@testset "LevelSetEquation" begin
    grid = CartesianGrid((-1, -1), (1, 1), (20, 20))
    ϕ = LevelSet(x -> x[1]^2 + x[2]^2 - 0.5^2, grid)
    𝐮 = MeshField(x -> SVector(1.0, 0.0), grid)
    eq = LevelSetEquation(; terms = (AdvectionTerm(𝐮),), ic = ϕ, bc = NeumannBC())

    s = showstr(eq)
    @test startswith(s, "LevelSetEquation")
    @test occursin("├─ equation: ϕₜ + 𝐮 ⋅ ∇ ϕ = 0", s)
    @test occursin("├─ time:     0.0", s)
    @test occursin("├─ integrator: RK2", s)
    @test occursin("│  └─ cfl: 0.5", s)
    @test occursin("├─ reinit:   none", s)
    @test occursin("├─ state: MeshField on CartesianGrid in ℝ²", s)
    @test occursin("│  ├─ bc:     Degree 0 extrapolation (all)", s)
    @test occursin("│  ├─ eltype:  Float64", s)
    @test occursin("├─ log: SimulationLog (empty)", s)
    @test endswith(s, "╰─")

    # Compact show (no MIME)
    @test sprint(show, eq) == "LevelSetEquation(ϕₜ + 𝐮 ⋅ ∇ ϕ = 0, t=0.0)"
end

@testset "SimulationLog" begin
    grid = CartesianGrid((-1, -1), (1, 1), (20, 20))
    ϕ = LevelSet(x -> x[1]^2 + x[2]^2 - 0.5^2, grid)
    𝐮 = MeshField(x -> SVector(1.0, 0.0), grid)
    eq = LevelSetEquation(; terms = (AdvectionTerm(𝐮),), ic = ϕ, bc = NeumannBC())

    @testset "empty" begin
        @test showstr(eq.log) == "SimulationLog (empty)"
    end

    integrate!(eq, 0.1)

    @testset "after integration" begin
        s = showstr(eq.log)
        @test occursin("SimulationLog:", s)
        @test occursin("steps", s)
        @test occursin("wall time", s)
        @test occursin("reinit:    none", s)
        @test occursin("𝐮 ⋅ ∇ ϕ", s)
        @test occursin("compute", s)
        @test occursin("ϕ range", s)
        @test occursin("Δt:", s) && occursin("min=", s) && occursin("max=", s) && occursin("avg=", s)
    end

    @testset "with reinit" begin
        ϕ2 = LevelSet(x -> x[1]^2 + x[2]^2 - 0.5^2, grid)
        eq2 = LevelSetEquation(;
            terms = (AdvectionTerm(𝐮),),
            ic = ϕ2,
            bc = NeumannBC(),
            reinit = NewtonReinitializer(; reinit_freq = 1),
        )
        integrate!(eq2, 0.05)
        s = showstr(eq2.log)
        @test occursin("reinit:", s)
        @test !occursin("reinit:    none", s)
    end
end
