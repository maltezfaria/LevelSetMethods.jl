using Test
using LinearAlgebra
using StaticArrays
using LevelSetMethods
import LevelSetMethods as LSM

@testset "Spiral curvature flow — narrow band matches full grid" begin
    # Spiral with multiple closely-spaced arms; stresses the band-rebuild logic
    # because inter-arm gaps can be narrower than the halfwidth.
    grid = LSM.CartesianGrid((-1.0, -1.0), (1.0, 1.0), (50, 50))
    d = 1; r0 = 0.5; θ0 = -π / 3; α = π / 100.0
    R = [cos(α) -sin(α); sin(α) cos(α)]; M = R * [1 / 0.06^2 0; 0 1 / (4π^2)] * R'
    ϕ = LSM.MeshField(grid) do (x, y)
        r = sqrt(x^2 + y^2); θ = atan(y, x); res = 1.0e30
        for i in 0:4
            θ1 = θ + (2i - 4) * π; v = [r - r0; θ1 - θ0]
            res = min(res, sqrt(v' * M * v) - d)
        end
        res
    end
    b = (x, t) -> -0.1

    eq_full = LevelSetEquation(;
        ic = deepcopy(ϕ), terms = (CurvatureTerm(b),),
        bc = ExtrapolationBC(2), reinit = LSM.NewtonReinitializer(; reinit_freq = 1),
    )
    nb = NarrowBandMeshField(deepcopy(ϕ); nlayers = 3, reinitialize = true)
    eq_nb = LevelSetEquation(;
        ic = nb, terms = (CurvatureTerm(b),),
        bc = ExtrapolationBC(2), reinit = LSM.NewtonReinitializer(; reinit_freq = 1),
    )

    tf = 0.1
    integrate!(eq_full, tf)
    integrate!(eq_nb, tf)

    ϕ_full = current_state(eq_full)
    ϕ_nb = current_state(eq_nb)
    max_err = maximum(LSM.nodeindices(ϕ_nb)) do I
        abs(ϕ_nb[I] - ϕ_full[I])
    end
    @test max_err < 0.05
end
