using Test
using LevelSetMethods
using LinearAlgebra
using Plots

nx,ny = 100,100
x     = LinRange(-1,1,nx)
y     = LinRange(-1,1,ny)
hx,hy = step(x),step(y)
grid = CartesianGrid(x,y)
ϕ    = LevelSet(grid) do (x,y)
    1.0
end
add_circle!(ϕ,SVector(0.5,0.0),0.25)
add_circle!(ϕ,SVector(-0.5,0.0),0.25)
add_rectangle!(ϕ,SVector(0.0,0.0),SVector(1.0,0.1))
plot(ϕ)
v     = MeshField(grid) do (x,y)
    -1.0
end
b     = MeshField(x->-0.5,grid)
term1  = NormalAdvectionTerm(v)
terms = (term1,)
# need at least two ghost cells for second order scheme for normal advection
bc    = PeriodicBC(2)
buffer = zero(ϕ)
dt   = 0.5*min(hx,hy)/maximum(abs.(v.vals)) # CFL for normal advection
pgap = 1
anim = @animate for n ∈ 0:30
    fill!(values(buffer),0)
    LevelSetMethods.evolve!(buffer,ϕ,terms,bc,dt)
    if n%pgap == 0
        plot(ϕ)
    end
end
gif(anim, "test.gif", fps = 15)
