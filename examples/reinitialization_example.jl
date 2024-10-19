using Test
using LevelSetMethods
using LinearAlgebra
using Plots

nx,ny = 100,100
x     = LinRange(-1,1,nx)
y     = LinRange(-1,1,ny)
hx,hy = step(x),step(y)
grid  = CartesianGrid(x,y)
bc    = PeriodicBC(2)
ϕ     = LevelSet(grid,bc) do (x,y)
    1.0
end
add_circle!(ϕ, SVector(0.,0.), .9)
remove_rectangle!(ϕ, SVector(0.,0.5), SVector(.5, 1.0))
# modify the level set function such that it is no longer
# a signed distance function
modifier = MeshField(grid) do (x,y)
    cos(x*4)*sin(y*4)
end
@. ϕ.vals = min(ϕ.vals,.5)^3.0*(abs(modifier.vals)+.01)

term1  = ReinitializationTerm()
terms  = (term1,)
b = zero(ϕ)
integrator = ForwardEuler(0.5)
eq = LevelSetEquation(;terms,integrator,state=ϕ,t=0,buffer=b)

# comparison between the level set before and after reinitialization
# heatmap(eq,title="Before")
# plot!(eq,color=:black,colorbar=true)
integrate!(eq,1.0)
heatmap(eq,title="After")
plot!(eq,color=:black,colorbar=true)
