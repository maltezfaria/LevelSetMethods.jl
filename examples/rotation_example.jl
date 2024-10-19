using Test
using LevelSetMethods
using LinearAlgebra
using Plots

t1 = time()

nx,ny = 201,201
x     = LinRange(-1,1,nx)
y     = LinRange(-1,1,ny)
hx,hy = step(x),step(y)
grid = CartesianGrid(x,y)
bc    = PeriodicBC(1)
ϕ    = LevelSet(grid,bc) do (x,y)
    sqrt((x+0.4)^2 + y^2) - 0.35
end    
𝐮     = MeshField(grid) do (x,y)
    2π*SVector(-y,x)
end    
term1  = AdvectionTerm(velocity=𝐮)
terms  = (term1,)
b = zero(ϕ)
integrator = ForwardEuler(0.5)

eq = LevelSetEquation(;terms,integrator,state=ϕ,t=0,buffer=b)
fig = plot(eq)
@time integrate!(eq,1)    
plot!(fig,eq)

t2 = time()

@info "Total time : $(t2 - t1)"

# anim = @animate for n ∈ 0:100
#     tf = 0.02*n    
#     integrate!(eq,tf)    
#     plot(eq)    
# end
# gif(anim, "test.gif")


