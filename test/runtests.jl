using Test

@testset "Meshes" begin
    include("meshes_test.jl")
end

@testset "Derivatives test" begin
    include("derivatives_test.jl")
end
