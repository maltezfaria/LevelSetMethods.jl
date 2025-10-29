"""
    Reinitializer(; upsample=8, maxiters=20, xtol=1e-8, ftol=1e-8, reinit_freq=1)

Configuration for Newton-based reinitialization to a signed distance function.
The `reinit_freq` parameter specifies how often reinitialization is performed (in time steps).
"""
Base.@kwdef struct Reinitializer
    upsample::Int = 8
    maxiters::Int = 20
    xtol::Float64 = 1.0e-8
    ftol::Float64 = 1.0e-8
    reinit_freq::Int = 1
end
