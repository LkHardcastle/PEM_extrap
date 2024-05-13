mutable struct State
    x::Matrix{Float64}
    v::Matrix{Float64}
    s::Matrix{Bool}
    t::Float64
end
abstract type Dynamics end

mutable struct Times 


end

mutable struct SamplerEval
    newton::Vector{Float64}
end

abstract type Priors end

struct PEMData
    y::Vector{Float64}
    cens::Vector{Float64}
    covar::Matrix{Float64}
    p::Int64
    n::Int64
    s::Vector{Float64}
    d::Vector{Int64}
end

struct Settings


end