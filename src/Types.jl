abstract type State end

mutable struct ZZS
    x::Matrix{Float64}
    v::Matrix{Float64}
    s::Matrix{Bool}
    t::Float64
    active::Vector{CartesianIndex}
end

mutable struct Dynamics

end

mutable struct Times 
    splits::PriorityQueue
    merges::PriorityQueue
    refresh::Vector{Float64}
    hyper::Vector{Float64}
    smps::Vector{Float64}
end

mutable struct Storage
    x::Matrix{Float64}
    v::Matrix{Float64}
    s::Matrix{Bool}
    t::Float64
    x_smp::Matrix{Float64}
    v_smp::Matrix{Float64}
    s_smp::Matrix{Bool}
    t_smp::Float64
end

mutable struct SamplerEval
    newton::Vector{Float64}
    gradient::Int64
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