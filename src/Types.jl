abstract type State end

mutable struct BPS
    x::Matrix{Float64}
    v::Matrix{Float64}
    s::Matrix{Bool}
    t::Float64
    active::Vector{CartesianIndex}
end

mutable struct Dynamics
    ind::Int64
    smp_ind::Int64
    t_det::Float64
    next_event::Int64
    c0::Vector{Float64}
    δ::Vector{Float64}
    d0::Vector{Float64}
    ∑v::Vector{Float64}
    δ∑v::Vector{Float64}
    sampler_eval::SamplerEval
end

mutable struct Times 
    splits::PriorityQueue
    merges::PriorityQueue
    refresh::Vector{Float64}
    hyper::Vector{Float64}
    smps::Vector{Float64}
end

mutable struct Storage
    x::Array{Float64}
    v::Array{Float64}
    s::Array{Bool}
    t::Vector{Float64}
    x_smp::Array{Float64}
    v_smp::Array{Float64}
    s_smp::Array{Bool}
    t_smp::Vector{Float64}
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
    max_ind::Int64
    max_smp::Int64
    max_time::Int64
    smp_rate::Float64
    h_rate::Float64
    r_rate::Float64
    verbose::Bool
end