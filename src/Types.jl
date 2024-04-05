# Inputs
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
    tb_init::Float64
    verbose::Bool
end

abstract type Prior end

struct FixedPrior <: Prior
    ω::Matrix{Float64}
    σ::Float64
    σ0::Float64
    μ0::Float64
end

# Sampler tracking
mutable struct SamplerEval
    bounds::Int64
    flip_attempts::Int64
    flips::Int64
    splits::Int64
    merges::Int64
end

mutable struct Dynamics
    ind::Int64
    t_bound::Matrix{Float64}
    a::Matrix{Float64}
    b::Matrix{Float64}
    t_set::Matrix{Float64}
    new_t::Matrix{Bool}
    last_type::String
    sampler_eval::SamplerEval
end



