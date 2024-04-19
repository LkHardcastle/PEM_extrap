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
    max_smp::Int64
    tb_init::Float64
    smp_rate::Float64
    h_rate::Float64
    v_abs::Matrix{Float64}
    verbose::Bool
end

abstract type Prior end

mutable struct FixedPrior <: Prior
    ω::Matrix{Float64}
    σ::Float64
    σ0::Float64
    μ0::Float64
    p_split::Float64
end

mutable struct GeomPrior <: Prior
    ω::Matrix{Float64}
    ω0::Float64
    geom_max::Int64
    σ::Float64
    σ0::Float64
    μ0::Float64
end

mutable struct HyperPrior <: Prior
    ω::Matrix{Float64}
    ω0::Float64
    σω::Float64
    σ::Float64
    σ0::Float64
    μ0::Float64
end

mutable struct HyperPrior2 <: Prior
    ω::Matrix{Float64}
    ω0::Float64
    a::Float64
    b::Float64
    σ::Float64
    σ0::Float64
    μ0::Float64
    p_split::Float64
end
# Sampler tracking
mutable struct SamplerEval
    bounds::Int64
    flip_attempts::Int64
    flips::Int64
    splits::Int64
    merges::Int64
    h_updates::Int64
    h_acc::Int64
end

mutable struct Dynamics
    ind::Int64
    smp_ind::Int64
    t_bound::Matrix{Float64}
    a::Matrix{Float64}
    b::Matrix{Float64}
    t_set::Matrix{Float64}
    new_t::Matrix{Bool}
    last_type::String
    v_abs::Matrix{Float64}
    adapt_h::Float64
    sampler_eval::SamplerEval
end



