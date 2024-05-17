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
    tb_init::Float64
    smp_rate::Float64
    h_rate::Float64
    r_rate::Float64
    v_abs::Matrix{Float64}
    verbose::Bool
end

mutable struct Times
    Q_s::PriorityQueue{CartesianIndex, Float64}
    Q_m::PriorityQueue{CartesianIndex, Float64}
    T_smp::Vector{Float64}
    T_h::Vector{Float64}
    T_ref::Vector{Float64}
end

mutable struct Track
    x_track::Array{Float64}
    v_track::Array{Float64}
    s_track::Array{Float64}
    t_track::Vector{Float64}
    h_track::Array{Float64}
    x_smp::Array{Float64}
    v_smp::Array{Float64}
    s_smp::Array{Float64}
    t_smp::Vector{Float64}
    h_smp::Array{Float64}
end

mutable struct SamplerEval
    bounds::Int64
    flip_attempts::Int64
    flips::Int64
    splits::Int64
    merges::Int64
    h_updates::Int64
    h_acc::Int64
end

abstract type Prior end

mutable struct BasicPrior <: Prior
    σ::Float64
    σ0::Float64
    μ0::Float64
end

mutable struct HyperPrior <: Prior
    ω::Matrix{Float64}
    ω0::Float64
    a::Float64
    b::Float64
    σ::Float64
    σ0::Float64
    μ0::Float64
    σa::Float64
    σb::Float64
    p_split::Float64
end

abstract type Dynamics end

mutable struct ZigZag <: Dynamics
    ind::Int64
    smp_ind::Int64
    t_bound::Float64
    a::Float64
    b::Float64
    next_bound_int::Float64
    next_event_int::Float64
    new_bound::Bool
    next_event_type::Int64
    next_event_coord::CartesianIndex
    t_set::Float64
    last_type::String
    v_abs::Matrix{Float64}
    sampler_eval::SamplerEval
end

mutable struct BPS <: Dynamics

end

mutable struct ECMC <: Dynamics

end