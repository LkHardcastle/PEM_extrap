# Inputs
struct PEMData
    y::Vector{Float64}
    cens::Vector{Float64}
    cov::Matrix{Float64}
    p::Int64
    n::Int64
    s::Vector{Float64}
    d::Vector{Int64}
end

struct Settings
    max_ind::Int64
end

abstract type Prior end


# Sampler tracking
mutable struct Dynamics
    ind::Int64
    t_bound::Matrix{Float64}
    a::Matrix{Float64}
    b::Matrix{Float64}
    t_set::Matrix{Float64}
    new_t::Matrix{Bool}
end

mutable struct SamplerEval

end

