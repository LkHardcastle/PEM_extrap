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
    T_flip::Float64
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