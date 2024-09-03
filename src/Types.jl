abstract type State end

mutable struct BPS <: State
    x::Matrix{Float64}
    v::Matrix{Float64}
    s::Matrix{Bool}
    g::Matrix{Bool}
    s_loc::Vector{Float64}
    J::Int64
    t::Float64
    active::Array{CartesianIndex{2}}
end

mutable struct ECMC <: State
    x::Matrix{Float64}
    v::Matrix{Float64}
    s::Matrix{Bool}
    g::Matrix{Bool}
    s_loc::Vector{Float64}
    J::Int64
    t::Float64
    active::Array{CartesianIndex{2}}
end

mutable struct ECMC2 <: State
    x::Matrix{Float64}
    v::Matrix{Float64}
    s::Matrix{Bool}
    g::Matrix{Bool}
    s_loc::Vector{Float64}
    t::Float64
    J::Int64
    b::Bool
    active::Array{CartesianIndex{2}}
end

mutable struct SamplerEval
    newton::Vector{Float64}
    gradient::Int64
    Brent_iter::Int64
    Barker_iter::Vector{Int64}
    Barker_acc::Vector{Int64}
    Barker_att::Vector{Int64}
end

mutable struct Dynamics
    ind::Int64
    smp_ind::Int64
    t_det::Float64
    next_event::Int64
    A::Matrix{Float64}
    V::Matrix{Float64}
    S::Matrix{Bool}
    δ::Matrix{Int64}
    W::Matrix{Float64}
    sampler_eval::SamplerEval
end

mutable struct Times 
    next_split::Float64
    next_merge::Float64
    next_merge_index::CartesianIndex
    refresh::Vector{Float64}
    hyper::Vector{Float64}
    smps::Vector{Float64}
end

mutable struct Storage
    x::Array{Float64}
    v::Array{Float64}
    s::Array{Bool}
    s_loc::Array{Float64}
    J::Vector{Int64}
    t::Vector{Float64}
    ω::Array{Float64}
    σ::Array{Float64}
    x_smp::Array{Float64}
    v_smp::Array{Float64}
    s_smp::Array{Bool}
    s_loc_smp::Array{Float64}
    J_smp::Vector{Int64}
    t_smp::Vector{Float64}
    ω_smp::Array{Float64}
    σ_smp::Array{Float64}
end

abstract type Variance end

mutable struct FixedV <: Variance
    σ::Vector{Float64}
end

mutable struct PC <: Variance
    σ::Vector{Float64}
    a::Vector{Float64}
    h::Vector{Float64}
    ind::Float64
end

abstract type Weight end

mutable struct FixedW <: Weight
    ω::Vector{Float64}
end

mutable struct Beta <: Weight
    ω::Vector{Float64}
    a::Vector{Float64}
    b::Vector{Float64}
end


abstract type Grid end

mutable struct Fixed <: Grid
    step::Float64
end

mutable struct Cts <: Grid
    Γ::Float64
    max_points::Int64
    max_time::Float64
end

mutable struct RJ <: Grid
    Γ::Float64
    σ::Float64
    max_time::Float64
end

abstract type Diffusion end

mutable struct RandomWalk <: Diffusion
end

mutable struct GaussLangevin <: Diffusion
    μ::Float64
    σ::Float64
end

mutable struct GammaLangevin <: Diffusion
    α::Float64
    β::Float64
end

mutable struct GompertzBaseline <: Diffusion
    α::Float64
end
abstract type Prior end

#mutable struct BasicPrior <: Prior
#    σ0::Float64
#    σ::Float64
#    ω::Float64
#    p_split::Float64
#end

mutable struct BasicPrior <: Prior
    σ0::Float64
    σ::Variance
    ω::Weight
    p_split::Float64
    grid::Grid
    diff::Vector{Diffusion}
end

mutable struct ARPrior <: Prior
    σ0::Float64
    μ0::Float64
    ω::Weight
    p_split::Float64
    diff::Diffusion
end


struct PEMData
    y::Vector{Float64}
    cens::Vector{Float64}
    covar::Matrix{Float64}
    grp::Vector{Int64}
    p::Int64
    n::Int64
    δ::Matrix{Int64}
    W::Matrix{Float64}
    UQ::Matrix{Float64}
end

struct Settings
    max_ind::Int64
    max_smp::Int64
    max_time::Int64
    smp_rate::Float64
    h_rate::Float64
    r_rate::Float64
    verbose::Bool
    skel::Bool
end

### Functions

function Base.copy(state::BPS)
    return BPS(copy(state.x), copy(state.v), copy(state.s), copy(state.g), copy(state.s_loc), copy(state.J), copy(state.t), copy(state.active))
end

function Base.copy(state::ECMC)
    return ECMC(copy(state.x), copy(state.v), copy(state.s), copy(state.g), copy(state.s_loc), copy(state.J), copy(state.t), copy(state.active))
end

function Base.copy(state::ECMC2)
    return ECMC2(copy(state.x), copy(state.v), copy(state.s), copy(state.g), copy(state.s_loc), copy(state.t), copy(state.J), copy(state.b), copy(state.active))
end

Dynamics(state::State, dat::PEMData) = Dynamics(1, 1, 0.0, 0, copy(state.x), copy(state.x), copy(state.s), copy(dat.δ), copy(dat.W), SamplerEval(zeros(2),0, 0 ,zeros(Int,size(state.x,2)), zeros(Int,size(state.x,2)), zeros(Int,size(state.x,2))))