function h_track_init(priors::FixedPrior, settings::Settings)
    return 0.0
end

function h_store!(h_track, priors::FixedPrior, dyn::Dynamics)
end

function hyper_update!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, Q_f::PriorityQueue, Q_s::PriorityQueue, Q_m::PriorityQueue, dat::PEMData, j::CartesianIndex, priors::FixedPrior, dyn::Dynamics)
    error("Fixed prior structure - hyper update rate ")
end

function w_order_int(s::Matrix{Bool}, priors::FixedPrior)
    return zeros(size(s,1),size(s,2)), priors
end