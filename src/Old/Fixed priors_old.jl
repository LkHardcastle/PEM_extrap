function h_track_init(priors::Union{FixedPrior}, settings::Settings)
    return 0.0
end

function h_smp_init(priors::Union{FixedPrior}, settings::Settings)
    return 0.0
end


function h_store!(h_track, priors::Union{FixedPrior}, dyn::Dynamics)
end

function h_store_smp!(h_track, priors::Union{FixedPrior}, dyn::Dynamics)
end
function h_post(h_smp, priors::Union{FixedPrior}, dyn::Dynamics)
    return 0.0
end
function hyper_update!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, Q_f::Float64, Q_s::PriorityQueue, Q_m::PriorityQueue, dat::PEMData, j::CartesianIndex, priors::Union{FixedPrior}, dyn::Dynamics)
    error("Fixed prior structure - hyper update rate ")
end



function w_order_int(s::Matrix{Bool}, priors::Union{FixedPrior})
    return zeros(size(s,1),size(s,2)), priors
end