function h_track_init(priors::Union{HyperPrior2}, settings::Settings)
    return zeros(2,settings.max_ind)
end

function h_smp_init(priors::Union{HyperPrior2}, settings::Settings)
    return zeros(2,settings.max_smp)
end

function h_store!(h_track, priors::Union{HyperPrior2}, dyn::Dynamics)
    h_track[1,dyn.ind] = priors.ω0
    h_track[2,dyn.ind] = priors.σ
end

function h_store_smp!(h_track, priors::Union{HyperPrior2}, dyn::Dynamics)
    h_track[1,dyn.smp_ind] = priors.ω0
    h_track[2,dyn.smp_ind] = priors.σ
end

function h_post(h_smp, priors::Union{HyperPrior2}, dyn::Dynamics)
    return h_smp[1:2,1:(dyn.smp_ind-1)]
end
function w_order_int(s::Matrix{Bool}, priors::Union{HyperPrior2})
    return zeros(size(s,1),size(s,2)), priors
end

function hyper_update!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, Q_f::PriorityQueue, Q_s::PriorityQueue, Q_m::PriorityQueue, dat::PEMData, j::CartesianIndex, priors::HyperPrior2, dyn::Dynamics)
    # Update ω
    dyn.sampler_eval.h_updates += 1
    Σs = sum(s)
    priors.ω0 = rand(Beta(priors.a + Σs, priors.b + prod(size(s)) - Σs))
    #println(priors.a + Σs);println(priors.b + prod(size(s)) - Σs)
    priors.ω = fill(priors.ω0, size(x0))
    # Update σ
    x_curr = x[:,2:end][findall(s0[:,2:end])]
    τ_new = rand(InverseGamma(priors.σa + length(x_curr)/2, priors.σb + sum(x_curr.^2)/2))
    priors.σ = sqrt(τ_new)
    for l in findall(s .== false)
        ## Split queue 
        delete!(Q_s,l)
        enqueue!(Q_s, l, t + rand(Exponential(1/split_rate(s, dat, priors, l ,dyn))))
    end
    for l in findall(s)
        new_bound!(Q_f,t,x,v,s,priors,dat,dyn,l,false)
    end
end