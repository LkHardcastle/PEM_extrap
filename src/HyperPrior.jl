function h_track_init(priors::Union{HyperPrior,HyperPrior2}, settings::Settings)
    return zeros(settings.max_ind)
end

function h_store!(h_track, priors::Union{HyperPrior,HyperPrior2}, dyn::Dynamics)
    h_track[dyn.ind] = priors.ω0
end

function w_order_int(s::Matrix{Bool}, priors::Union{HyperPrior,HyperPrior2})
    return zeros(size(s,1),size(s,2)), priors
end

function hyper_update!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, Q_f::PriorityQueue, Q_s::PriorityQueue, Q_m::PriorityQueue, dat::PEMData, j::CartesianIndex, priors::HyperPrior, dyn::Dynamics)
    # Update ω
    dyn.sampler_eval.h_updates += 1
    Σs = sum(s)
    logpi1 = (sum(size(s)) - Σs)*log(1-priors.ω0) + Σs*log(priors.ω0) + logpdf(Cauchy(0,priors.σω), log(priors.ω0/(1-priors.ω0)))
    h = rand(Normal(0, sqrt(dyn.adapt_h)))
    ω_prop = 1/(1 + exp(-log(priors.ω0/(1-priors.ω0)) - h))
    logpi2 = (sum(size(s)) - Σs)*log(1-ω_prop) + Σs*log(ω_prop) + logpdf(Cauchy(0,priors.σω), log(ω_prop/(1-ω_prop)))
    if logpi2 - logpi1 > -300
        if min(1,exp(logpi2 - logpi1)) > rand()
            priors.ω0 = copy(ω_prop)
            dyn.sampler_eval.h_acc += 1
            priors.ω = fill(priors.ω0, size(x0))
        end
    end
    # Adaptation
    dyn.adapt_h = exp(log(dyn.adapt_h) + dyn.sampler_eval.h_updates^(-0.6)*(min(1,exp(logpi2 - logpi1)) - 0.44))
    for j in findall(s .== false)
        ## Split queue 
        delete!(Q_s,j)
        enqueue!(Q_s, j, t + rand(Exponential(1/split_rate(priors, j))))
    end
end

function hyper_update!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, Q_f::PriorityQueue, Q_s::PriorityQueue, Q_m::PriorityQueue, dat::PEMData, j::CartesianIndex, priors::HyperPrior2, dyn::Dynamics)
    # Update ω
    dyn.sampler_eval.h_updates += 1
    Σs = sum(s)
    priors.ω0 = rand(Beta(priors.a + Σs, priors.b + sum(size(s)) - Σs))
    priors.ω = fill(priors.ω0, size(x0))
    for j in findall(s .== false)
        ## Split queue 
        delete!(Q_s,j)
        enqueue!(Q_s, j, t + rand(Exponential(1/split_rate(priors, j))))
    end
end