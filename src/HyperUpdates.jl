function hyper_update!(priors::Prior)
    variance_update!(state, priors, priors.σ)
    weight_update!(state, priors, priors.ω)
end

function variance_update!(state::State, priors::Prior, σ::FixedV)

end

function variance_update!(state::State, priors::Prior, σ::PC)
    # Sampling from a Gumbel(1/2,θ) distribution with observations $x$
    τ = 1/priors.σ.σ^2
    τ_prop = τ + rand(Normal(0,sqrt(priors.σ.h)))
    log_prop_dens = sum(logpdf.(Normal(0,sqrt(1/τ_prop)), state.x[state.active[2:end]])) + Gumbel2_logpdf(τ_prop, priors.σ.a)
    α = min(1, exp(log_prop_dens - priors.log_dens))
    acc = 0
    if rand() < α
        acc = 1
        priors.σ.σ = 1/sqrt(τ_prop)
        priors.log_dens = copy(log_prop_dens)
    end
    # Adaptation 
    if inds < 100
        priors.σ.h = exp(log(priors.σ.h) + (0.6^priors.σ.ind)*(α - 0.234))
    end
end

function Gumbel2_logpdf(τ::Float64,a::Float64)
    return -1.5*log(τ) - a/sqrt(τ)
end

function weight_update!(state::State, priors::Prior, ω::FixedW)

end

function weight_update!(state::State, priors::Prior, ω::Beta)
    priors.ω.ω = rand(Beta(priors.ω.a + size(state.active,1) - 1, priors.ω.b + prod(size(state.s)) - size(state.active,1) + 1))
end