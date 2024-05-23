function hyper_update!(priors::Prior)
    variance_update!(state, priors, priors.σ)
    weight_update!(state, priors, priors.ω)
end

function variance_update!(state::State, priors::Prior, σ::Fixed)

end

function variance_update!(state::State, priors::Prior, σ::Cauchy)
    x_curr = state.x[:,2:end][state.active]
    τ_new = rand(InverseGamma(priors.σ.a + length(x_curr)/2, priors.σ.b + sum(x_curr.^2)/2))
    priors.σ = sqrt(τ_new)
end

function weight_update!(state::State, priors::Prior, ω::Fixed)

end

function weight_update!(state::State, priors::Prior, ω::Beta)
    priors.ω.ω = rand(Beta(priors.ω.a + size(state.active,1), priors.ω.b + prod(size(state.s)) - size(state.active,1)))
end