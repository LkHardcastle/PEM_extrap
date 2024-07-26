function hyper_update!(state::State, priors::Prior)
    if rand() < 0.1
        variance_update!(state, priors, priors.σ)
        weight_update!(state, priors, priors.ω)
    else
        barker_update!(state, priors, priors.diff)
    end
end

function barker_update!(state::State, priors::Prior, diff::RandomWalk)
end

function barker_update!(state::State, priors::Prior, diff::OU)
    # Prior terms cancel via symmetry, could probably reduce computation too but leave for now.
    # Can do this selection better
    j = rand(state.active)
    if j[2] != 1
        Σθ = cumsum(state.x.*state.ξ, dims = 2)
        U1 = sum((exp.(transpose(dat.UQ)*Σθ).*dat.W .- dat.δ.*(transpose(dat.UQ)*Σθ))) 
        state_new = copy(state.ξ)
        state_new[j] = -state_new[j]
        U2 = sum((exp.(transpose(dat.UQ)*cumsum(state.x.*state_new, dims = 2)).*dat.W .- dat.δ.*(transpose(dat.UQ)*cumsum(state.x.*state_new, dims = 2))))
        # b = P(ξ = 1)
        b = (1 + exp(2*diff.ϕ*state.x[j]*Σθ[j[1],j[2]-1]/priors.σ.σ))^-1
        println("++++");println(state.x[j]);println(Σθ[j[1],j[2]-1]);println(b)
        if state.ξ[j] == 1
            logb = log(1-b) - log(b)
        else
            logb = log(b) - log(1-b)
        end
        A = exp(-U2 + U1 + logb)
        println("----");println(A)
        if min(1,A) > rand()
            println("Hello")
            state.ξ[j] = -state.ξ[j]
        end
    end
end

function variance_update!(state::State, priors::Prior, σ::FixedV)

end

function variance_update!(state::State, priors::Prior, σ::PC)
    σ_prop = exp(log(priors.σ.σ) + rand(Normal(0,priors.σ.h)))
    log_prop_dens = sum(logpdf.(Normal(0,σ_prop), state.x[state.active[2:end]])) + log_exp_logpdf(σ_prop, priors.σ.a)
    #if isinf(priors.σ.log_dens)
    priors.σ.log_dens = sum(logpdf.(Normal(0,priors.σ.σ), state.x[state.active[2:end]])) + log_exp_logpdf(log(priors.σ.σ), priors.σ.a)
    #end
    α = min(1, exp(log_prop_dens - priors.σ.log_dens))
    acc = 0
    if rand() < α
        acc = 1
        priors.σ.σ = copy(σ_prop)
        priors.σ.log_dens = copy(log_prop_dens)
    end
    priors.σ.ind += 1
    # Adaptation 
    if priors.σ.ind < 1_000
        priors.σ.h = exp(log(priors.σ.h) + (priors.σ.ind^(-0.6))*(α - 0.234))
    end
    #println("------")
    #println(priors.σ.σ)
    #println(priors.σ.h)
    #println(priors.σ.log_dens)
    #println(log_prop_dens)
end 

function log_Gumbel2_logpdf(τ::Float64,a::Float64)
    return -0.5*log(τ) - a/sqrt(τ)
end


function log_exp_logpdf(logσ::Float64,a::Float64)
    return logσ - exp(logσ)*a
end

function weight_update!(state::State, priors::Prior, ω::FixedW)

end

function weight_update!(state::State, priors::Prior, ω::Beta)
    priors.ω.ω = rand(Distributions.Beta(priors.ω.a + size(state.active,1) - 1, priors.ω.b + prod(size(state.s)) - size(state.active,1) + 1))
end