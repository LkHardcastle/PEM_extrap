function AV_calc!(state::State, dyn::Dynamics)
    #active = findall(sum.(eachcol(state.s)) .!= 0.0)
    A = cumsum(state.x.*state.ξ, dims = 2)
    dyn.A = transpose(dat.UQ)*A
    V = cumsum(state.v, dims = 2)
    dyn.V = transpose(dat.UQ)*V
end


function U_new!(state::State, dyn::Dynamics, priors::Prior, dat::PEMData)
    ## Calculate the potential, rate of change of potential and constants for updating
    AV_calc!(state, dyn)
    U_, ∂U_, ∂2U_ = U_eval(state, 0.0, dyn, priors, dat)
    return U_, ∂U_, ∂2U_
end



function U_eval(state::State, t::Float64, dyn::Dynamics, priors::BasicPrior, dat::PEMData)
    θ = dyn.A .+ t.*dyn.V
    U_ = sum((exp.(θ).*dyn.W .- dyn.δ.*θ)) 
    ∂U_ = sum(dyn.V.*(exp.(θ).*dyn.W .- dyn.δ)) 
    ∂2U_ = sum((dyn.V.^2).*exp.(θ).*dyn.W) 
    for j in state.active
        if j[2] > 1
            U_ += (1/(2*priors.σ.σ^2))*(state.x[j] + state.v[j]*t)^2
            ∂U_ += (state.v[j]/(priors.σ.σ^2))*(state.x[j] + state.v[j]*t)
            ∂2U_ += (state.v[j]^2)/(priors.σ.σ^2)
        else
            U_ += (1/(2*priors.σ0^2))*(state.x[j] + state.v[j]*t)^2
            ∂U_ += (state.v[j]/(priors.σ0^2))*(state.x[j] + state.v[j]*t)
            ∂2U_ += (state.v[j]^2)/(priors.σ0^2)
        end
    end
    return U_, ∂U_, ∂2U_
end

function grad_optim(∂U::Float64, ∂2U::Float64, state::State, dyn::Dynamics, priors::Prior, dat::PEMData)
    # Conduct a line search along the time-gradient of the potential to find ∂_tU(θ + vt) = 0
    t0 = 0.0
    f = copy(∂U)
    f1 = copy(∂2U)
    iter = 1
    while abs(f) > 1e-10
        t0 = t0 - f/f1
        blank, f, f1 = U_eval(state, t0, dyn, priors, dat)
        dyn.sampler_eval.newton[1] += 1
        iter += 1
        if iter > 1_000
            println(state.x);println(state.v)
            println(t0);println(blank);println(f);println(f1)
            println(∂U);println(∂2U)
            error("Too many its in grad optim")
        end
    end
    if isnan(t0)
        verbose(dyn, state)
        error("Grad optim error")
    end
    #println(t0)
    return t0
end

function ∇U(state::State, dat::PEMData, dyn::Dynamics, priors::Prior)
    ∇U_out = zeros(size(state.active))
    AV_calc!(state, dyn)
    # L x J matrix
    U_ind = reverse(cumsum(reverse(exp.(dyn.A).*dyn.W .- dyn.δ, dims = 2), dims = 2), dims = 2)
    # Convert to p x J matrix
    U_ind = dat.UQ*U_ind
    ∇U_out = U_ind[state.active]
    for i in eachindex(∇U_out)
        ∇U_out[i] += prior_add(state, priors, state.active[i])
    end
    return ∇U_out
end

function prior_add(state::State, priors::BasicPrior, k::CartesianIndex)
    if k[2] == 1
        return state.x[k]/priors.σ0^2
    else
        return state.x[k]/priors.σ.σ^2
    end
end

function prior_add(state::State, priors::ARPrior, k::CartesianIndex)
    return sum(cumsum(state.x, dims = 2)[k[2]:end] .- priors.μ0)/priors.σ0^2
end