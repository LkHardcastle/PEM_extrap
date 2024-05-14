function U_new!(state::State, dyn::Dynamics, priors::Prior, dat::PEMData)
    ## Calculate the potential, rate of change of potential and constants for updating
    for j in 1:size(state.active,1)
        if j == 1
            range = CartesianIndex(state.active[1][1],1):state.active[1]
        else
            range = (state.active[j-1] + CartesianIndex(0,1)):state.active[j]
        end
        d = findall(dat.d .∈ range)
        c = findall(dat.d .> state.active[j][2])
        if j > 1
            sj_1 = dat.s[state.active[j-1][2]]
        else
            sj_1 = 0.0
        end
        dyn.c0[state.active[j]] = exp(sum(state.x[1:state.active[j][2]]))*(sum(dat.y[d] .- sj_1) + length(c)*(dat.s[state.active[j][2]] - sj_1))
        dyn.δ[state.active[j]] = sum(dat.cens[d])
        dyn.d0[state.active[j]] = dyn.δ[state.active[j]]*sum(state.x[1:state.active[j][2]])
    end
    dyn.∑v = cumsum(state.v, dims = 2)
    dyn.δ∑v = dyn.δ.*dyn.∑v
    U, ∂U, ∂2U = U(state, 0.0, dyn, priors)
    return U, ∂U, ∂2U
end

function U(state::State, t::Float64, dyn::Dynamics, priors::Prior)
    # Use known constants to calculate potential and rate of change of potential
    vec1 = exp.(t*dyn.∑v[state.active]).*dyn.c0[state.active]
    U_ = sum(vec1 .- dyn.d0[state.active] .- t*dyn.δ∑v[state.active]) + (1/(2*priors.σ^2))*sum((state.x[state.active][2:end] + state.v[state.active][2:end].*t).^2) + (1/(2*priors.σ0^2))*(state.x[state.active][1] + state.v[state.active][1].*t).^2
    ∂U_ = sum(dyn.∑v[state.active].*(vec1 .- dyn.δ[instate.activeds])) + (1/priors.σ)*sum((state.x[state.active][2:end] .+ state.v[state.active][2:end])) + (1/priors.σ0)*(state.x[state.active][1] + state.v[state.active][1].*t)
    ∂2U_ = sum(dyn.∑v[state.active]^2 .*vec1) + v[state.active][1]/priors.σ0^2 + sum(v[state.active][2:end])/priors.σ^2
    return U_, ∂U_, ∂2U_
end

function grad_optim(∂U::Float64, ∂2U::Float64, state::State, dyn::Dynamics, priors::Prior)
    # Conduct a line search along the time-gradient of the potential to find ∂_tU(θ + vt) = 0
    t0 = 0.0
    f = copy(∂U)
    f1 = copy(∂2U)
    while abs(f) > 1e-10
        t0 = t0 - f/f1
        blank, f, f1 = U(state, t0, dyn, priors)
        dyn.sampler_eval.newton[1] += 1
    end
    return t0
end

function potential_optim(V::Float64, U::Float64, ∂U::Float64, state::State, dyn::Dynamics, priors::Prior)
    # Conduct a line search along U(θ + vτ) - U(θ) = -log(V) to find τ
    t0 = 0.0
    Uθ = copy(U)
    f = log(V)
    f1 = copy(∂U)
    while abs(f) > 1e-10
        t0 = t0 - f/f1
        f_, f1, blank = U(state, t0, dyn, priors)
        f = f_ - Uθ + log(V)
        dyn.sampler_eval.newton[2] += 1
    end
    return t0
end

function ∇U(state::State, dat::PEMData, priors::Prior)
    ∇Uλ = 0.0
    ∇U_out = []
    for j in size(state.active):1
        if j == 1
            range = CartesianIndex(state.active[1][1],1):state.active[1]
        else
            range = (state.active[j-1] + CartesianIndex(0,1)):state.active[j]
        end
        d = findall(dat.d .∈ range)
        c = findall(dat.d .> state.active[j][2])
        if j > 1
            sj_1 = dat.s[state.active[j-1][2]]
        else
            sj_1 = 0.0
        end
        ∇Uλ += exp(sum(state.x[1:state.active[j][2]]))*(sum(dat.y[d] .- sj_1) + length(c)*(dat.s[state.active[j][2]] - sj_1)) - sum(dat.cens[d])
        pushfirst!(∇U_out, ∇Uλ + prior_add(state, priors, state.active[j][2]))
    end
    return ∇U_out
end

function prior_add(state::State, priors::Prior, k::CartesianIndex)
    if k[2] == 1
        return state.x[k]/priors.σ0^2
    else
        return state.x[k]/priors.σ^2
    end
end