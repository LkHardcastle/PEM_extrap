#function U_new!(state::State, dyn::Dynamics, priors::Prior, dat::PEMData)
#    ## Calculate the potential, rate of change of potential and constants for updating
#    for j in 1:size(state.active,1)
#        if j == 1
#            range = 1:state.active[1][2]
#        else
#            range = (state.active[j-1][2] + 1):state.active[j][2]
#        end
#        d = findall(dat.d .∈ [range])
#        c = findall(dat.d .> state.active[j][2])
#        if j > 1
#            sj_1 = dat.s[state.active[j-1][2]]
#        else
#            sj_1 = 0.0
#        end
#        dyn.c0[state.active[j]] = exp(sum(state.x[1:state.active[j][2]]))*(sum(dat.y[d] .- sj_1) + length(c)*(dat.s[state.active[j][2]] - sj_1))
#        dyn.δ[state.active[j]] = sum(dat.cens[d])
#        dyn.d0[state.active[j]] = dyn.δ[state.active[j]]*sum(state.x[1:state.active[j][2]])
#    end
#    dyn.∑v = cumsum(state.v, dims = 2)
#    dyn.δ∑v = dyn.δ.*dyn.∑v
#    U_, ∂U_, ∂2U_ = U_eval(state, 0.0, dyn, priors)
#    return U_, ∂U_, ∂2U_
#end

function AV_calc!(state::State, dyn::Dynamics)
    #active = findall(sum.(eachcol(state.s)) .!= 0.0)
    A = cumsum(state.x, dims = 2)
    dyn.A = transpose(dat.UQ)*A
    V = cumsum(state.v, dims = 2)
    dyn.V = transpose(dat.UQ)*V
    #dyn.A = A[:, active]
    #dyn.V = V[:, active]
    #dyn.S = state.s[:,active]
end

function W_calc!(state::State, dyn::Dynamics, dat::PEMData)
    # Faster way of doing this with knowledge of splits but this is fine for now
    #active = findall(sum.(eachcol(state.s)) .!= 0.0)
    #W = zeros(size(dat.W,1),size(active,1))
    #δ = zeros(size(dat.δ,1),size(active,1))
    #for i in eachindex(active)
    #    if i != length(active)
    #        range = active[i]:(active[i+1] - 1)
    #    else
    #        range = active[i]:(size(active,1))
    #    end
    #    W[:,i] = cumsum(dat.W[:,range], dims = 2)
    #    δ[:,i] = cumsum(dat.δ[:,range], dims = 2)
    #end
    #dyn.W = copy(W)
    #dyn.δ = copy(δ)
end


function U_new!(state::State, dyn::Dynamics, priors::Prior, dat::PEMData)
    ## Calculate the potential, rate of change of potential and constants for updating
    AV_calc!(state, dyn)
    U_, ∂U_, ∂2U_ = U_eval(state, 0.0, dyn, priors, dat)
    return U_, ∂U_, ∂2U_
end



function U_eval(state::State, t::Float64, dyn::Dynamics, priors::BasicPrior, dat::PEMData)
    #println(state.x);println(state.v);println(dyn.A);println(dyn.V);error("")
    θ = dyn.A .+ t.*dyn.V
    U_ = sum((exp.(θ).*dat.W .- dat.δ.*θ)) 
    #println(dyn.V);println(θ)
    ∂U_ = sum(dyn.V.*(exp.(θ).*dat.W .- dat.δ)) 
    ∂2U_ = sum((dyn.V.^2).*exp.(θ).*dat.W) 
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
    #println("grad optim");println(state.x)
    t0 = 0.0
    f = copy(∂U)
    f1 = copy(∂2U)
    iter = 1
    while abs(f) > 1e-10
        #println("Grad optim");println(t0);println(f);println(f1);
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

function potential_optim(t_switch::Float64, V::Float64, U_::Float64, ∂U::Float64, state::State, dyn::Dynamics, priors::Prior)
    # Conduct a line search along U(θ + vτ) - U(θ) = -log(V) to find τ
    println("potential optim")
    t0 = 0.0
    Uθ = Base.copy(U_)
    f = log(V)
    f1 = Base.copy(∂U)
    t0 = 0.5
    f_, f1, blank = U_eval(state, t0 + t_switch, dyn, priors)
    f = f_ - Uθ + log(V)        
    dyn.sampler_eval.newton[2] += 1
    k = 1
    while abs(f) > 1e-5
        #println("Potential optim");println(t0);println(f);println(f1)
        t0 = t0 - f/f1
        f_, f1, blank = U_eval(state, t0 + t_switch, dyn, priors)
        f = f_ - Uθ + log(V)
        dyn.sampler_eval.newton[2] += 1
        k += 1
        if k > 1000
            return -100.0
        end
    end
    if isnan(t0) || isinf(t0)
        println(t0)
        verbose(dyn, state)
        error("Potential optim error")
    end
    return t0
end

function ∇U(state::State, dat::PEMData, dyn::Dynamics, priors::Prior)
    ∇U_out = zeros(size(state.active))
    AV_calc!(state, dyn)
    # L x J matrix
    U_ind = reverse(cumsum(reverse(exp.(dyn.A).*dat.W .- dat.δ, dims = 2), dims = 2), dims = 2)
    # Convert to p x J matrix
    U_ind = dat.UQ*U_ind
    ∇U_out = U_ind[state.active]
    for i in eachindex(∇U_out)
        ∇U_out[i] += prior_add(state, priors, state.active[i])
    end
    #U_ind = reverse(cumsum(reverse(exp.(dyn.A).*dyn.W .- dyn.δ, dims = 2), dims = 2), dims = 2)
    #p x J' matrix with correct entries (except for prior terms)
    #println(U_ind)
    #U_ind = dat.UQ*U_ind
    #println(U_ind);error("")
    #j = 1
    #for i in eachindex(dyn.A)
    #    if dyn.S[i]
    #        ∇U_out[j] =  U_ind[j] + prior_add(state, priors, state.active[j])
    #        j += 1
    #    end
    #end
    #println(∇U_out);println(state.v);println(dot(∇U_out,state.v));println("---------")
    return ∇U_out
    #∇Uλ = 0.0
    #∇U_out = Float64[]
    #for j in size(state.active,1):-1:1
    #    if j == 1
    #        range = 1:state.active[1][2]
    #    else
    #        range = (state.active[j-1][2] + 1):state.active[j][2]
    #    end
    #    d = findall(dat.d .∈ [range])
    #    c = findall(dat.d .> state.active[j][2])
    #    if j > 1
    #        sj_1 = dat.s[state.active[j-1][2]]
    #    else
    #        sj_1 = 0.0
    #    end
    #    ∇Uλ += exp(sum(state.x[1:state.active[j][2]]))*(sum(dat.y[d]) - length(d)*sj_1 + length(c)*(dat.s[state.active[j][2]] - sj_1)) - sum(dat.cens[d])
    #    pushfirst!(∇U_out, ∇Uλ + prior_add(state, priors, state.active[j]))
    #end
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