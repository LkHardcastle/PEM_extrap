function AV_calc!(state::State, dyn::Dynamics)
    #active = findall(sum.(eachcol(state.s)) .!= 0.0)
    A = cumsum(state.x, dims = 2)
    dyn.A = transpose(dat.UQ)*A
    V = cumsum(state.v, dims = 2)
    dyn.V = transpose(dat.UQ)*V
end


function U_new!(state::State, dyn::Dynamics, priors::Prior)
    ## Calculate the potential, rate of change of potential and constants for updating
    AV_calc!(state, dyn)
    U_, ∂U_, ∂2U_ = U_eval(state, 0.0, dyn, priors)
    return U_, ∂U_, ∂2U_
end

function logistic(x::Float64)
    return 1/(1 + exp(-x))
end

function U_eval(state::State, t::Float64, dyn::Dynamics, priors::Prior)
    θ = dyn.A .+ t.*dyn.V
    U_ = sum((exp.(θ).*dyn.W .- dyn.δ.*θ)) 
    ∂U_ = sum(dyn.V.*(exp.(θ).*dyn.W .- dyn.δ)) 
    ∂2U_ = sum((dyn.V.^2).*exp.(θ).*dyn.W) 
    Σθ = cumsum(state.x .+ t.*state.v, dims = 2)
    μθ = zeros(size(state.x))
    ∂μθ = zeros(size(state.x,1), size(state.x,2))
    for i in 1:size(state.x, 1)
        μθ[i, :] = drift_U(Σθ[i, :], priors.diff[i])
        ∂μθ[i,:] = drift_deriv_t(Σθ[i, :], priors.diff[i])
    end
    ∑v = cumsum(state.v, dims = 2)
    for j in state.active
        if j != state.active[1]
            U_ += (1/(2*priors.σ.σ^2))*(state.x[j] + state.v[j]*t)^2
            U_ += -log(1 + tanh(μθ[j[1], j[2]-1]*(state.x[j] + state.v[j]*t)))
            ∂U_ += (state.v[j]/(priors.σ.σ^2))*(state.x[j] + state.v[j]*t) 
            ∂U_ += -2*(∑v[j[1],j[2] - 1]*(state.x[j] + state.v[j]*t)*∂μθ[j[1], j[2]-1] + state.v[j]*μθ[j[1], j[2]-1])/(exp(2*(state.x[j] + state.v[j]*t)*μθ[j[1], j[2]-1]) + 1)
            ∂2U_ += (state.v[j]^2)/(priors.σ.σ^2)
        else
            U_ += (1/(2*priors.σ0^2))*(state.x[j] + state.v[j]*t)^2
            ∂U_ += (state.v[j]/(priors.σ0^2))*(state.x[j] + state.v[j]*t)
            ∂2U_ += (state.v[j]^2)/(priors.σ0^2)
        end
    end
    return U_, ∂U_, ∂2U_
end

function ∇U(state::State, dat::PEMData, dyn::Dynamics, priors::Prior)
    ∇U_out = zeros(size(state.active))
    AV_calc!(state, dyn)
    # L x J matrix
    U_ind = reverse(cumsum(reverse(exp.(dyn.A).*dyn.W .- dyn.δ, dims = 2), dims = 2), dims = 2)
    # Convert to p x J matrix
    U_ind = dat.UQ*U_ind
    ∇U_out = U_ind[state.active]
    Σθ = cumsum(state.x, dims = 2)
    μθ = zeros(size(state.x))
    ∂μθ = zeros(size(state.x,1), size(state.x,2), size(state.x,2))
    for i in 1:size(state.x, 1)
        μθ[i, :] = drift(Σθ[i, :], priors.diff[i])
        ∂μθ[i, :, :] = drift_deriv(Σθ[i, :], priors.diff[i])
    end
    for i in eachindex(∇U_out)
        ∇U_out[i] += prior_add(state, priors, state.active[i])
        ∇U_out[i] += drift_add(state.x, μθ, ∂μθ, priors.diff, state.active[i])
    end
    return ∇U_out
end

############ Random Walk

function drift(θ, diff::RandomWalk)
    return zeros(size(θ))
end

function drift_U(θ, diff::Union{RandomWalk, GammaLangevin})
    return zeros(size(θ))
end

function drift_deriv(θ, diff::RandomWalk)
    return zeros(size(θ,1), size(θ,2), size(θ,2))
end

function drift_deriv_t(θ, diff::Union{RandomWalk, GammaLangevin})
    return zeros(size(θ))
end

function drift_add(x, μθ, ∂μθ, diff::RandomWalk, j::CartesianIndex)
    return 0.0
end

################ GaussLangevin

function drift(θ, diff::GaussLangevin)
    return -0.5.*(θ .- diff.μ)./diff.σ^2
end

function drift_U(θ, diff::GaussLangevin)
    return -0.5.*(θ .- diff.μ)./diff.σ^2
end

function drift_deriv(θ, diff::GaussLangevin)
    return fill(-1/(2*diff.σ^2), size(θ,2), size(θ,2))
end

function drift_deriv_t(θ, diff::GaussLangevin)
    return fill(-1/(2*diff.σ^2), size(θ))
end

###### GammaLangevin

function drift(θ, diff::GammaLangevin)
    return 0.5*(diff.α .- diff.β.*exp.(θ))
end

function drift_deriv(θ, diff::GammaLangevin)
    #return fill(-1/(2*diff.σ^2), size(θ,1), size(θ,2), size(θ,2))
    #error("")
    out = Array{Float64, 3}(undef, size(θ,1), size(θ,2))
    for i in 1:size(θ, 2)
        out[i,:] = -0.5.*diff.β.*exp.(θ)
    end
    return out
end

##################

function drift_add(x, μθ, ∂μθ, diff::Union{GaussLangevin, GammaLangevin}, j::CartesianIndex)
    if j[2] > 1
        out = μθ[j[1], j[2] - 1]*(tanh(x[j]*μθ[j[1], j[2] - 1]) - 1)
    else
        out = 0.0
    end
    if j[2] < size(x,2)
        for k in (j[2] + 1):size(x,2)
            out += x[j[1], k]*∂μθ[j[1], j[2], k - 1]*(tanh(x[j[1], k]*μθ[j[1], k - 1]) - 1)
        end
    end
    return  out
end

function prior_add(state::State, priors::Prior, k::CartesianIndex)
    if k[2] == 1
        return state.x[k]/priors.σ0^2
    else
        return state.x[k]/(priors.σ.σ^2)
    end
end

function diffusion_time!(state::State, priors::Prior, dyn::Dynamics, t_end::Float64, t_switch::Float64)
    for j in eachindex(priors.diff)
        t_end, t_switch = diffusion_time_inner!(state, priors, dyn, t_end, t_switch, j, priors.diff[j])
    end
    return t_end, t_switch
end

function diffusion_time_inner!(state::State, priors::Prior, dyn::Dynamics, t_end::Float64, t_switch::Float64, j::Int64, diff::GammaLangevin)
    # Compute end state
    state_curr = copy(state)
    state_curr.x += state.v.*t_switch
    state_new = copy(state)
    state_new.x += state.v.*t_end
    # Compute bound
    Λ = max(λ_diff(state_curr, priors, j), λ_diff(state_new, priors, j)) #+ 10.0
    if Λ > 0.0
        Λ += 2.0
    end
    if Λ > 10_000
        println(Λ)
        t_move = min(rand(Exponential(1/λ_diff(state_curr, priors, j))),0.00001)
        if t_move < t_end
            dyn.next_event = 5
            return t_move, 0.0
        else
            return t_end, t_switch
        end
    end
    t_move = 0.0
    while t_move < t_end 
        t_new = rand(Exponential(1/Λ))
        t_move += t_new
        if t_move < t_end
            state_curr.x += t_new.*state.v
            λ = λ_diff(state_curr, priors, j)
            if λ/Λ > 1.0
                println(λ);println(Λ)
                error("Bad bound")
            end
            if λ/Λ > rand()
                dyn.next_event = 5
                return t_move, 0.0
            end
        end
    end
    return t_end, t_switch
end

function diffusion_time_inner!(state::State, priors::Prior, dyn::Dynamics, t_end::Float64, t_switch::Float64, j::Int64, diff::Union{GaussLangevin, RandomWalk})
    return t_end, t_switch
end

function λ_diff(state::State, priors::Prior, j::Int64)
    # Add time movement
    ∇U_out = zeros(size(state.active))
    Σθ = cumsum(state.x, dims = 2)
    μθ = drift(Σθ[j,:], priors.diff[j])
    ∂μθ = drift_deriv(Σθ[j,:], priors.diff)
    for i in eachindex(∇U_out)
        if state.active[i][1] == j
            ∇U_out[i] += drift_add(state.x, μθ, ∂μθ, priors.diff[j], state.active[i])
        end
    end
    return max(0, dot(state.v[state.active],∇U_out))
end