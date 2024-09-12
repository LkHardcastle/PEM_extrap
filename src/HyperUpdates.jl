
function hyper_update!(state::State, dyn::Dynamics, dat::PEMData, priors::Prior)
    for k in axes(state.x,1)
        variance_update!(state, priors, priors.σ, k)
        weight_update!(state, priors, priors.ω, k)
    end
    grid_update!(state, dyn, dat, priors, priors.grid)
end

function grid_update!(state::State, dyn::Dynamics, dat::PEMData, priors::Prior, Grid::Fixed)
end

function grid_update!(state::State, dyn::Dynamics, dat::PEMData, priors::Prior, grid::Cts)
    rem_ind = findall(sum.(eachcol(state.s)) .!= 0.0)
    state.x, state.v, state.s, state.g, state.s_loc = state.x[:,rem_ind], state.v[:,rem_ind], state.s[:,rem_ind], state.g[:,rem_ind], state.s_loc[rem_ind]
    J_curr = sum(state.s)
    Pois_new = []
    weight_vec = []
    for k in axes(state.x,1)
        push!(Pois_new, rand(Poisson(priors.grid.max_time*priors.grid.Γ*(1 - priors.ω.ω[k]))))
        push!(weight_vec, (1 - priors.ω.ω[k]))
    end
    J_new = min(sum(Pois_new), priors.grid.max_points - J_curr)
    weight_vec = weight_vec/sum(weight_vec)
    J_row = rand(Categorical(weight_vec), J_new)
    J_loc = rand(Uniform(0,priors.grid.max_time), J_new)
    g_new = fill(false, size(state.x, 1), J_new)
    for i in 1:J_new
        g_new[J_row[i], i] = true
    end
    ind_new = sortperm(vcat(state.s_loc, J_loc))
    zero_mat = zeros(size(state.x, 1),J_new)
    state.s_loc = vcat(state.s_loc, J_loc)[ind_new]
    state.x = hcat(state.x, zero_mat)[:,ind_new]
    state.v = hcat(state.v, zero_mat)[:,ind_new]
    state.s = hcat(state.s, fill(false, size(zero_mat)))[:,ind_new]
    state.g = hcat(state.g, g_new)[:,ind_new]
    state.active = findall(state.s)
    state.J = length(state.s_loc)
    dat_update!(state, dyn, dat)
end

function grid_update!(state::State, dyn::Dynamics, dat::PEMData, priors::Prior, grid::RJ)
    grid_merge!()
    grid_split!()
end

function grid_merge!()
    # Select point to remove uniformly at random
    s_remove = rand(DiscreteUniform(1,state.J))
    u = state.x[1,s_remove]
    state_merge = merge_state(state, s_remove)
    A = log_MHG_ratio(state, state_merge, u, grid)
    if rand() < min(1, exp(-A))
        state = copy(state_merge)
    end

end

function merge_state(state::State, s_remove::Int64)
    state_merge = copy(state)
    # Update state

    # Update velocity

    # Remove 
    state_merge.x, state_merge.v, state_merge.s, state_merge.g, state_merge.s_loc = state_merge.x[:,rem_ind], state_merge.v[:,rem_ind], state_merge.s[:,rem_ind], state_merge.g[:,rem_ind], state_merge.s_loc[rem_ind]
    return state_merge
end

function grid_split!()
    # Find new location
    s_new = rand(Uniform(0,grid.max_time))
    u = rand(Normal(0, grid.σ))
    state_new = split_state(state, s_new, u)
    A = log_MHG_ratio(state_new, state, u, grid)
    if rand() < min(1, exp(A))
        state = copy(state_new)
    end
end

function split_state(state::State, s_new::Int64, u::Float64)
    # Place new point
    state_new = copy(state)
    g_new = fill(true, size(state_new.x, 1), 1)
    ind_new = sortperm(vcat(state_new.s_loc, s_new))
    new_point = fill(u, size(state_new.x, 1), 1)
    state_new.s_loc = vcat(state_new.s_loc, s_new)[ind_new]
    state_new.x = hcat(state_new.x, new_point)[:,ind_new]
    # Fix velocity updates
    state_new.v = hcat(state_new.v, new_point)[:,ind_new]error("")
    state_new.s = hcat(state_new.s, fill(true, size(new_point)))[:,ind_new]
    state_new.g = hcat(state_new.g, g_new)[:,ind_new]
    state_new.active = findall(state_new.s)
    state_new.J = length(state_new.s_loc)
    # Update next point
    state_new.x[1,findfirst(state_new.x[1,:] .== u) + 1] -= u
    return state_new
end

function log_MHG_ratio(state_split::State, state_curr::State, u::Float64, grid::RJ)
    A = U_eval(state_split) - U_eval(state) - logpdf(Normal(0, grid.σ), u) - log(state_split.J - 1)
    return A
end

function dat_update!(state::State, dyn::Dynamics, dat::PEMData)
    L = size(dat.UQ, 2)
    J = size(state.s_loc,1)
    W = zeros(L,J)
    δ = zeros(L,J)
    d = zeros(Int, length(dat.y))
    for i in eachindex(dat.y)
        if isnothing(findfirst(state.s_loc .> dat.y[i]))
            d[i] = J
        else
            d[i] = findfirst(state.s_loc .> dat.y[i])
        end
    end
    for l in 1:L
        yl = dat.y[findall(dat.grp .== l)]
        dl = d[findall(dat.grp .== l)]
        δl = dat.cens[findall(dat.grp .== l)]
        for j in 1:J
            if j == 1
                sj1 = 0.0
            else
                sj1 = state.s_loc[j-1]
            end
            W[l,j] = sum(yl[findall(dl .== j)]) .- length(findall(dl .== j))*sj1 + length(findall(dl .> j))*(state.s_loc[j] - sj1)
            δ[l,j] = length(intersect(findall(δl .== 1), findall(dl .== j)))
        end
    end
    dyn.W = copy(W)
    dyn.δ = copy(δ)
end

function variance_update!(state::State, priors::Prior, σ::FixedV, k::Int64)

end

function variance_update!(state::State, priors::Prior, σ::PC, k::Int64)
    # Drift terms don't depend on σ and cancel 
    σ_prop = exp(log(priors.σ.σ[k]) + rand(Normal(0,priors.σ.h[k])))
    active_j = filter(idx -> idx[1] == k, state.active)
    if length(active_j) > 1
        popfirst!(active_j)
        log_prop_dens = sum(logpdf.(Normal(0,σ_prop), state.x[active_j])) + log_exp_logpdf(σ_prop, priors.σ.a[k])
        log_new_dens = sum(logpdf.(Normal(0,priors.σ.σ[k]), state.x[active_j])) + log_exp_logpdf(log(priors.σ.σ[k]), priors.σ.a[k])
        α = min(1, exp(log_prop_dens - log_new_dens))
        acc = 0
        if rand() < α
            acc = 1
            priors.σ.σ[k] = copy(σ_prop)
        end
        priors.σ.ind += 1
        # Adaptation 
        if priors.σ.ind < 1_000
            priors.σ.h[k] = exp(log(priors.σ.h[k]) + (priors.σ.ind^(-0.6))*(α - 0.44))
        end
    end
end 


function log_exp_logpdf(logσ::Float64,a::Float64)
    return logσ - exp(logσ)*a
end

function weight_update!(state::State, priors::Prior, ω::FixedW, k::Int64)

end

function weight_update!(state::State, priors::Prior, ω::Beta, k::Int64)
    active_j = filter(idx -> idx[1] == k, state.active)
    priors.ω.ω[k] = rand(Distributions.Beta(priors.ω.a[k] + size(active_j,1) - 1, priors.ω.b[k] + prod(size(state.s,2)) - size(active_j,1) + 1))
end
