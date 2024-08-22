
function hyper_update!(state::State, dyn::Dynamics, dat::PEMData, priors::Prior)
    variance_update!(state, priors, priors.σ)
    weight_update!(state, priors, priors.ω)
    grid_update!(state, dyn, dat, priors, priors.grid)
end

function grid_update!(state::State, dyn::Dynamics, dat::PEMData, priors::Prior, Grid::Fixed)
end

function grid_update!(state::State, dyn::Dynamics, dat::PEMData, priors::Prior, grid::Cts)
    rem_ind = findall(sum.(eachcol(state.s)) .!= 0.0)
    state.x, state.v, state.s, state.g, state.s_loc = state.x[:,rem_ind], state.v[:,rem_ind], state.s[:,rem_ind], state.g[:,rem_ind], state.s_loc[rem_ind]
    J_curr = sum(state.s)
    J_new = min(rand(Poisson(priors.grid.max_time*priors.grid.Γ*(1 - priors.ω.ω))), priors.grid.max_points - J_curr)
    J_row = rand(DiscreteUniform(1,size(state.x,1)), J_new)
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
    state.ξ = hcat(state.ξ, zero_mat)[:,ind_new]
    state.s = hcat(state.s, fill(false, size(zero_mat)))[:,ind_new]
    state.g = hcat(state.g, g_new)[:,ind_new]
    state.active = findall(state.s)
    state.J = length(state.s_loc)
    dat_update!(state, dyn, dat)
end

function dat_update!(state::State, dyn::Dynamics, dat::PEMData)
    #println(state.s_loc)
    L = size(dat.UQ, 2)
    #println(state.s_loc)
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


function log_exp_logpdf(logσ::Float64,a::Float64)
    return logσ - exp(logσ)*a
end

function weight_update!(state::State, priors::Prior, ω::FixedW)

end

function weight_update!(state::State, priors::Prior, ω::Beta)
    priors.ω.ω = rand(Distributions.Beta(priors.ω.a + size(state.active,1) - 1, priors.ω.b + prod(size(state.s)) - size(state.active,1) + 1))
end
