
function grid_update!(state::State, dyn::Dynamics, dat::PEMData, priors::Prior, grid::RJ)
    if rand() < 0.5
        if state.J > 1
            grid_merge!(state, dyn, priors)
        end
    else
        if state.J < priors.grid.max_points
            grid_split!(state, dyn, priors)
        end
    end
    AV_calc!(state, dyn, priors, dat)
    dat_update!(state, dyn, dat)
end

function grid_merge!(state::State, dyn::Dynamics, priors::Prior)
    # Select point to remove uniformly at random
    s_remove = rand(DiscreteUniform(2,state.J))
    u = state.x[1,s_remove]
    state_merge = merge_state(state, s_remove)
    A = log_MHG_ratio(state, state_merge, u, state.v[s_remove], dyn, priors)
    #println("Merge");println(state.J);println(exp(-A))
    if rand() < min(1, exp(-A))
        state.x, state.v, state.s, state.g, state.s_loc, state.t, state.J, state.b, state.active = copy(state_merge.x), copy(state_merge.v), copy(state_merge.s), copy(state_merge.g), copy(state_merge.s_loc), copy(state_merge.t), copy(state_merge.J), copy(state_merge.b), copy(state_merge.active)
    end
    AV_calc!(state, dyn, priors, dat)
    dat_update!(state, dyn, dat)
end

function merge_state(state::ECMC2, s_remove::Int64)
    state_merge = ECMC2(copy(state.x), copy(state.v), copy(state.s), copy(state.g), copy(state.s_loc), copy(state.t), copy(state.J), copy(state.b), copy(state.active))
    #if s_remove < state_merge.J
    #    state_merge.x[1,s_remove + 1] +=  state.x[1,s_remove]
    #end
    state_merge.x = state_merge.x[:, 1:end .!= s_remove]
    state_merge.v = state_merge.v[:, 1:end .!= s_remove]
    state_merge.s = state_merge.s[:, 1:end .!= s_remove]
    state_merge.g = state_merge.g[:, 1:end .!= s_remove]
    deleteat!(state_merge.s_loc, s_remove)
    state_merge.v /= norm(state_merge.v)
    state_merge.active = findall(state_merge.s)
    state_merge.J -= 1
    return state_merge
end

function merge_state(state::RWM, s_remove::Int64)
    state_merge = RWM(copy(state.x), copy(state.v), copy(state.s), copy(state.g), copy(state.s_loc), copy(state.t), copy(state.J), copy(state.b), copy(state.active), copy(state.step_size), copy(state.acc))
    #if s_remove < state_merge.J
    #    state_merge.x[1,s_remove + 1] +=  state.x[1,s_remove]
    #end
    state_merge.x = state_merge.x[:, 1:end .!= s_remove]
    state_merge.v = state_merge.v[:, 1:end .!= s_remove]
    state_merge.s = state_merge.s[:, 1:end .!= s_remove]
    state_merge.g = state_merge.g[:, 1:end .!= s_remove]
    deleteat!(state_merge.s_loc, s_remove)
    state_merge.v /= norm(state_merge.v)
    state_merge.active = findall(state_merge.s)
    state_merge.J -= 1
    return state_merge
end

function grid_split!(state::State, dyn::Dynamics, priors::Prior)
    # Find new location
    s_new = rand(Uniform(state.s_loc[1], priors.grid.max_time))
    u = rand(Normal(0, priors.grid.σ))
    state_new = split_state(state, s_new, u, priors)
    A = log_MHG_ratio(state_new, state, u, state_new.v[findfirst(state_new.s_loc .== s_new)], dyn, priors)
    #println("Split");println(state.J);println(exp(A))
    if rand() < min(1, exp(A))
        state.x, state.v, state.s, state.g, state.s_loc, state.t, state.J, state.b, state.active = copy(state_new.x), copy(state_new.v), copy(state_new.s), copy(state_new.g), copy(state_new.s_loc), copy(state_new.t), copy(state_new.J), copy(state_new.b), copy(state_new.active)
    end
    AV_calc!(state, dyn, priors, dat)
    dat_update!(state, dyn, dat)
end

function split_state(state::ECMC2, s_new::Float64, u::Float64, priors::Prior)
    # Place new point
    state_new = ECMC2(copy(state.x), copy(state.v), copy(state.s), copy(state.g), copy(state.s_loc), copy(state.t), copy(state.J), copy(state.b), copy(state.active))
    g_new = fill(true, size(state_new.x, 1), 1)
    ind_new = sortperm(vcat(state_new.s_loc, s_new))
    new_point = fill(u, size(state_new.x, 1), 1)
    state_new.s_loc = vcat(state_new.s_loc, s_new)[ind_new]
    state_new.x = hcat(state_new.x, new_point)[:,ind_new]
    v_new = split_velocity(state_new, priors)
    state_new.v[state.active] *= sqrt(1-v_new^2)
    state_new.v = hcat(state_new.v, v_new)[:,ind_new]
    state_new.s = hcat(state_new.s, fill(true, size(new_point)))[:,ind_new]
    state_new.g = hcat(state_new.g, g_new)[:,ind_new]
    state_new.active = findall(state_new.s)
    state_new.J = length(state_new.s_loc)
    # Update next point
    #if findfirst(state_new.s_loc .== s_new)[1] < state_new.J
    #    state_new.x[1, findfirst(state_new.s_loc .== s_new) + 1] -= u
    #end
    return state_new
end


function split_state(state::RWM, s_new::Float64, u::Float64, priors::Prior)
    # Place new point
    state_new = RWM(copy(state.x), copy(state.v), copy(state.s), copy(state.g), copy(state.s_loc), copy(state.t), copy(state.J), copy(state.b), copy(state.active), copy(state.step_size), copy(state.acc))
    g_new = fill(true, size(state_new.x, 1), 1)
    ind_new = sortperm(vcat(state_new.s_loc, s_new))
    new_point = fill(u, size(state_new.x, 1), 1)
    state_new.s_loc = vcat(state_new.s_loc, s_new)[ind_new]
    state_new.x = hcat(state_new.x, new_point)[:,ind_new]
    v_new = split_velocity(state_new, priors)
    state_new.v[state.active] *= sqrt(1-v_new^2)
    state_new.v = hcat(state_new.v, v_new)[:,ind_new]
    state_new.s = hcat(state_new.s, fill(true, size(new_point)))[:,ind_new]
    state_new.g = hcat(state_new.g, g_new)[:,ind_new]
    state_new.active = findall(state_new.s)
    state_new.J = length(state_new.s_loc)
    # Update next point
    #if findfirst(state_new.s_loc .== s_new)[1] < state_new.J
    #    state_new.x[1, findfirst(state_new.s_loc .== s_new) + 1] -= u
    #end
    return state_new
end

function log_MHG_ratio(state_split::State, state_curr::State, u::Float64, v::Float64, dyn::Dynamics, priors::Prior)
    AV_calc!(state_curr, dyn, priors, dat)
    dat_update!(state_curr, dyn, dat)
    U1 = U_new!(state_curr, dyn, priors)[1] 
    AV_calc!(state_split, dyn, priors, dat)
    dat_update!(state_split, dyn, dat)
    U2 = U_new!(state_split, dyn, priors)[1] 
    logpriors = logpdf(Poisson(priors.grid.Γ*(priors.grid.max_time - state_curr.s_loc[1])*priors.ω.ω[1]), state_split.J - 1) - 
                logpdf(Poisson(priors.grid.Γ*(priors.grid.max_time - state_curr.s_loc[1])*priors.ω.ω[1]), state_curr.J - 1)
    prop_terms = -logpdf(Normal(0, priors.grid.σ), u) - log(state_split.J - 1)
    A = -U2 + U1 + prop_terms + logpriors
    if state_split.J > priors.grid.max_points
        A = -Inf
    end
    return A
end 

function Jacobian(state_curr::ECMC2, v::Float64)
    #return log(2*sphere_area(size(state_curr.active,1) - 1)/(sphere_area(size(state_curr.active,1))*(size(state_curr.active,1)))) + log(abs(v)*sqrt(1-v^2)^(size(state_curr.active,1)-2))
    0.0
end

function Jacobian(state_curr::RWM, v::Float64)
    0.0
end