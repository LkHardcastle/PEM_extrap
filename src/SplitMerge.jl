function merge_time(state::State, j::CartesianIndex, priors::Prior)
    if rand() < priors.p_split
        if (state.x[j] > 0.0 && state.v[j] < 0.0) || (state.x[j] < 0.0 && state.v[j] > 0.0)
            return abs(state.x[j])/abs(state.v[j])
        else
            return Inf
        end 
    else
        return Inf
    end
end

function split_rate(state::State, priors::BasicPrior)
    rate = priors.p_split*(priors.ω/(1 - priors.ω))*(sqrt(2*pi*priors.σ^2))^-1
    J = 2*sphere_area(size(state.active,1) - 1)/(sphere_area(size(state.active,1))*size(state.active,1))
    return rate*J
end
function sphere_area(d::Int64)
    return 2*π^(0.5*d+0.5)/gamma(0.5*d+0.5)
end

function split!(state::State)
    j = state.active[rand(DiscreteUniform(1,size(state.active,1)))]
    if j[2] != 1
        state.s[j] = true
        # Add to state.active
        state.active = findall(state.s)
        # New velocities
        a = split_velocity(state)
        state.v = (1-a^2)*state.v
        state.v[j] = a
    end
end

function split_velocity(state::State)
    # Draw new velocity 
    return sqrt(1 - rand()^(2/size(state.active,1)))*(2*rand(Bernoulli(0.5) - 1))
end


function merge!(state::State, j::CartesianIndex)
    state.s[j] = false
    state.v[j] = 0.0
    state.x[j] = 0.0
    # Remove from state.active
    state.active = findall(state.s)
    state.v[state.active] /= norm(state.v[state.active])
end

function split_time!(state::State, times::Times, priors::Prior)
    # Update split time
    if priors.p_split > 0.0
        times.T_split = size(state.active,1)*split_rate(state, priors)
    else 
        times.T_split = Inf
    end
end

function merge_time!(state::State, times::Times, priors::Prior)
    # Update merge times
    merge_curr = Inf
    j_curr = CartesianIndex(0,0)
    for j in state.active
        if j[2] > 1
            if rand() < priors.p_split
                merge_cand = merge_time(state, j, priors)
                if merge_cand < merge_curr
                    merge_curr = copy(merge_cand)
                    j_curr = copy(j)
                end
            end
        end
    end
    times.next_merge = copy(merge_curr)
    times.next_merge_index = copy(j_curr)
end

function merge_time_ref!(state::Union{BPS,ECMC}, times::Times, priors::Prior)
    merge_time!(state, times, priors)
end

function merge_time_ref!(state::ECMC2, times::Times, priors::Prior)
end