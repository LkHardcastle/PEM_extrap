function sampler_inner!(state::RWM, dyn::Dynamics, priors::Prior, dat::PEMData, times::Times, settings::Settings)
    metropolis_step!(state, dyn, priors, dat)
    grid_update!(state, dyn, dat, priors, priors.grid)
end

function split_inner!(state::RWM, dyn::Dynamics, priors::Prior, dat::PEMData, times::Times, settings::Settings)
    metropolis_step!(state, dyn, priors, dat)
    grid_update!(state, dyn, dat, priors, priors.grid)
end

function metropolis_step!(state::RWM, dyn::Dynamics, priors::Prior, dat::PEMData)
    state.t += 1
    u = rand(Normal(0.0, state.step_size), state.J)
    AV_calc!(state, dyn, priors, dat)
    dat_update!(state, dyn, dat)
    U1 = U_new!(state, dyn, priors)[1] 
    state_prop = RWM(copy(state.x), copy(state.v), copy(state.s), copy(state.g), copy(state.s_loc), copy(state.t), copy(state.J), copy(state.b), copy(state.active), copy(state.step_size), copy(state.acc))
    state_prop.x[1,1:state.J] += u
    AV_calc!(state_prop, dyn, priors, dat)
    dat_update!(state_prop, dyn, dat)
    U2 = U_new!(state_prop, dyn, priors)[1] 
    A = -U2 + U1
    if rand() < min(1, exp(A))
        state.acc += 1
        state.x[1,1:state.J] += u
    end
end