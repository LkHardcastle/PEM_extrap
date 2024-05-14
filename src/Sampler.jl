function pem_sample()
    ### Setup
    times = time_setup(state, settings)
    while dyn.ind < settings.max_ind
        if settings.verbose
            verbose(state)
        end
        sampler_inner!(state, dyn, priors, dat, time)
        store_state!(state, storage, dyn; skel = settings.skel)
        store_smps!(state, storage, dyn, times)
    end
    out = sampler_end(state, dynamics)
    return out  
end

function time_setup(state::State, settings::Settings)
    Q_m = PriorityQueue{CartesianIndex, Float64}(Base.Order.Forward)
    Q_s = PriorityQueue{CartesianIndex, Float64}(Base.Order.Forward)
    T_smp = exp_vector(settings, settings.smp_rate)
    T_h = exp_vector(settings, settings.h_rate)
    T_ref = exp_vector(settings, settings.r_rate)
    return Times(Q_m, Q_s, T_ref, T_h, T_smp)
end

function sampler_inner!(state::State, dyn::Dynamics, priors::Prior, dat::PEMData, times::Times)
    ## Evaluate potential at current point to get constants
    U, ∂U, ∂2U = U_new!(state, dyn, priors, dat)
    ## Get next deterministic event and evaluate at that point
    get_time!(dyn, times)
    U_det, ∂U_det = U(state, dyn.t_det, dyn, priors)
    ## If potential decreasing at that point jump to it and break
    if ∂U_det < 0.0
        update!(state, dyn.t_det - state.t)
        event!(state, dyn, priors)
    end
    ## Elseif potential decreasing at initialpoint line search for point where gradient begins to increase
    t_switch = 0.0
    if ∂U < 0.0
        t_switch = grad_optim(∂U, ∂2U, state, dyn, priors)
        U, ∂U = U(state, t_switch, dyn, priors)
    end
    ## Generate uniform r.v and check if deterministic time is close enough - if so break
    V = rand()
    if U_det - U < -log(V)
        update!(state, dyn.t_det - state.t)
        event!(state, dyn, priors)
    end
    ## Generate next time via time-scale transformation
    ## Need to be careful
    t_event = potential_optim(V, U, ∂U, state, dyn, priors)
    update!(state, t_switch + t_event)
    flip!(state, dat, priors, dyn)
    ## Exit
end

function get_time!(dyn::Dynamics, times::Times)
    dyn.t_det, dyn.next_event = findmin([Inf, Inf, times.refresh, times.hyper])
end

function store_state!(state::State, storage::Storage, dyn::Dynamics; skel = true)
    if !skel
        dyn_ind -= 1
    end
    storage.x[:,:,dyn.ind] = copy(state.x)
    storage.v[:,:,dyn.ind] = copy(state.v)
    storage.s[:,:,dyn.ind] = copy(state.s)
    storage.t[dyn.ind] = copy(state.t)
    dyn.ind += 1
end

function store_smps!(state::State, storage::Storage, dyn::Dynamics, times::Times)
    ind_end = findfirst(times.smps .> state.t) - 1
    if !isnothing(ind_end)
        t_old = storage.t[dyn.ind - 1]
        x_old = storage.x[:,:,dyn.ind - 1]
        v_old = storage.v[:,:,dyn.ind - 1]
        s_old = storage.s[:,:,dyn.ind - 1]
        for i in 1:ind_end
            storage.t_smp[dyn.smp_ind] = times.smps[i]
            storage.x_smp[:,:,dyn.smp_ind] = x_old + v_old*(times.smps[i] - t_old)
            storage.v_smp = copy(v_old)
            storage.s_smp = copy(s_old)
            dyn.smp_ind += 1
        end
        deleteat!(times.smps, 1:ind_end)
    end
end

function sampler_end!(state::State, dynamics::Dynamics)

end

function verbose(state::State)
end