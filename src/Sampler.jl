include("Types.jl")
include("Potential.jl")
include("Updating.jl")

function pem_sample(state0::State, dat::PEMData, priors::Prior, settings::Settings)
    ### Setup
    state = copy(state0)
    times = time_setup(state, settings)
    dyn = Dynamics(1, 1, 0.0, 0, Base.copy(state.x), Base.copy(state.x), Base.copy(state.x), Base.copy(state.x), Base.copy(state.x), SamplerEval(zeros(2),0))
    # Set up storage 
    storage = storage_start!(state, settings, dyn)
    k = 1
    while dyn.ind < settings.max_ind
        if settings.verbose
            verbose(dyn, state)
        end
        sampler_inner!(state, dyn, priors, dat, times)
        store_state!(state, storage, dyn; skel = settings.skel)
        store_smps!(state, storage, dyn, times)
        stop = sampler_stop(state, dyn, settings)
        if stop
            break
        end
        if dyn.ind % 10_000 == 0
            verbose(dyn, state)
        end
    end
    out = sampler_end(storage, dyn, settings)
    return out  
end

function Base.copy(state::BPS)
    return BPS(state.x, state.v, state.s, state.t, state.active)
end

function Base.copy(state::ECMC)
    return ECMC(state.x, state.v, state.s, state.t, state.active)
end

function storage_start!(state::State, settings::Settings, dyn::Dynamics)
    storage = Storage(fill(Inf,size(state.x, 1),size(state.x, 2), settings.max_ind + 1),
                        fill(Inf,size(state.v, 1),size(state.v, 2), settings.max_ind + 1), 
                        fill(false,size(state.s, 1),size(state.s, 2), settings.max_ind + 1),
                        zeros(settings.max_ind+ 1),
                        fill(Inf,size(state.x, 1),size(state.x, 2), settings.max_smp + 1),
                        fill(Inf,size(state.v, 1),size(state.v, 2), settings.max_smp + 1), 
                        fill(false,size(state.s, 1),size(state.s, 2), settings.max_smp + 1),
                        zeros(settings.max_smp + 1))
    store_state!(state, storage, dyn; skel = settings.skel)
    return storage
end

function sampler_stop(state::State, dyn::Dynamics, settings::Settings)
    if dyn.ind >= settings.max_ind
        println("Stopping for max_iter")
        return true
    end
    if dyn.smp_ind >= settings.max_smp
        println("Stopping for max_smp")
        return true
    end
    if state.t >= settings.max_time
        println("Stopping for max_time")
        return true
    end
    return false
end

function time_setup(state::State, settings::Settings)
    Q_m = PriorityQueue{CartesianIndex, Float64}(Base.Order.Forward)
    Q_s = PriorityQueue{CartesianIndex, Float64}(Base.Order.Forward)
    T_smp = exp_vector(settings, settings.smp_rate)
    T_h = exp_vector(settings, settings.h_rate)
    T_ref = exp_vector(settings, settings.r_rate)
    return Times(Q_s, Q_m, T_ref, T_h, T_smp)
end

function exp_vector(settings::Settings, rate::Float64)
    if rate > 0.0
        out = cumsum(rand(Exponential(1/rate), trunc(Int, rate*settings.max_time + 100)))
        if out[end] < settings.max_time
            while out[end] < settings.max_time
                push!(out, out[end] + rand(Exponential(1/rate)))
            end
        end
    else
        out = [Inf]
    end
    return out
end

function sampler_inner!(state::State, dyn::Dynamics, priors::Prior, dat::PEMData, times::Times)
    ## Evaluate potential at current point to get constants
    Uθt, ∂U, ∂2U = U_new!(state, dyn, priors, dat)
    ## Get next deterministic event and evaluate at that point
    get_time!(dyn, times)
    if !isinf(dyn.t_det)
        U_det, ∂U_det = U_eval(state, dyn.t_det - state.t, dyn, priors)
    else
        U_det, ∂U_det = Inf, Inf
    end
    #println([U_det, ∂U_det])
    ## If potential decreasing at that point jump to it and break
    if ∂U_det < 0.0
        update!(state, dyn.t_det - state.t)
        event!(state, dyn, priors, times)
    else
        ## Elseif potential decreasing at initialpoint line search for point where gradient begins to increase
        t_switch = 0.0
        if ∂U < 0.0
            t_switch = grad_optim(∂U, ∂2U, state, dyn, priors)
            Uθt, ∂U = U_eval(state, t_switch, dyn, priors)
        end
        ## Generate uniform r.v and check if deterministic time is close enough - if so break
        V = rand()
        if U_det - Uθt < -log(V)
            update!(state, dyn.t_det - state.t)
            event!(state, dyn, priors, times)
        else
            ## Generate next time via time-scale transformation
            ## Need to be careful
            t_event = potential_optim(t_switch, V, Uθt, ∂U, state, dyn, priors)
            update!(state, t_switch + t_event)
            flip!(state, dat, priors)
        end
    end
end

function get_time!(dyn::Dynamics, times::Times)
    dyn.t_det, dyn.next_event = findmin([Inf, Inf, times.refresh[1], times.hyper[1]])
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
    ind_end = findfirst(times.smps .> state.t)
    if isnothing(ind_end)
        ind_end = size(times.smps,1)
    end
    ind_end -= 1
    t_old = storage.t[dyn.ind - 2]
    x_old = storage.x[:,:,dyn.ind - 2]
    v_old = storage.v[:,:,dyn.ind - 2]
    s_old = storage.s[:,:,dyn.ind - 2]
    for i in 1:ind_end
        storage.t_smp[dyn.smp_ind] = times.smps[i]
        storage.x_smp[:,:,dyn.smp_ind] = x_old + v_old*(times.smps[i] - t_old)
        storage.v_smp[:,:,dyn.smp_ind] = copy(v_old)
        storage.s_smp[:,:,dyn.smp_ind] = copy(s_old)
        dyn.smp_ind += 1
    end
    deleteat!(times.smps, 1:ind_end)
end

function sampler_end(storage::Storage, dyn::Dynamics, settings::Settings)
    if settings.skel
        out = Dict("Sk_x" => storage.x[:,:,1:(dyn.ind-1)], "Sk_v" => storage.v[:,:,1:(dyn.ind-1)], "Sk_s" => storage.s[:,:,1:(dyn.ind-1)], "Sk_t" => storage.t[1:(dyn.ind-1)],
                    "Smp_x" => storage.x_smp[:,:,1:(dyn.smp_ind-1)], "Smp_v" => storage.v_smp[:,:,1:(dyn.smp_ind-1)], "Smp_s" => storage.s_smp[:,:,1:(dyn.smp_ind-1)], "Smp_t" => storage.t_smp[1:(dyn.smp_ind-1)],
                    "Eval" => dyn.sampler_eval) 
    else
        out = Dict("Smp_x" => storage.x_smp[:,:,1:(dyn.smp_ind-1)], "Smp_v" => storage.v_smp[:,:,1:(dyn.smp_ind-1)], "Smp_s" => storage.s_smp[:,:,1:(dyn.smp_ind-1)], "Smp_t" => storage.t_smp[1:(dyn.smp_ind-1)],
                    "Eval" => dyn.sampler_eval) 
    end
    return out
end

function verbose(dyn::Dynamics, state::State)
    println("----------------------")
    print("Iteration: ");print(dyn.ind);print("\n")
    println(state.t)
    println(state.active)
    println(vec(state.x[state.active]))
    println(vec(state.v[state.active]))
    println("----------------------")
end