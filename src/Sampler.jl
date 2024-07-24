include("Types.jl")
include("Potential.jl")
include("Updating.jl")
include("SplitMerge.jl")
include("HyperUpdates.jl")

function pem_sample(state0::State, dat::PEMData, priors::Prior, settings::Settings)
    ### Setup
    state = copy(state0)
    times = time_setup(state, settings, priors)
    dyn = Dynamics(1, 1, 0.0, 0, copy(state.x), copy(state.x), copy(state.x), copy(state.x), copy(state.s), SamplerEval(zeros(2),0, 0))
    # Set up storage 
    if settings.skel == false
        dyn.ind = 2
    end
    storage = storage_start!(state, settings, dyn, priors)
    AV_calc!(state, dyn)
    W_calc!(state, dyn, dat)
    println(∇U(state, dat, dyn, priors))
    error("")
    println("Starting sampling")
    while dyn.ind < settings.max_ind
        if settings.verbose
            verbose(dyn, state)
        end
        sampler_inner!(state, dyn, priors, dat, times)
        store_state!(state, storage, dyn, priors; skel = settings.skel)
        store_smps!(state, storage, dyn, times, priors)
        stop = sampler_stop(state, dyn, settings)
        if stop
            break
        end
        #if dyn.ind % 10_000 == 0
        #    verbose(dyn, state)
        #end
    end
    out = sampler_end(storage, dyn, settings)
    return out  
end

function Base.copy(state::BPS)
    return BPS(copy(state.x), copy(state.v), copy(state.s), copy(state.t), copy(state.active))
end

function Base.copy(state::ECMC)
    return ECMC(copy(state.x), copy(state.v), copy(state.s), copy(state.t), copy(state.active))
end

function Base.copy(state::ECMC2)
    return ECMC2(copy(state.x), copy(state.v), copy(state.s), copy(state.t), copy(state.b), copy(state.active))
end

function storage_start!(state::State, settings::Settings, dyn::Dynamics, priors::Prior)
    storage = Storage(fill(Inf,size(state.x, 1),size(state.x, 2), settings.max_ind + 1),
                        fill(Inf,size(state.v, 1),size(state.v, 2), settings.max_ind + 1), 
                        fill(false,size(state.s, 1),size(state.s, 2), settings.max_ind + 1),
                        zeros(settings.max_ind+ 1),
                        fill(Inf, 2, settings.max_ind + 1),
                        fill(Inf,size(state.x, 1),size(state.x, 2), settings.max_smp + 1),
                        fill(Inf,size(state.v, 1),size(state.v, 2), settings.max_smp + 1), 
                        fill(false,size(state.s, 1),size(state.s, 2), settings.max_smp + 1),
                        zeros(settings.max_smp + 1),
                        fill(Inf, 2, settings.max_smp + 1))
    store_state!(state, storage, dyn, priors; skel = settings.skel)
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

function time_setup(state::State, settings::Settings, priors::Prior)
    merge_curr = Inf
    j_curr = CartesianIndex(0,0)
    for j in state.active
        if j[2] > 1
            if rand() < priors.p_split
                merge_cand = merge_time(state, j, priors)
                if merge_cand < merge_curr
                    merge_curr = copy(merge_cand)
                    j_curr = CartesianIndex(j[1],j[2])
                end
            end
        end
    end
    if priors.p_split > 0.0
        T_split = rand(Exponential(1/(size(findall(state.s .== false),1)*split_rate(state, priors))))
    else 
        T_split = Inf
    end
    T_smp = exp_vector(settings, settings.smp_rate)
    T_h = exp_vector(settings, settings.h_rate)
    T_ref = exp_vector(settings, settings.r_rate)
    return Times(T_split, merge_curr, j_curr, T_ref, T_h, T_smp)
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
        event!(state, dat, dyn, priors, times)
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
            event!(state, dat, dyn, priors, times)
        else
            ## Generate next time via time-scale transformation
            if isinf(dyn.t_det)
                t_event = find_zero(x -> U_eval(state, x + t_switch, dyn, priors)[1] - Uθt + log(V), (0.0, 5), A42())
            else
                t_event = find_zero(x -> U_eval(state, x + t_switch, dyn, priors)[1] - Uθt + log(V), (0.0, dyn.t_det - state.t), A42())
            end
            update!(state, t_switch + t_event)
            flip!(state, dat, dyn, priors)
            merge_time!(state, times, priors)
        end
    end
end

function get_time!(dyn::Dynamics, times::Times)
    dyn.t_det, dyn.next_event = findmin([times.next_split, times.next_merge, times.refresh[1], times.hyper[1]])
end

function store_state!(state::State, storage::Storage, dyn::Dynamics, priors::BasicPrior; skel = true)
    if !skel
        dyn.ind -= 1
    end
    storage.x[:,:,dyn.ind] = copy(state.x)
    storage.v[:,:,dyn.ind] = copy(state.v)
    storage.s[:,:,dyn.ind] = copy(state.s)
    storage.t[dyn.ind] = copy(state.t)
    storage.h[1,dyn.ind] = copy(priors.σ.σ) 
    storage.h[2,dyn.ind] = copy(priors.ω.ω)
    dyn.ind += 1
end

function store_state!(state::State, storage::Storage, dyn::Dynamics, priors::ARPrior; skel = true)
    if !skel
        dyn.ind -= 1
    end
    storage.x[:,:,dyn.ind] = copy(state.x)
    storage.v[:,:,dyn.ind] = copy(state.v)
    storage.s[:,:,dyn.ind] = copy(state.s)
    storage.t[dyn.ind] = copy(state.t)
    storage.h[1,dyn.ind] = copy(priors.σ0) 
    storage.h[2,dyn.ind] = copy(priors.ω.ω)
    dyn.ind += 1
end

function store_smps!(state::State, storage::Storage, dyn::Dynamics, times::Times, priors::BasicPrior)
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
        storage.h_smp[1,dyn.smp_ind] = copy(priors.σ.σ) 
        storage.h_smp[2,dyn.smp_ind] = copy(priors.ω.ω)
        dyn.smp_ind += 1
    end
    deleteat!(times.smps, 1:ind_end)
end

function store_smps!(state::State, storage::Storage, dyn::Dynamics, times::Times, priors::ARPrior)
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
        storage.h_smp[1,dyn.smp_ind] = copy(priors.σ0) 
        storage.h_smp[2,dyn.smp_ind] = copy(priors.ω.ω)
        dyn.smp_ind += 1
    end
    deleteat!(times.smps, 1:ind_end)
end

function sampler_end(storage::Storage, dyn::Dynamics, settings::Settings)
    if settings.skel
        out = Dict("Sk_x" => storage.x[:,:,1:(dyn.ind-1)], "Sk_v" => storage.v[:,:,1:(dyn.ind-1)], "Sk_s" => storage.s[:,:,1:(dyn.ind-1)], "Sk_t" => storage.t[1:(dyn.ind-1)], 
                    "Sk_h" => storage.h[:,1:(dyn.ind-1)],
                    "Smp_x" => storage.x_smp[:,:,1:(dyn.smp_ind-1)], "Smp_v" => storage.v_smp[:,:,1:(dyn.smp_ind-1)], "Smp_s" => storage.s_smp[:,:,1:(dyn.smp_ind-1)], 
                    "Smp_t" => storage.t_smp[1:(dyn.smp_ind-1)], "Smp_h" => storage.h_smp[:,1:(dyn.smp_ind-1)],
                    "Smp_trans" => transform_smps(storage.x_smp[:,:,1:(dyn.smp_ind-1)]), "Eval" => dyn.sampler_eval) 
    else
        out = Dict("Smp_x" => storage.x_smp[:,:,1:(dyn.smp_ind-1)], "Smp_v" => storage.v_smp[:,:,1:(dyn.smp_ind-1)], "Smp_s" => storage.s_smp[:,:,1:(dyn.smp_ind-1)], 
                    "Smp_t" => storage.t_smp[1:(dyn.smp_ind-1)], "Smp_h" => storage.h_smp[:,1:(dyn.smp_ind-1)],
                    "Smp_trans" => transform_smps(storage.x_smp[:,:,1:(dyn.smp_ind-1)]),"Eval" => dyn.sampler_eval) 
    end
    return out
end

function verbose(dyn::Dynamics, state::State)
    println("----------------------")
    print("Iteration: ");print(dyn.ind);print("\n");
    println(dyn.next_event)
    println(state.t)
    println(state.active)
    println(vec(state.x))
    println(vec(state.v))
    println("----------------------")
end