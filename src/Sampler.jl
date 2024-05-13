function pem_sample()
    ### Setup

    while dyn.ind < settings.max_ind
        sampler_inner!()
        store_state!()
        store_smps!()
    end
end

function sampler_inner!()
    ## Evaluate potential at current point to get constants
    U, ∂U, ∂2U = U_new!(state, dyn, priors, dat)
    ## Get next deterministic event and evaluate at that point
    dyn.t_det = get_time!(dyn)
    U_det, ∂U_det = U(state, dyn.t_det, dyn, priors)
    ## If potential decreasing at that point jump to it and break
    if ∂U_det < 0.0
        update!()
        event!()
        break
    end
    ## Elseif potential decreasing at initialpoint line search for point where gradient begins to increase
    if ∂U < 0.0
        t_switch = grad_optim(∂U, ∂2U, state, dyn, priors)
        U, ∂U = U(state, t_switch, dyn, priors)
    end
    ## Generate uniform r.v and check if deterministic time is close enough - if so break
    V = rand()
    if U_det - U < -log(V)
        update!()
        event!()
    end
    ## Generate next time via time-scale transformation
    ## Need to be careful
    t_event = potential_optim(V, U, ∂U, state, dyn, priors)
    update!()
    flip!()
    ## Exit
end

function get_time!()

end