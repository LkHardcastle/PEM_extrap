include("Types.jl")
include("Helper.jl")
include("Potential.jl")
include("ZigZag.jl")

function pem_sample(x0::Matrix{Float64}, s0::Matrix{Bool}, v0::Matrix{Float64}, t0::Float64, dat::PEMData, priors::Prior, settings::Settings, dyn::Dynamics)
    x, s, v, t = copy(x0), copy(s0), copy(v0), copy(t0)
    times = time_build(x, v, s, settings, priors)
    track = track_init!(x, v, s, t, dat, priors, settings, dyn)
    next_event!(t, times, dyn)
    ∇U_bound!(x, v, s, dat, priors, CartesianIndex(1,1), dyn)
    sampler_eval.bounds += 1
    while dyn.ind < settings.max_ind
        # Generate next event time
        if settings.verbose
            verbose_talk(x, v, s, t, dyn)
        end
        x, v, s, t = sampler_inner!(x, s, v, t, dat, priors, dyn)
        if dyn.last_type != "Sample"        
            track_store!(x, v, s, t, dyn, priors, settings)
        else
            smp_store!(x, v, s, t, dyn, priors, settings)
            if dyn.smp_ind > settings.max_smp
                println("Max samples reached")
                break
            end
        end
    end
    out = post_sort!(track, dyn, priors)
    # Final things
    return out
end

function sampler_inner!(x::Matrix{Float64}, s::Matrix{Bool}, v::Matrix{Float64}, t::Float64, dat::PEMData, priors::Prior, dyn::Dynamics)
    inner_stop = false
    while !inner_stop
        τ = copy(t)
        t_prop = poisson_time(dyn.a, dyn.b, rand())
        t_, action = findmin(t_prop, dyn.next_bound_int)
        if action == 1
            # Update next thing to 
            dyn.next_event_type = 1
            dyn.next_event_int = t_ - τ
            dyn.new_bound = false
        end
        # Update x
        x .+= v.*dyn.next_event_int
        if dyn.next_event_type == 1
            if dyn.new_bound
                sampler_eval.bounds += 1
                next_event!(t, times, dyn)
                ∇U_bound!(x, v, s, dat, priors, CartesianIndex(1,1), dyn)
            else
                dyn.sampler_eval.flip_attempts += 1
                inner_stop = flip_attempt!()
            end
        elseif dyn.next_event_type ∈ [2,3,4]
            error("Not yet")
        #elseif dyn.next_event_type == 2
        #    dyn.sampler_eval.splits += 1
        #    dyn.last_type = "Split"
        #    split_int!()
        #    inner_stop = true
        #    next_event!(t, times, dyn)
        #    ∇U_bound!(x, v, s, dat, priors, CartesianIndex(1,1), dyn)
        #elseif dyn.next_event_type == 3
        #    if rand() < priors.p_split
        #        dyn.sampler_eval.merges += 1
        #        merge_int!()
        #        dyn.last_type = "Merge"
        #        inner_stop = true
        #        next_event!(t, times, dyn)
        #        ∇U_bound!(x, v, s, dat, priors, CartesianIndex(1,1), dyn)
        #    else
        #        delete!(Q_m, j)
        #        enqueue!(Q_m, j, Inf)
        #        next_event!(t, times, dyn)
        #        ∇U_bound!(x, v, s, dat, priors, CartesianIndex(1,1), dyn)
        #    end
        #elseif dyn.next_event_type == 4
        #    hyper_update!()
        #    dyn.last_type = "Hyper"
        #    inner_stop = true
        #    next_event!(t, times, dyn)
        #    ∇U_bound!(x, v, s, dat, priors, CartesianIndex(1,1), dyn)
        elseif dyn.next_event_type == 5
            dyn.last_type = "Refresh"
            inner_stop = true
            next_event!(t, times, dyn)
            ∇U_bound!(x, v, s, dat, priors, CartesianIndex(1,1), dyn)
        elseif dyn.next_event_type == 6
            dyn.last_type = "Sample"
            inner_stop = true
        end
    end
    return x, v, s, t
end