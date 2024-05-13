function time_build(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, settings::Settings, priors::Prior)
    Q_m = PriorityQueue{CartesianIndex, Float64}(Base.Order.Forward)
    Q_s = PriorityQueue{CartesianIndex, Float64}(Base.Order.Forward)
    for j in CartesianIndex(1,1):CartesianIndex(size(s,1),size(s,2))
        #if s[j]
        #    if j[2] > 1
        #        enqueue!(Q_m, j, merge_time(x, v, j))
        #    end
        #else
        #    new_split!(Q_s, s, 0.0, priors, j, dyn)
        #end
    end
    enqueue!(Q_m, CartesianIndex(0,0), Inf)
    enqueue!(Q_s, CartesianIndex(0,0), Inf)
    T_smp = exp_vector(settings, settings.smp_rate)
    T_h = exp_vector(settings, settings.h_rate)
    T_ref = exp_vector(settings, settings.r_rate)
    return Times(Q_s, Q_m, T_smp, T_h, T_ref)
end

function exp_vector(settings::Settings, rate::Float64)
    if rate > 0.0
        out = cumsum(rand(Exponential(1/rate), trunc(Int, rate*settings.max_time)))
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

function track_init!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, dat::PEMData, priors::Prior, settings::Settings, dyn::Dynamics)
    x_track = zeros(dat.p, size(dat.s,1), settings.max_ind)
    v_track = zeros(dat.p, size(dat.s,1), settings.max_ind)
    s_track = zeros(dat.p, size(dat.s,1), settings.max_ind)
    t_track = zeros(settings.max_ind)
    x_track[:,:,1] = copy(x)
    v_track[:,:,1] = copy(v)
    s_track[:,:,1] = copy(s)
    x_smp = zeros(dat.p, size(dat.s,1), settings.max_smp)
    x_smp[:,:,1] = copy(x)
    v_smp = zeros(dat.p, size(dat.s,1), settings.max_smp)
    v_smp[:,:,1] = copy(v)
    s_smp = zeros(dat.p, size(dat.s,1), settings.max_smp)
    s_smp[:,:,1] = copy(s)
    t_smp = zeros(settings.max_smp)
    # Hyperparameters depends on the prior specification so need functions
    h_track = h_track_init(priors, settings)
    h_store!(h_track, priors, dyn)
    h_smp = h_smp_init(priors, settings)
    h_store_smp!(h_smp, priors, dyn)
    t_track[1] = copy(t) 
    track = Track(x_track, v_track, s_track, t_track, h_track, x_smp, v_smp, s_smp, t_smp, h_smp)
    dyn.smp_ind += 1
    dyn.ind += 1
    return track
end

function track_store(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, dyn::Dynamics, priors::Prior, settings::Settings)
    track.x_track[:,:,dyn.ind] = copy(x)
    track.v_track[:,:,dyn.ind] = copy(v)
    track.s_track[:,:,dyn.ind] = copy(s)
    track.t_track[dyn.ind] = copy(t) 
    h_store!(track.h_track, priors, dyn)
    dyn.ind += 1
    # Print?
    if (dyn.ind % trunc(Int, settings.max_ind/10)) == 0.0
        sampler_update(dyn, settings, t)
    end
end

function smp_store(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, dyn::Dynamics, priors::Prior, settings::Settings)
    track.x_smp[:,:,dyn.smp_ind] = copy(x)
    track.v_smp[:,:,dyn.smp_ind] = copy(v)
    track.s_smp[:,:,dyn.smp_ind] = copy(s)
    track.t_smp[dyn.smp_ind] = copy(t) 
    h_store_smp!(track.h_smp, priors, dyn)
    dyn.smp_ind += 1
end

function post_sort!(track::Track, dyn::Dynamics, priors::Prior)
    smps = post_smps(track.x_smp[:,:,1:(dyn.smp_ind - 1)])
    v_smp = transpose(post_smps(track.v_smp[:,:,1:(dyn.smp_ind - 1)]))
    s_smps = post_smps(track.s_smp[:,:,1:(dyn.smp_ind - 1)])
    t_smp = track.t_smp[1:(dyn.smp_ind - 1)]
    smps_new = transform_smps(smps)
    h_smp = h_post(track.h_smp, priors, dyn)
    out = Dict("Smp_trans" => smps_new, "Smp_h" => h_smp, "Smp_s" =>s_smps, "Smp_x" => smps, "Smp_t" => t_smp, "Smp_v" => v_smp, 
                "Sk_x" => x_track, "Sk_v" => v_track, "Sk_s" => s_track, "Sk_h" => h_track, "t" => t_track, 
                "Eval" => dyn.sampler_eval)
    return out
end

function next_event!(t::Float64, times::Times, dyn::Dynamics)
    j1, t1 = CartesianIndex(0,0), Inf
    #j2, t2 = peek(times.Q_s)
    #j3, t3 = peek(times.Q_m)
    #j4, t4 = CartesianIndex(0,0), times.T_h
    j5, t5 = CartesianIndex(0,0), times.T_ref[1]
    j6, t6 = CartesianIndex(0,0), times.T_smp[1]
    #type = findmin([t1,t2,t3,t4,t5,t6])[2]
    #τ = [t1,t2,t3,t4,t5,t6][type]
    #j = [j1,j2,j3,j4,j5,j6][type]
    type = findmin([t1,Inf,Inf,Inf,t5,t6])[2]
    τ = [t1,Inf,Inf,Inf,t5,t6][type]
    j = CartesianIndex(0,0)
    dyn.next_event_type = type
    dyn.next_event_coord = j
    if τ <= t
        error("Sure, unless time is linear")
    end
    if type != 6
        if τ - t > dyn.t_bound
            dyn.next_bound_int = copy(dyn.t_bound)
            dyn.new_bound = true
            dyn.next_event_type = 1
        else
            dyn.next_bound_int = τ - t
        end
    else
        #τ_ = [t1,t2,t3,t4,t5][findmin([t1,t2,t3,t4,t5])[2]]
        τ_ = [t1,Inf,Inf,Inf,t5][findmin([t1,Inf,Inf,Inf,t5])[2]]
        if τ_ <= t
            error("Sure, unless time is linear")
        end
        if τ_ - t > dyn.t_bound
            dyn.next_bound_int = copy(dyn.t_bound)
            dyn.new_bound = true
            dyn.next_event_type = 1
        else
            dyn.next_bound_int = τ - t
        end
    end
    dyn.t_set = copy(t)
end

function h_track_init(priors::Union{BasicPrior,HyperPrior}, settings::Settings)
    return zeros(2,settings.max_ind)
end

function h_smp_init(priors::Union{BasicPrior,HyperPrior}, settings::Settings)
    return zeros(2,settings.max_smp)
end

function h_store!(h_track, priors::Union{HyperPrior}, dyn::Dynamics)
    h_track[1,dyn.ind] = priors.ω0
    h_track[2,dyn.ind] = priors.σ
end

function h_store!(h_track, priors::Union{BasicPrior}, dyn::Dynamics)

end

function h_store_smp!(h_track, priors::Union{HyperPrior}, dyn::Dynamics)
    h_track[1,dyn.smp_ind] = priors.ω0
    h_track[2,dyn.smp_ind] = priors.σ
end

function h_store_smp!(h_track, priors::Union{BasicPrior}, dyn::Dynamics)

end

function h_track_init(priors::Union{BasicPrior}, settings::Settings)
    return zeros(2,settings.max_ind)
end

function h_post(h_smp, priors::Union{BasicPrior}, dyn::Dynamics)
    return 0.0
end
function hyper_update!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, Q_f::Float64, Q_s::PriorityQueue, Q_m::PriorityQueue, dat::PEMData, j::CartesianIndex, priors::Union{BasicPrior}, dyn::Dynamics)
    error("Fixed prior structure - hyper update rate ")
end

function verbose_talk(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, dyn::Dynamics)
    println(dyn.last_type)
    println(t);println(x);println(s);println(v)
    println("---------------------------------------------------")
end