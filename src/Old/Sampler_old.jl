include("Types.jl")
include("Potential.jl")
include("Fixed priors.jl")
#include("GeomPrior.jl")
include("HyperPrior.jl")
include("SplitMerge.jl")
function pem_sample(x0::Matrix{Float64}, s0::Matrix{Bool}, v0::Matrix{Float64}, t0::Float64, dat::PEMData, priors::Prior, settings::Settings)
    x, s, v, t = copy(x0), copy(s0), copy(v0), copy(t0)
    Q_f = 0.0
    Q_s = PriorityQueue{CartesianIndex, Float64}(Base.Order.Forward)
    Q_m = PriorityQueue{CartesianIndex, Float64}(Base.Order.Forward)
    #w_order = w_order_init(s, priors)
    dyn = Dynamics(1, 1, settings.tb_init, zeros(size(x)), zeros(size(x)), zeros(size(x)), fill(false, size(x)), "Start", settings.v_abs, 0.1,
                    SamplerEval(0,0,0,0,0,0,0))
    start_queue!(Q_f, Q_s, Q_m, x, s, v, dat, dyn)  
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
    dyn.smp_ind += 1
    dyn.ind += 1
    i = 2
    while dyn.ind < settings.max_ind
        # Generate next event time
        if settings.verbose
            println(dyn.last_type)
            println(t);println(x);println(s);println(v);
            println(Q_f);println(Q_s);println(Q_m)
            println("---------------------------------------------------")
        end
        x, v, s, t = sampler_inner!(x, v, s, t, Q_f, Q_s, Q_m, priors, dat, dyn)
        if dyn.last_type != "Sample"        
            x_track[:,:,dyn.ind] = copy(x)
            v_track[:,:,dyn.ind] = copy(v)
            s_track[:,:,dyn.ind] = copy(s)
            t_track[dyn.ind] = copy(t) 
            h_store!(h_track, priors, dyn)
            dyn.ind += 1
            # Print?
            if (dyn.ind % trunc(Int, settings.max_ind/10)) == 0.0
                sampler_update(dyn, settings, t)
            end
        else
            x_smp[:,:,dyn.smp_ind] = copy(x)
            v_smp[:,:,dyn.smp_ind] = copy(v)
            s_smp[:,:,dyn.smp_ind] = copy(s)
            t_smp[dyn.smp_ind] = copy(t)
            h_store_smp!(h_smp, priors, dyn)
            dyn.smp_ind += 1
            if dyn.smp_ind > settings.max_smp
                println("Max samples reached")
                break
            end
        end
    end
    smps = post_smps(x_smp[:,:,1:(dyn.smp_ind - 1)])
    v_smp = transpose(post_smps(v_smp[:,:,1:(dyn.smp_ind - 1)]))
    s_smps = post_smps(s_smp[:,:,1:(dyn.smp_ind - 1)])
    t_smp = t_smp[1:(dyn.smp_ind - 1)]
    smps_new = transform_smps(smps)
    smps = transpose(smps)
    h_smp = h_post(h_smp, priors, dyn)
    # Final things
    return Dict("Smp_trans" => smps_new, "Smp_h" => h_smp, "Smp_s" =>s_smps, "Smp_x" => smps, "Smp_t" => t_smp, "Smp_v" => v_smp, "Sk_x" => x_track, "Sk_v" => v_track, "Sk_s" => s_track, "Sk_h" => h_track, "t" => t_track, "Eval" => dyn.sampler_eval)
end


function start_queue!(Q_f::Float64, Q_s::PriorityQueue, Q_m::PriorityQueue, x::Matrix{Float64}, s::Matrix{Bool}, v::Matrix{Float64}, dat::PEMData,  dyn::Dynamics)
    ∇U_bound!(x, v, s, dat, priors, CartesianIndex(1,1), dyn)
    Q_f = poisson_time(sum(dyn.a), sum(dyn.b), rand())
    dyn.t_set = 0.0
    if Q_f > dyn.t_bound
        # If greater than bounding interval
        # set to bounding interval 
        Q_f = dyn.t_bound
        dyn.new_t = true
    else
        dyn.new_t = false
    end
    for j in findall(s)
        ## Merge queue
        if j[2] > 1
            enqueue!(Q_m, j, merge_time(x, v, j))
        end
    end
    for j in findall(s .== false)
        ## Split queue 
        new_split!(Q_s, s, 0.0, priors, j, dyn)
    end
end

function sampler_inner!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, Q_f::Float64, Q_s::PriorityQueue, Q_m::PriorityQueue, priors::Prior, dat::PEMData, dyn::Dynamics)
    inner_stop = false
    while !inner_stop
        τ = copy(t)
        t, j, type = event_find(Q_f, Q_s, Q_m, τ, settings)
        # Update x
        t_ = t - τ
        x .+= v.*t_
        if type == 1
            if dyn.new_t
                new_bound!(Q_f, t, x, v, s, priors, dat, dyn, j, false)
            else
                dyn.sampler_eval.flip_attempts += 1
                inner_stop = flip_attempt!(x, v, s, t, Q_f, Q_s, Q_m, dat, j, priors, dyn)
            end
        end
        if type == 2
            dyn.sampler_eval.splits += 1
            dyn.last_type = "Split"
            split_int!(x, v, s, t, Q_f, Q_s, Q_m, dat, j, priors, dyn)
            inner_stop = true
        end
        if type == 3
            if rand() < priors.p_split
                dyn.sampler_eval.merges += 1
                merge_int!(x, v, s, t, Q_f, Q_s, Q_m, dat, j, priors, dyn)
                dyn.last_type = "Merge"
                inner_stop = true
            else
                delete!(Q_m, j)
                enqueue!(Q_m, j, Inf)
            end
        end
        if type == 4
            hyper_update!(x, v, s, t, Q_f, Q_s, Q_m, dat, j, priors, dyn)
            dyn.last_type = "Hyper"
            inner_stop = true
        end
        if type == 5
            dyn.last_type = "Sample"
            inner_stop = true
        end
    end
    return x, v, s, t
end

function event_find(Q_f::Float64, Q_s::PriorityQueue, Q_m::PriorityQueue, τ::Float64, settings::Settings)
    j1, t1 = CartesianIndex(0,0), Q_f
    if !isempty(Q_s)
        j2, t2 = peek(Q_s)
    else
        j2, t2 = CartesianIndex(0,0), Inf
    end
    if !isempty(Q_m)
        j3, t3 = peek(Q_m)
    else
        j3, t3 = CartesianIndex(0,0), Inf
    end
    if settings.h_rate > 0.0
        t4 = τ + rand(Exponential(1/settings.h_rate))
        j4 = CartesianIndex(0,0)
    else
        t4 = Inf
        j4 = CartesianIndex(0,0)
    end
    if settings.smp_rate > 0.0
        t5 = τ + rand(Exponential(1/settings.smp_rate))
        j5 = CartesianIndex(0,0)
    else
        t5 = Inf
        j5 = CartesianIndex(0,0)
    end
    type = findmin([t1,t2,t3,t4,t5])[2]
    t = [t1,t2,t3,t4,t5][type]
    j = [j1,j2,j3,j4,j5][type]
    if τ > t
        println(t);println(τ)
        println(Q_f);println(Q_s);println(Q_m)
        error("Queueing problems")
    end
    return t, j, type
end

function flip_attempt!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, Q_f::Float64, Q_s::PriorityQueue, Q_m::PriorityQueue, dat::PEMData, j::CartesianIndex, priors::Prior, dyn::Dynamics)
    ∇U(x, s, dat, CartesianIndex(1,1))
    # True rate
    λ = max(v[j]*(∇U(x, s, dat, j)[1] + ∇U_p(x, s, j, priors)) , 0.0)
    # Upper bound
    Λ = dyn.a[j] + (t - dyn.t_set[j])*dyn.b[j]
    if  λ/Λ > 1 + 1e-5
        print("At iteration: ");print(dyn.ind);print("\n")
        println(λ);println(Λ); println(t)
        println(x);println(v)
        println(s);
        println(j);
        println(dyn.t_set[j]);
        println(dyn.a[j]);println(dyn.b[j])
        error("Incorrect flipping upper bound")
    end
    if λ/Λ > rand()
        dyn.sampler_eval.flips += 1
        v[j] = - v[j]
        new_bound!(Q_f, t, x, v, s, priors, dat, dyn)
        new_merge!(Q_m, t, x, v, s, j, false)
        dyn.last_type = "Flip"
        return true
    else
        new_bound!(Q_f, t, x, v, s, priors, dat, dyn)
        return false
    end
end



function neighbourhood(j::CartesianIndex, s::Matrix{Bool})
    int_start = findlast(s[j[1],begin:(j[2]-1)])
    if isnothing(int_start)
        int_start = 1
    end
    int_end = findfirst(s[j[1],(j[2]+1):end])
    if isnothing(int_end)
        int_end = j[2]
    else
        int_end += j[2]
    end
    nhood = findall(s[:,int_start:int_end])
    nhood = findall(s)[intersect(findall(last.(Tuple.(findall(s))) .<= int_end), findall(last.(Tuple.(findall(s))) .>= int_start))]
    k = 1
    for s_ in eachrow(s)
        if isnothing(findfirst(findall(s_) .> int_end))
            ind = findall(s_)[findfirst(findall(s_) .>= int_end)]
        else
            ind = findall(s_)[findfirst(findall(s_) .> int_end)]
        end
        push!(nhood, CartesianIndex(k, ind))
        k += 1
    end
    return unique(nhood)
end

function new_bound!(Q_f::Float64, t::Float64, x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, priors::Prior, dat::PEMData, dyn::Dynamics)
    ∇U_bound!(x, v, s, dat, priors, CartesianIndex(1,1), dyn)
    dyn.sampler_eval.bounds += 1
    τ = poisson_time(sum(dyn.a), sum(dyn.b), rand())
    if τ > dyn.t_bound
        # If greater than bounding interval
        # set to bounding interval 
        τ = dyn.t_bound
        dyn.new_t = true
    else
        dyn.new_t = false
    end 
    Q_f = τ + t
end

function sampler_update(dyn::Dynamics, settings::Settings, t::Float64)
    print("Iteration: ");print(dyn.ind);print("/");print(settings.max_ind);print("; At time: ");print(t);print("\n")
end