include("Types.jl")
include("Potential.jl")
include("SplitMerge.jl")

function pem_sample(x0::Matrix{Float64}, s0::Vector{Bool}, v0::Matrix{Float64}, t0::Float64, dat::PEMData, priors::Prior, settings::Settings)
    x, s, v, t = copy(x0), copy(s0), copy(v0), copy(t0)
    Q_f = PriorityQueue{CartesianIndex, Float64}(Base.Order.Forward)
    Q_s = PriorityQueue{CartesianIndex, Float64}(Base.Order.Forward)
    Q_m = PriorityQueue{CartesianIndex, Float64}(Base.Order.Forward)
    dyn = Dynamics()
    start_queue!(Q_f, Q_s, Q_m, x, s, v, dat, dyn)  
    sampler_eval = SamplerEval()
    x_track = zeros(dat.p, size(dat.s,1), settings.max_ind)
    v_track = zeros(dat.p, size(dat.s,1), settings.max_ind)
    s_track = zeros(dat.p, size(dat.s,1), settings.max_ind)
    t_track = zeros(dat.p, size(dat.s,1), settings.max_ind)
    x_track[:,:,1] = copy(x)
    v_track[:,:,1] = copy(v)
    s_track[:,:,1] = copy(s)
    t_track[1] = copy(t) 
    dyn.ind += 1
    for i in 1:settings.max_ind
        # Generate next event time
        sampler_inner(x, s, v, t, dat, priors, settings, Q_f, Q_s, Q_m, dyn)
        # Store next event
        sampler_store(x_track, s_track, v_track, t_track)
        x_track[:,:,dyn.ind] = copy(x)
        v_track[:,:,dyn.ind] = copy(v)
        s_track[:,:,dyn.ind] = copy(s)
        t_track[dyn.ind] = copy(t) 
        dyn.ind += 1
        # Print?
        if dyn.ind % 10_000
            sampler_update(dyn, settings, t)
        end
    end
    # Final things
    return Dict("Sk_x" => x_track, "Sk_v" => v_track, "Sk_s" => s_track, "t" => t_track, "Eval" => sampler_eval)
end

function start_queue!(Q_f::PriorityQueue, Q_s::PriorityQueue, Q_m::PriorityQueue, x::Vector{Float64}, s::Vector{Bool}, v::Vector{Float64}, dat::PEMData,  dyn::Dynamics)
    for j in findall(s)
        # Getting bounding parameters and store final rate eval
        dyn.a[j], dyn.b[j] = ∇U_bound(x, v, s, dat, priors, j, dyn)
        # Generate next event time (from upper bound pre-thinned)
        τ = poisson_time(dyn.a[j], dyn.b[j], rand())
        if τ > dyn.t_bound[j]
            # If greater than bounding interval
            # set to bounding interval 
            τ = dyn.t_bound[j]
            dyn.new_t[j] = true
        else
            dyn.new_t[j] = false
        end
        dyn.t_set[j] = 0.0
        enqueue!(Q_f, j, τ)
        ## Merge queue
        l = findfirst(s[j[1],(j[2] + 1):end])
        if l != j[2]
            if v[j] == v[l,j[2]]
                enqueue!(Q_m, j, Inf)
            elseif (x[j] < x[l,j[2]]) && (v[j] > 0.0)
                enqueue!(Q_m, j, (x[l,j[2]] - x[j])/2)
            elseif (x[j] > x[l,j[2]]) && (v[j] < 0.0)
                enqueue!(Q_m, j, (x[j] - x[l,j[2]])/2)
            else
                enqueue!(Q_m, j, Inf)
            end
        end
    end
    for j in findall(s .== false)
        ## Split queue 
        enqueue!(Q_s, j, rand(Exponential(1/split_rate(priors, j))))
    end
end

function sampler_inner()
    inner_stop = false
    while !inner_stop
        τ = copy(t)
        t, j, type = event_find(Q_f, Q_s, Q_m, τ)
        # Update x
        t_ = t - τ
        x .+= v.*t_
        if type == 1
            if dyn.new_t[j]
                new_bound!(Q_f, t, x, v, s, priors, dat, dyn, j)
            else
                inner_stop = flip_attempt!(x, v, s, dat, j, priors, dyn)
            end
        end
        if type == 2
            split_int!()
            inner_stop = true
        end
        if type == 3
            merge_int!()
            inner_stop = true
        end
    end
end

function event_find(Q_f::PriorityQueue, Q_s::PriorityQueue, Q_m::PriorityQueue, τ::Float64)
    j1, t1 = peek(Q_f)
    j2, t2 = peek(Q_s)
    j3, t3 = peek(Q_m)
    type = findmin([t1,t2,t3])[2]
    t = copy([t1,t2,t3][type])
    j = copy([j1,j2,j3][type])
    if τ > t
        println(Q);println(Q_s);println(Q_m)
        error("Queueing problems")
    end
    return t, j, type
end

function flip_attempt!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, dat::PEMData, j::CartesianIndex, priors::Prior, dyn::Dynamics)
    # True rate
    λ = max(v[j]*∇U(x, s, dat, j, priors), 0.0)
    # Upper bound
    Λ = dyn.a[j] + (t -dyn.t_set[j])*dyn.b[j]
    if Λ > λ + 1e-05
        println(λ);println(Λ);
        error("Incorrect flipping upper bound")
    end
    if λ/Λ > rand()
        v[j] = - v[j]
        nhood = neighbourhood(j, s)
        for l in nhood
            new_bound!(Q_f, t, x, v, s, priors, dat, dyn, l)
            new_split!(Q_s, priors, l)
            new_merge!(Q_m, t, x, v, s, l)
        end
        return true
    else
        new_bound!(Q_f, t, x, v, s, priors, dat, dyn, l)
        return false
    end
end

function split_int!()
    s[j] = true
    new_ind = findfirst(s[j[1],(j[2] + 1):end])
    v[new_ind] = 2*rand(Bernoulli(0.5)) - 1.0
    v[j] = -v[new_ind]
    x[j] = x[j[1], j[2] + new_ind]
    delete!(Q_s, j)
    delete!(Q_m, j)
    delete!(Q_f, j)
    nhood = neighbourhood(j, s)
    # Update interval
    for l in nhood
        new_bound!(Q_f, t, x, v, s, priors, dat, dyn, l)
        new_merge!(Q_m, t, x, v, s, l)
        new_split!(Q_s, priors, l)
    end
end

function merge_int!()
    s[j] = false
    x[j] = Inf
    delete!(Q_s, j)
    delete!(Q_m, j)
    delete!(Q_f, j)
    nhood = neighbourhood(j, s)
    # Update interval
    for l in nhood
        new_bound!(Q_f, t, x, v, s, priors, dat, dyn, l)
        new_merge!(Q_m, t, x, v, s, l)
        new_split!(Q_s, priors, l)
    end
end

function neighbourhood(j::CartesianIndex, s::Matrix{Bool})
    int_start = findlast(s[j[1],begin:(j[2]-1)])
    if isnothing(int_start)
        int_start = 1
    end
    nhood = findall(s[:,int_start:j[2]])
    push!(nhood, findfirst.(eachrow(s[:,(j[2] + 1):end])))
    return nhood
end

function new_bound!(Q_f::PriorityQueue, t::Float64, x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, priors::Prior, dat::PEMData, dyn::Dynamics, j::CartesianIndex)
    delete!(Q_f,j)
    dyn.a[j], dyn.b[j] = ∇U_bound(x, v, s, dat, priors, j, dyn)
    τ = poisson_time(dyn.a[j], dyn.b[j], rand())
    if τ > dyn.t_bound[j]
        # If greater than bounding interval
        # set to bounding interval 
        τ = dyn.t_bound[j]
        dyn.new_t[j] = true
    else
        dyn.new_t[j] = false
    end
    dyn.t_set[j] = t
    enqueue!(Q_f, j, τ)
end

function new_split!(Q_s::PriorityQueue, priors::Prior, j::CartesianIndex)
    delete!(Q_s, j)
    enqueue!(Q_s, j, t + rand(Exponential(1/split_rate(priors, j))))
end

function new_merge!(Q_m::PriorityQueue, t::Float64, x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, j::CartesianIndex)
    delete!(Q_m,j)
    l = findfirst(s[j[1],(j[2] + 1):end])
    if l != j[2]
        if v[j] == v[l,j[2]]
            enqueue!(Q_m, j, Inf)
        elseif (x[j] < x[l,j[2]]) && (v[j] > 0.0)
            enqueue!(Q_m, j, t + (x[l,j[2]] - x[j])/2)
        elseif (x[j] > x[l,j[2]]) && (v[j] < 0.0)
            enqueue!(Q_m, j, t + (x[j] - x[l,j[2]])/2)
        else
            enqueue!(Q_m, j, Inf)
        end
    end
end

function sampler_update(dyn::Dynamics, settings::Settings, t::Float64)
    print("Iteration: ");print(dyn.ind);print("/");print(settings.max_ind);print("; At time: ");print(t);print("\n")
end