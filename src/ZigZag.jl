function flip_attempt!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, dat::PEMData, times::Times, dyn::ZigZag, priors::Prior)
    λ = max.(0.0, v.*∇U(x, s, dat, CartesianIndex(1,1)))
    Σλ = sum(λ)
    Λ = dyn.a + (t- dyn.t_set)*dyn.b
    out = false
    if  Σλ/Λ > 1 + 1e-5
        verbose_talk(x, v, s, t, dyn)
        error("Bad flip bound")
    end
    if Σλ/Λ > rand()
        dyn.sampler_eval.flips += 1
        j = rand(Categorical(λ./Σλ))
        v[j] = -v[j]
        new_merge!(times, t, x, v, j, false)
        dyn.last_type = "Flip"
        out = true
    end
    next_event!(t, times, dyn)
    ∇U_bound!(x, v, s, dat, priors, CartesianIndex(1,1), dyn)
    return out
end

function split_int!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, j::CartesianIndex, dyn::ZigZag)
    s[j] = true
    delete!(times.Q_s, j)
    v[j] = (2*rand(Bernoulli(0.5)) - 1.0)*dyn.v_abs[j]
    x[j] = 0.0
    enqueue!(times.Q_m, j, Inf)
end

function merge_int!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, j::CartesianIndex, priors::Prior, dyn::ZigZag)
    s[j] = false
    v[j] = 0.0
    x[j] = 0.0
    new_split!(times, s, t, priors, j, dyn)
    delete!(times.Q_m, j)
end

function new_split!(times::Times, s::Matrix{Bool}, t::Float64, priors::Prior, j::CartesianIndex, dyn::ZigZag)
    enqueue!(times.Q_s, j, t + rand(Exponential(1/split_rate(s, dat, priors, j, dyn))))
end

function new_merge!(times::Times, t::Float64, x::Matrix{Float64}, v::Matrix{Float64}, j::CartesianIndex, new::Bool)
    if j[2] > 1
        if !new
            delete!(times.Q_m,j)
        end
        enqueue!(times.Q_m, j, t + merge_time(x, v, j))
    end
end

function split_rate(s::Matrix{Bool}, dat::PEMData, priors::Union{FixedPrior,HyperPrior2}, j::CartesianIndex, dyn::ZigZag)
    #'return 0.375*abs(sum(dot([1,-2],[1,-1])))*priors.p_split*(priors.ω[j]/(1 - priors.ω[j]))*(sqrt(2*pi*priors.σ^2))^-1
    return abs(dyn.v_abs[j])*priors.p_split*(priors.ω[j]/(1 - priors.ω[j]))*(sqrt(2*pi*priors.σ^2))^-1
end