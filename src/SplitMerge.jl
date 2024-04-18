function split_int!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, Q_f::PriorityQueue, Q_s::PriorityQueue, Q_m::PriorityQueue, dat::PEMData, j::CartesianIndex, priors::Prior, dyn::Dynamics)
    s[j] = true
    delete!(Q_s, j)
    v[j] = 2*rand(Bernoulli(0.5)) - 1.0
    x[j] = 0.0
    new_bound!(Q_f, t, x, v, s, priors, dat, dyn, j, true)
    new_merge!(Q_m, t, x, v, s, j, true)
end

function merge_int!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, Q_f::PriorityQueue, Q_s::PriorityQueue, Q_m::PriorityQueue, dat::PEMData, j::CartesianIndex, priors::Prior, dyn::Dynamics)
    s[j] = false
    v[j] = 0.0
    new_split!(Q_s, s, t, priors, j)
    delete!(Q_m, j)
    delete!(Q_f, j)
end

function new_split!(Q_s::PriorityQueue, s::Matrix{Bool}, t::Float64, priors::Prior, j::CartesianIndex)
    enqueue!(Q_s, j, t + rand(Exponential(1/split_rate(s, dat, priors, j))))
end

function new_merge!(Q_m::PriorityQueue, t::Float64, x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, j::CartesianIndex, new::Bool)
    if j[2] > 1
        if !new
            delete!(Q_m,j)
        end
        enqueue!(Q_m, j, t + merge_time(x, v, j))
    end
end

function merge_time(x::Matrix{Float64}, v::Matrix{Float64}, j::CartesianIndex)
    if (x[j] > 0.0 && v[j] < 0.0) || (x[j] < 0.0 && v[j] > 0.0)
        return abs(x[j])/abs(v[j])
    else
        return Inf
    end
end

function split_rate(s::Matrix{Bool}, dat::PEMData, priors::Prior, j::CartesianIndex)
    #'return 0.375*abs(sum(dot([1,-2],[1,-1])))*priors.p_split*(priors.ω[j]/(1 - priors.ω[j]))*(sqrt(2*pi*priors.σ^2))^-1
    return priors.p_split*(priors.ω[j]/(1 - priors.ω[j]))*(sqrt(2*pi*priors.σ^2))^-1
end

function v_size(s::Matrix{Bool}, j::CartesianIndex)
    return sum(s[j[1],1:(j[2])])
end