function split_int!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, Q_f::PriorityQueue, Q_s::PriorityQueue, Q_m::PriorityQueue, dat::PEMData, j::CartesianIndex, priors::Prior, dyn::Dynamics)
    s[j] = true
    delete!(Q_s, j)
    new_ind = CartesianIndex(j[1],findfirst(s[j[1],(j[2] + 1):end]) + j[2])
    l = findlast(s[j[1],begin:(j[2]-1)])
    if isnothing(l)
        l = 1
    else
        l += 1
    end
    prev_ind = CartesianIndex(j[1],l)
    if rand() < 0.0
        v_sign = sign(v[j])
        v[prev_ind:new_ind] .= v_sign*v_size(s,j)
    else
        v_sign = 2*rand(Bernoulli(0.5)) - 1.0
        v[(j + CartesianIndex(0,1)):new_ind] .= v_sign*v_size(s,new_ind)
        if isnothing(findlast(s[j[1],begin:(j[2]-1)]))
            v[prev_ind:j] .= -v_sign*v_size(s,j)
        else
            v[prev_ind:j] .= -v_sign*v_size(s,j)
        end
    end
    x[prev_ind:new_ind] .= x[new_ind]
    nhood = neighbourhood(j, s)
    for l in nhood
        if l == j
            new_bound!(Q_f, t, x, v, s, priors, dat, dyn, l, true)
        else
            new_bound!(Q_f, t, x, v, s, priors, dat, dyn, l, false)
        end
        if l[1] == j[1]
            if l == j
                new_merge!(Q_m, t, x, v, s, l, true)
            else
                new_merge!(Q_m, t, x, v, s, l, false)
            end
        end
    end
end

function merge_int!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, t::Float64, Q_f::PriorityQueue, Q_s::PriorityQueue, Q_m::PriorityQueue, dat::PEMData, j::CartesianIndex, priors::Prior, dyn::Dynamics)
    s[j] = false
    new_ind = CartesianIndex(j[1],findfirst(s[j[1],(j[2] + 1):end]) + j[2])
    l = findlast(s[j[1],begin:(j[2]-1)])
    if isnothing(l)
        l = 1
        prev_ind = CartesianIndex(j[1],l)
        if sign(v[j]) == sign(v[new_ind])
            v[prev_ind:new_ind] .= sign(v[j])*v_size(s,new_ind)
        else
            v[prev_ind:new_ind] .= (2*rand(Bernoulli(0.5)) - 1.0)*v_size(s,new_ind)
        end
    else
        l += 1
        prev_ind = CartesianIndex(j[1],l)
        if sign(v[j]) == sign(v[new_ind])
            v[prev_ind:new_ind] .= sign(v[j])*v_size(s,new_ind)
        else
            v[prev_ind:new_ind] .= (2*rand(Bernoulli(0.5)) - 1.0)*v_size(s,new_ind)
        end
    end
    x[prev_ind:new_ind] .= x[new_ind]
    new_split!(Q_s, s, t, priors, j)
    delete!(Q_m, j)
    delete!(Q_f, j)
    nhood = neighbourhood(j, s)
    for l in nhood
        new_bound!(Q_f, t, x, v, s, priors, dat, dyn, l, false)
        if l[1] == j[1]
            new_merge!(Q_m, t, x, v, s, l, false)
        end
    end
end

function new_split!(Q_s::PriorityQueue, s::Matrix{Bool}, t::Float64, priors::Prior, j::CartesianIndex)
    enqueue!(Q_s, j, t + rand(Exponential(1/split_rate(s, dat, priors, j))))
end

function new_merge!(Q_m::PriorityQueue, t::Float64, x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, j::CartesianIndex, new::Bool)
    if !new
        delete!(Q_m,j)
    end
    l = findfirst(s[j[1],(j[2] + 1):end])
    if !isnothing(l)
        l += j[2]
        if l != j[2]
            enqueue!(Q_m, j, t + merge_time(x, v, j, l))
        end
    else
        enqueue!(Q_m, j, Inf)
    end
end

function merge_time(x::Matrix{Float64}, v::Matrix{Float64}, j::CartesianIndex, l::Int64)
    if sign(v[j]) == sign(v[j[1],l])
        #if abs(v[j]) != abs(v[j[1],l])
        #    if x[j] < x[j[1],l]
        #        if (abs(v[j]) > abs(v[j[1],l])) && (sign(v[j]) > 0.0)
        #            return abs(x[j[1],l] - x[j])/(abs(v[j] - v[j[1],l]))
        #        elseif (abs(v[j]) < abs(v[j[1],l])) && (sign(v[j]) < 0.0)
        #            return abs(x[j[1],l] - x[j])/(abs(v[j] - v[j[1],l]))
        #        else
        #            return Inf
        #        end
        #    elseif x[j] > x[j[1],l]
        #        if (abs(v[j]) < abs(v[j[1],l])) && (sign(v[j]) > 0.0)
        #            return abs(x[j[1],l] - x[j])/(abs(v[j] - v[j[1],l]))
        #        elseif (abs(v[j]) > abs(v[j[1],l])) && (sign(v[j]) < 0.0)
        #            return abs(x[j[1],l] - x[j])/(abs(v[j] - v[j[1],l]))
        #        else
        #            return Inf
        #        end
        #    else
        #        return Inf
        #    end
        #else
            return Inf
        #end
    else
        if (x[j] < x[j[1],l]) && (v[j] > 0.0)
            return (x[j[1],l] - x[j])/(abs(v[j]) + abs(v[j[1],l]))
        elseif (x[j] > x[j[1],l]) && (v[j] < 0.0)
            return (x[j] - x[j[1],l])/(abs(v[j]) + abs(v[j[1],l]))
        else
            return Inf
        end
    end
end

function split_rate(s::Matrix{Bool}, dat::PEMData, priors::Prior, j::CartesianIndex)
    #'return 0.375*abs(sum(dot([1,-2],[1,-1])))*priors.p_split*(priors.ω[j]/(1 - priors.ω[j]))*(sqrt(2*pi*priors.σ^2))^-1
    return priors.p_split*(priors.ω[j]/(1 - priors.ω[j]))*(sqrt(2*pi*priors.σ^2))^-1
end

function v_size(s::Matrix{Bool}, j::CartesianIndex)
    return sum(s[j[1],1:(j[2])])
end