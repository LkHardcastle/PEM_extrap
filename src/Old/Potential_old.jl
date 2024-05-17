function ∇U(x::Matrix{Float64}, s::Matrix{Bool}, dat::PEMData, j::CartesianIndex)
    ∇Uλ = 0.0
    Uλ = 0.0
    ∇U_out = []
    k = CartesianIndex(j[1], size(s,2))
    while k[2] >= j[2] 
        if s[k]
            int_start = findlast(s[k[1],begin:(k[2] - 1)])
            if isnothing(int_start)
                int_start = 0
            end
            #println(int_start);println(j);println(dat.d)
            d = findall(dat.d .∈ (int_start+1):k[2])
            c = findall(dat.d .> k[2])
            if k[2] > 1
                sj_1 = dat.s[int_start]
            else
                sj_1 = 0.0
            end
            ∇Uλ += exp(sum(x[1:k[2]]))*(sum(dat.y[d] .- sj_1) + length(c)*(dat.s[k[2]] - sj_1)) - sum(dat.cens[d])
            pushfirst!(∇U_out, ∇Uλ)
        end
        k = k - CartesianIndex(0,1)
    end
    return ∇U_out
end

function ∇U_p(x::Matrix{Float64}, s::Matrix{Bool}, j::CartesianIndex, priors::Prior)
    ∇Uλ = prior_add(x, s, priors, j)
    return ∇Uλ
end
function prior_add(x::Matrix{Float64}, s::Matrix{Bool}, priors::Union{FixedPrior,HyperPrior2}, j::CartesianIndex)
    last_ind = findlast(s[j[1],1:(j[2]-1)])
    if isnothing(last_ind)
        # First evaluation - draw from initial prior
        return (1/priors.σ0^2)*(x[j] - priors.μ0)
    else
        return (1/priors.σ^2)*x[j]
    end
end

function prior_add(x::Matrix{Float64}, s::Matrix{Bool}, priors::Union{HyperPrior3}, j::CartesianIndex)
    last_ind = findlast(s[j[1],1:(j[2]-1)])
    if isnothing(last_ind)
        # First evaluation - draw from initial prior
        return (1/priors.σ0^2)*(x[j] - priors.μ0)
    else
        return (1/priors.σ[j]^2)*x[j]
    end
end

function ∇U_bound!(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, dat::PEMData, priors::Prior, j::CartesianIndex, dyn::Dynamics)
    ∇U1 = ∇U(x, s, dat, CartesianIndex(1,1))
    ∇U2 = ∇U(x .+ v.*dyn.t_bound, s, dat, CartesianIndex(1,1))
    m = 1
    for k ∈ CartesianIndex(1,1):CartesianIndex(1,size(s,2))
        if s[k]
            a_, b_ = 0.0, 0.0
            if v[k] > 0.0
                ΛU1 = max(v[k]*∇U1[m], 0.0)
                ΛU2 = max(v[k]*∇U2[m], 0.0)
                a = ΛU1 #+ 0.01 
                b = (ΛU2 - ΛU1)/dyn.t_bound[k]
            elseif v[j] < 0.0
                ΛU1 = max(v[k]*∇U1[m], 0.0)
                ΛU2 = max(v[k]*∇U2[m], 0.0)
                a = max(ΛU1, ΛU2) #+ 0.01
                b = 0.0
            end
            ΛU1p = max(v[k]*∇U_p(x, s, k, priors), 0.0)
            ΛU2p = max(v[k]*∇U_p(x .+ v.*dyn.t_bound, s, k, priors), 0.0)
            a_ += ΛU1p #+ 0.01 
            b_ += (ΛU2p - ΛU1p)/dyn.t_bound
            dyn.a[k] = a_
            dyn.b[k] = b_
            m += 1
        end
    end
end

function poisson_time(a, b, u)
    ######## From ZigZagBoomerang.jl
    if b > 0
        if a < 0
            return sqrt(-log(u)*2.0/b) - a/b
        else # a[i]>0
            return sqrt((a/b)^2 - log(u)*2.0/b) - a/b
        end
    elseif b == 0
        if a > 0
            return -log(u)/a
        else # a[i] <= 0
            return Inf
        end
    else # b[i] < 0
        if a <= 0
            return Inf
        elseif -log(u) <= -a^2/b + a^2/(2*b)
            return -sqrt((a/b)^2 - log(u)*2.0/b) - a/b
        else
            return Inf
        end
    end
end


