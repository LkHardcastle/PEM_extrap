function ∇U(x::Matrix{Float64}, s::Matrix{Bool}, dat::PEMData, j::CartesianIndex, priors::Prior)
    ∇Uλ = 0.0
    j1 = j[1]
    j2 = j[2]
    int_start = findlast(s[j1,begin:(j2-1)])
    if isnothing(int_start)
        int_start = 1
    end
    sub_ints = vcat(int_start, intersect(findall(sum(s, dims = 1) .> 0.0), (int_start + 1):j2))
    for k in 2:length(sub_ints)
        d = findall(dat.d .∈ (sub_ints[k-1] + 1):sub_ints[k])
        c = (d[end] + 1):dat.n
        ∇Uλ += sum(dat.covar[j1,d].*exp.(sum(dot.(x[:,j2],dat.covar[:,d]),dims = 1)).*(dat.y[d] - dat.s[s[j2-1]])) - sum(dat.cens[d])
        ∇Uλ += sum(dat.covar[j1,c].*exp.(sum(dot.(x[:,j2],dat.covar[:,c]),dims = 1)))*(dat.s[s[j2]] - dat.s[s[j2-1]])
    end
    ∇Uλ += prior_add(x, s, priors, j)
    return ∇Uλ
end

function prior_add(x::Matrix{Float64}, s::Matrix{Bool}, priors::Prior, j::CartesianIndex)
    if isnothing(findlast(s[j[1],1:(j[2]-1)]))
        # First evaluation - draw from initial prior
        return (1/priors.σ0^2)*(x[j] - priors.μ0)
    else
        return (1/priors.σ^2)*(x[j] - x[j[1], findlast(s[j[1],1:(j[2]-1)])])
    end
end

function ∇U_bound(x::Matrix{Float64}, v::Matrix{Float64}, s::Matrix{Bool}, dat::PEMData, priors::Prior, j::CartesianIndex, dyn::Dynamics)
    if v[j] > 0.0
        ΛU1 = max(v[j]*∇U(x, s, dat, j, priors), 0.0)
        ΛU2 = max(v[j]*∇U(x .+ v.*dyn.t_bound[j], s, dat, j, priors), 0.0)
        a = ΛU1 + 0.01 
        b = (ΛU2 - ΛU1)/dyn.t_bound[j]
    elseif v[j] < 0.0
        ΛU1 = max(v[j]*∇U(x, s, dat, j, priors), 0.0)
        ΛU2 = max(v[j]*∇U(x .+ v.*dyn.t_bound[j], s, dat, j, priors), 0.0)
        a = max(ΛU1, ΛU2) + 0.01
        b = 0.0
    end
    return a, b
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

function split_rate(s::Matrix{Bool}, dat::PEMData, priors::Prior, j::CartesianIndex)
    j_prev = findlast(s[j[1],1:j[2]])
    if isnothing(j_prev)
        v1_prop = 100*(sum(dat.cens[findall(dat.y .< dat.s[j[2]])])/max(sum(dat.cens),1) + 0.01*j[2])
    else
        j_prev += 1
        v1_prop = 100*(sum(dat.cens[intersect(findall(dat.y .< dat.s[j[2]]), findall(dat.y .> dat.s[j_prev]))])/max(sum(dat.cens),1) + 0.01*j[2])
    end
    j_next = findfirst(s[j[1],(j[2] + 1):end]) + j[2]
    v2_prop = 100*(sum(dat.cens[intersect(findall(dat.y .< dat.s[j_next]),findall(dat.y .> dat.s[(j[2]+1)]))])/max(sum(dat.cens),1) + 0.01*j_next)
    return 0.5*(priors.ω[j]/(1 - priors.ω[j]))*pdf(Normal(0, priors.σ),0)*(v1_prop + v2_prop)
end
