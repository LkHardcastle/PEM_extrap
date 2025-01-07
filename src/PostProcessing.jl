function post_estimates(out, dat, t_smp)
    smps = zeros(length(t_smp), length(dat.s))
    for i in eachindex(dat.s)
        est = hcat(out["t"], vec(out["Sk_x"][:,i,:]), vec(out["Sk_v"][:,i,:]))
        ind = 1
        for t in eachindex(t_smp)
            ind = findfirst(out["t"][ind:end] .> t_smp[t]) - 1 + (ind-1)
            t_diff = t_smp[t] - out["t"][ind]
            smps[t, i] = est[ind,2] + est[ind,3]*t_diff
        end
    end
    return smps
end

function post_smps(smps::Array{Float64})
    est = zeros(size(smps[:,:,1],1)*size(smps[:,:,1],2), size(smps,3))
    for i in axes(smps,3)
        est[:,i] = vec(smps[:,:,i])
    end
    return est
end

function transform_smps(smps::Array{Float64})
    out_smps = copy(smps)
    # Faster way of doing this but don't worry for now
    for i in axes(smps, 1)
        out_smps[i,:, :] = cumsum(smps[i, :, :], dims = 1)
    end
    return out_smps
end

function survival_plot(t, breaks, h_vec, break_int)
    S_store = []
    for t_ in t
        past_ind = findfirst(breaks .>= t_) - 1
        if past_ind == 0
            push!(S_store, exp(-h_vec[past_ind + 1]*(t_)))
        else
            S_init = exp(-break_int*sum(h_vec[1:past_ind]))
            push!(S_store, S_init*exp(-h_vec[past_ind + 1]*(t_ - breaks[past_ind])))
        end
    end
    return hcat(t, S_store)
end

function cts_transform(x::Array{Float64}, s_loc::Array{Float64}, grid::Vector{Float64})
    out = zeros(size(x,1), size(grid,1), size(x,3))
    for k in 1:size(x,3)
        for i in 1:length(grid)
            if isnothing(findlast(s_loc[:,k] .< grid[i]))
                ind = 1
            else
                ind = findlast(s_loc[:,k] .< grid[i]) + 1
            end
            if ind == (size(s_loc,1) + 1)
                ind -= 1 
            end
            for j in 1:size(x,1)
                if isinf(x[j, ind, k])
                    #out[j,i,k] = x[j, ind - 1, k]
                    out[j,i,k] = x[j, findlast(isinf.(x[j,:,k]) .== false), k]
                else
                    out[j,i,k] = x[j, ind, k]
                end
            end
        end
    end
    return out
end

function pem_survival(λ::Matrix{Float64}, times::Vector{Float64})
    t_ = times[2:end] .- times[1:(end -1)]
    return cumprod(exp.(.- t_'.*λ'), dims = 2)'
end

function get_DIC(out, dat::PEMData)
    deviance = zeros(size(out["Smp_x"],3))
    for i in 1:size(out["Smp_x"],3)
        J = out["Smp_J"][i]
        θ = cumsum(out["Smp_x"][:,1:J,i],dims = 2)
        s_loc = out["Smp_s_loc"][1:J,i]
        L = size(dat.UQ, 2)
        W = zeros(L,J)
        δ = zeros(L,J)
        d = zeros(Int, length(dat.y))
        for i in eachindex(dat.y)
            if isnothing(findfirst(s_loc .> dat.y[i]))
                d[i] = J
            else
                d[i] = findfirst(s_loc .> dat.y[i])
            end
        end
        for l in 1:L
            yl = dat.y[findall(dat.grp .== l)]
            dl = d[findall(dat.grp .== l)]
            δl = dat.cens[findall(dat.grp .== l)]
            for j in 1:J
                if j == 1
                    sj1 = 0.0
                else
                    sj1 = s_loc[j-1]
                end
                W[l,j] = sum(yl[findall(dl .== j)]) .- length(findall(dl .== j))*sj1 + length(findall(dl .> j))*(s_loc[j] - sj1)
                δ[l,j] = length(intersect(findall(δl .== 1), findall(dl .== j)))
            end
        end
        deviance[i] = 2*sum(exp.(θ).*W .- δ.*θ) 
    end
    DIC = mean(deviance[findall(.!isnan.(deviance))]) + 0.5*var(deviance[findall(.!isnan.(deviance))])
    return deviance, DIC
end

function get_meansurv(out, dat::PEMData)

end