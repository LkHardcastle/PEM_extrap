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