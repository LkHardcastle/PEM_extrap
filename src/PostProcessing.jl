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
        out_smps[i,:] = sum(smps[1:i,:], dims = 1)
    end
    return out_smps
end