function post_estimates(out, dat, t_smp)
    smps = zeros(length(t_smp), length(dat.s))
    for i in eachindex(dat.s)
        est = hcat(out["t"], vec(out["Sk_x"][:,i,:]), vec(out["Sk_v"][:,i,:]))
        for t in eachindex(t_smp)
            ind = findlast(out["t"] .< t_smp[t])
            t_diff = t_smp[t] - out["t"][ind]
            smps[t, i] = est[ind,2] + est[ind,3]*t_diff
        end
    end
    return smps
end