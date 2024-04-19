function init_params(p::Int64, dat::PEMData)
    x0 = zeros(p, length(dat.s))
    v0 = rand(p, length(dat.s))
    s0 = fill(false, p, length(dat.s))
    for j in 1:p
        ind = sort(unique(vcat(unique(rand(DiscreteUniform(1, length(dat.s)), max(trunc(Int, 0.1*length(dat.s)),5) )), length(dat.s))))
        s0[j,ind] .= true
        x_int = rand(Normal(0.0,1), length(ind))
        v_int = 2 .*rand(Bernoulli(0.5), length(ind)) .- 1.0
        k = 1
        for i in axes(x0,2)
            x0[j,i] = x_int[k]
            if k == 1
                v0[j,i] = v_int[k]
            else
                v0[j,i] = v_int[k]
            end
            if s0[j,i] 
                k += 1
            end
        end
    end
    return x0, v0, s0
end

function init_data(y, cens, covar, breaks)
    ind = sortperm(y)
    y = y[ind]
    cens = cens[ind]
    covar = covar[:,ind]
    d = zeros(Int, length(y))
    for i in eachindex(y)
        d[i] = findfirst(breaks .> y[i])
    end
    return PEMData(y, cens, covar, size(covar,1), size(covar, 2), breaks, d)
end