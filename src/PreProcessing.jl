function init_params(breaks::Vector{Float64}, p::Int64)
    x0 = zeros(p, length(breaks))
    v0 = rand(p, length(breaks))
    s0 = fill(false, p, length(breaks))
    for j in 1:p
        ind = sort(vcat(unique(rand( DiscreteUniform(1, length(breaks)), max(trunc(Int, 0.1*length(breaks)),5) )), length(breaks)))
        s0[j,ind] .= true
        x_int = rand(Normal(0.0,1), length(ind))
        v_int = 2 .*rand(Bernoulli(0.5), length(ind)) .- 1.0
        k = 1
        for i in axes(x0,2)
            x0[j,i] = x_int[k]
            v0[j,i] = v_int[k]
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