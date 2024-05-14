function init_params(p::Int64, dat::PEMData, v_abs::Vector{Float64})
    x0 = rand(Normal(0.0,0.1), p, length(dat.s))
    v0 = v_abs'.*(2 .*(rand(Bernoulli(0.5),p, length(dat.s))) .- 1.0)
    s0 = fill(true, p, length(dat.s))
    return x0, v0, s0
end

function init_params(p::Int64, dat::PEMData)
    x0 = rand(Normal(0.0,0.1), p, length(dat.s))
    v0 = rand(Normal(0,1),size(x0))
    s0 = fill(true, p, length(dat.s))
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