using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random, Optim, Roots, SpecialFunctions, Statistics
using Plots, CSV, DataFrames, RCall, Interpolations, MCMCDiagnosticTools

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))

R"""
library(ggplot2)
library(dplyr)
library(tidyr)
library(cowplot)
library(coda)
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
"""


Random.seed!(12515)
n = 0
y = rand(Exponential(1.0),n)
breaks = collect(0.1:0.1:3.0)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 10_000
nsmp = 1_000
test_times = collect(0.05:0.05:2.95)
sampler = []
J_vec = []
h_mat = Matrix{Float64}(undef, 3, 0)
it = []
tuning = []
exp_its = 5
h_test = [0.5, 1.5, 2.5]
h_ess = Vector{Vector{Float64}}()


Random.seed!(12515)
nits = 10_000
priors1 = BasicPrior(1.0, FixedV(1.0), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.1), [RandomWalk()], [], 2)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
nits = 20_000
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
@time out1 = pem_fit(state0, dat, priors1, settings, test_times, 1_000)

Random.seed!(12515)
nits = 200_000
priors3 = BasicPrior(1.0, FixedV(1.0), FixedW([0.5]), 0.0, RJ(10.0, 1, 0.1, 100.0, 3.1), [RandomWalk()], [], 2)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 50.0)
@time out2 = pem_fit(state0, dat, priors3, settings, test_times, 1_000)

Random.seed!(12515)
nits = 200_000
priors3 = BasicPrior(1.0, FixedV(1.0), FixedW([0.5]), 0.0, RJ(10.0, 1, 0.1, 100.0, 3.1), [RandomWalk()], [], 2)
breaks = [0.1, 1.0, 2.0, 3.0]
p = 1
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 50.0)
@time out3 = pem_fit(state0, dat, priors3, settings, test_times, 1_000)

histogram(out1[1]["Sk_θ"][1,1,:], alpha = 0.1)
histogram!(out2[1]["Sk_θ"][1,1,:], alpha = 0.1)
histogram!(out3[1]["Sk_θ"][1,1,:], alpha = 0.1)

quantile(out1[1]["Sk_θ"][1,3,findall(out1[1]["Sk_θ"][1,3,:] .!= 0.0)], [0.025,0.975])
quantile(out2[1]["Sk_θ"][1,1,:], [0.025,0.975])
quantile(out3[1]["Sk_θ"][1,1,:], [0.025,0.975])

plot(out2[1]["Sk_θ"][1,2,:])
plot!(out2[2]["Sk_θ"][1,2,:])

mean(out2[2]["Sk_J"]*2)
mean(out1[1]["Sk_J"])

nits = 10_000
for i in 1:exp_its
    priors1 = BasicPrior(1.0, FixedV(0.5), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.1), [RandomWalk()], [], 2)
    x0, v0, s0 = init_params(p, dat)
    v0 = v0./norm(v0)
    t0 = 0.0
    state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
    settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
    out = pem_fit(state0, dat, priors1, settings, test_times, 1_000)
    push!(J_vec,mean(sum(out[1]["Sk_s"],dims = 2)[1,1,:]))
    h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[1]["Sk_θ"], dims = 2), out[1]["Sk_s_loc"], h_test), dims = 3)[1,:,1])
    push!(J_vec,mean(sum(out[2]["Sk_s"],dims = 2)[1,1,:]))
    h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[1]["Sk_θ"], dims = 2), out[2]["Sk_s_loc"], h_test), dims = 3)[1,:,1])
    push!(it, 2*i - 1)
    push!(it, 2*i)
    push!(sampler,"PDMP")
    push!(tuning, 0.0)
    push!(sampler,"PDMP")
    push!(tuning, 0.0)
    push!(h_ess, out[4])
end

println("-----------")
nits = 100_000
tuning_param = [0.01,0.1,0.2,0.5,1.0]
#tuning_param = [0.1]
for σ in tuning_param
    for i in 1:exp_its
        priors2 = BasicPrior(1.0, FixedV(0.5), FixedW([0.5]), 0.0, RJ(10.0, 1, σ, 100.0, 3.1), [RandomWalk()], [], 2)
        x0, v0, s0 = init_params(p, dat)
        v0 = v0./norm(v0)
        t0 = 0.0
        state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
        settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 5.0)
        out = pem_fit(state0, dat, priors2, settings, test_times, 10_000)
        push!(J_vec,mean(sum(out[1]["Sk_s"][:,:,1:10:end] ,dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[1]["Sk_θ"][:,:,1:10:end], dims = 2), out[1]["Sk_s_loc"][:,1:10:end], h_test), dims = 3)[1,:,1])
        push!(J_vec,mean(sum(out[2]["Sk_s"][:,:,1:10:end],dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[2]["Sk_θ"][:,:,1:10:end], dims = 2), out[2]["Sk_s_loc"][:,1:10:end], h_test), dims = 3)[1,:,1])
        push!(it, 2*i - 1)
        push!(it, 2*i)
        push!(sampler,"PDMPRJ")
        push!(tuning, σ)
        push!(sampler,"PDMPRJ")
        push!(tuning, σ)
        push!(h_ess, out[4])
    end
end
println("-----------")
nits = 100_000
for σ in tuning_param
    for i in 1:exp_its
        priors3 = BasicPrior(1.0, FixedV(0.5), FixedW([0.5]), 0.0, RJ(10.0, 1, σ, 100.0, 3.1), [RandomWalk()], [], 2)
        x0, v0, s0 = init_params(p, dat)
        v0 = v0./norm(v0)
        t0 = 0.0
        state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
        settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 50.0)
        out = pem_fit(state0, dat, priors3, settings, test_times, 10_000)
        push!(J_vec,mean(sum(out[1]["Sk_s"][:,:,1:10:end] ,dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[1]["Sk_θ"][:,:,1:10:end], dims = 2), out[1]["Sk_s_loc"][:,1:10:end], h_test), dims = 3)[1,:,1])
        push!(J_vec,mean(sum(out[2]["Sk_s"][:,:,1:10:end],dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[2]["Sk_θ"][:,:,1:10:end], dims = 2), out[2]["Sk_s_loc"][:,1:10:end], h_test), dims = 3)[1,:,1])
        push!(it, 2*i - 1)
        push!(it, 2*i)
        push!(sampler,"MHRJ")
        push!(tuning, σ)
        push!(sampler,"MHRJ")
        push!(tuning, σ)
        push!(h_ess, out[4])
    end
end

df = DataFrame(Sampler = sampler, Iter = it, Tuning = tuning, J = J_vec,  h1 = h_mat[1,:], h2 = h_mat[2,:], h3 = h_mat[3,:])
ess1 = copy(h_ess)
CSV.write(datadir("RJExp1.csv"), df)


Random.seed!(12515)
n = 200
breaks = collect(0.1:0.1:3.1)
p = 1
y = zeros(n)
cens = zeros(n)
y1 = rand(Exponential(2.0),n)
y2 = rand(Exponential(1.0),n)
for i in 1:n
    if y1[i] < 1.0
        y[i] = y1[i]
        cens[i] = 1.0
    elseif y2[i] < 2.0
        y[i] = y2[i] + 1.0
        cens[i] = 1.0
    else
        y[i] = 3.0 
        cens[i] = 0.0
    end
end
p = 1
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nsmp = 1_000
test_times = [0.5, 1.5, 2.5]
sampler = []
J_vec = []
h_mat = Matrix{Float64}(undef, 3, 0)
it = []
tuning = []
exp_its = 5
h_test = [0.5, 1.5, 2.5]
h_ess = Vector{Vector{Float64}}()

Random.seed!(12515)
priors1 = BasicPrior(1.0, FixedV(1.0), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.1), [RandomWalk()], [], 2)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
nits = 100_000
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 5.0, false, true, 0.01, 50.0)
@time out1 = pem_fit(state0, dat, priors1, settings, test_times, 1_000)

Random.seed!(12515)
nits = 100_000
priors3 = BasicPrior(1.0, FixedV(1.0), FixedW([0.5]), 0.0, RJ(10.0, 1, 0.5, 100.0, 3.1), [RandomWalk()], [], 2)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 50.0)
@time out2 = pem_fit(state0, dat, priors3, settings, test_times, 10_000)

Random.seed!(12515)
nits = 500_000
priors3 = BasicPrior(1.0, FixedV(1.0), FixedW([0.5]), 0.0, RJ(10.0, 1, 0.01, 100.0, 3.1), [RandomWalk()], [], 2)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 50.0)
@time out3 = pem_fit(state0, dat, priors3, settings, test_times, 10_000)


c1 = cts_transform(cumsum(out1[1]["Sk_θ"], dims = 2), out1[1]["Sk_s_loc"], h_test)
c2 = cts_transform(cumsum(out1[2]["Sk_θ"], dims = 2), out1[2]["Sk_s_loc"], h_test)
h1 = cts_transform(cumsum(out1[2]["Sk_θ"], dims = 2), out1[2]["Sk_s_loc"], collect(0.01:0.01:3.0))
h2 = cts_transform(cumsum(out1[1]["Sk_θ"], dims = 2), out1[1]["Sk_s_loc"], collect(0.01:0.01:3.0))
c1_ = cts_transform(cumsum(out2[1]["Sk_θ"], dims = 2), out2[1]["Sk_s_loc"], h_test)
c2_ = cts_transform(cumsum(out2[2]["Sk_θ"], dims = 2), out2[2]["Sk_s_loc"], h_test)
h1_ = cts_transform(cumsum(out2[1]["Sk_θ"][:,:,1:5:100_000], dims = 2), out2[1]["Sk_s_loc"][:,1:5:100_000], collect(0.01:0.01:3.0))
h2_ = cts_transform(cumsum(out2[2]["Sk_θ"][:,:,1:5:100_000], dims = 2), out2[2]["Sk_s_loc"][:,1:5:100_000], collect(0.01:0.01:3.0))
c11 = cts_transform(cumsum(out3[1]["Sk_θ"], dims = 2), out3[1]["Sk_s_loc"], h_test)
c21 = cts_transform(cumsum(out3[2]["Sk_θ"], dims = 2), out3[2]["Sk_s_loc"], h_test)
h11 = cts_transform(cumsum(out3[2]["Sk_θ"][:,:,1:25:500_000], dims = 2), out3[2]["Sk_s_loc"][:,1:25:500_000], collect(0.01:0.01:3.0))
h21 = cts_transform(cumsum(out3[1]["Sk_θ"][:,:,1:25:500_000], dims = 2), out3[1]["Sk_s_loc"][:,1:25:500_000], collect(0.01:0.01:3.0))


mean(vec(sum(out1[1]["Sk_s"], dims = 2)))
mean(vec(sum(out2[1]["Sk_s"], dims = 2)))
mean(vec(sum(out3[1]["Sk_s"], dims = 2)))
mean(out1[1]["Sk_J"])
mean(out2[1]["Sk_J"])
mean(out3[1]["Sk_J"])
histogram(vec(out1[1]["Sk_s_loc"]), normalize = true)
histogram(vec(out3[2]["Sk_s_loc"]), normalize = true)
histogram!(vec(out3[1]["Sk_s_loc"]), normalize = true)

plot(vec(sum(out1[1]["Sk_s"], dims = 2)))
plot!(out2[1]["Sk_J"])
plot!(out3[1]["Sk_J"])
median(c1[:,:,:], dims = 3)
median(c2[:,:,:], dims = 3)
median(c1_[:,:,:], dims = 3)
median(c2_[:,:,:], dims = 3)
median(c11[:,:,:], dims = 3)
median(c21[:,:,:], dims = 3)
plot(h1[1,98,:])
plot!(h1_[1,98,:])
plot!(h11[1,98,:])
quantile(h2[1,50,:], [0.025,0.975])
quantile(h2_[1,50,:], [0.025,0.975])
quantile(h21[1,50,:], [0.025,0.975])

findall(findfirst.(x -> x == 2, eachcol(cumsum(out1[2]["Sk_s"][1,:,:],dims = 1))) .== nothing)
findfirst.(x -> x == 2, eachcol(cumsum(out1[2]["Sk_s"][1,:,:],dims = 1)))
plot(out1[1]["Sk_s_loc"][1,:])
plot(out2[1]["Sk_s_loc"][10,:])
plot(out3[1]["Sk_s_loc"][10,:])


plot(collect(0.01:0.01:3.0), vec(median(h1[:,:,10_000:end], dims = 3)))
plot!(collect(0.01:0.01:3.0), vec(median(h2[:,:,10_000:end], dims = 3)))
plot!(collect(0.01:0.01:3.0), vec(median(h1_[:,:,10_000:end], dims = 3)))
plot!(collect(0.01:0.01:3.0), vec(median(h2_[:,:,10_000:end], dims = 3)))
plot!(collect(0.01:0.01:3.0), vec(median(h11[:,:,10_000:end], dims = 3)))
plot!(collect(0.01:0.01:3.0), vec(median(h21[:,:,10_000:end], dims = 3)))

plot(exp.(h1[1,98,10_000:end]))
plot!(exp.(h1_[1,98,10_000:end]))
plot!(exp.(h11[1,98,10_000:end]))

histogram(h1[1,200,10_000:end], alpha = 0.1, normalize = true)
histogram!(h1_[1,200,10_000:end], alpha = 0.1, normalize = true)
histogram!(h11[1,200,10_000:end], alpha = 0.1, normalize = true)

plot(h1[1,98,:], h1[1,99,:])
plot!(h1_[1,98,:], h1_[1,99,:])

i = 99
histogram(h1[1,i,findall(h1[1,i,:] .- h1[1,i-1,:] .!= 0.0)] .- h1[1,i-1,findall(h1[1,i,:] .- h1[1,i-1,:] .!= 0.0)], alpha = 0.2, normalize = true)
histogram!(h11[1,i,findall(h11[1,i,:] .- h11[1,i-1,:] .!= 0.0)] .- h11[1,i-1,findall(h11[1,i,:] .- h11[1,i-1,:] .!= 0.0)], alpha = 0.2, normalize = true)

plot(scatter(exp.(h1[1,98,10_000:end]), exp.(h1[1,99,10_000:end]), alpha = 0.1))
scatter!(exp.(h11[1,98,10_000:end]), exp.(h11[1,99,10_000:end]), alpha = 0.1)

plot(scatter(exp.(h1[1,98,:]), exp.(h1[1,99,:]), alpha = 0.1))
scatter!(exp.(h11[1,98,:]), exp.(h11[1,99,:]), alpha = 0.1)

plot(collect(0.01:0.01:3.0), exp.(vec(median(h1[:,:,:], dims = 3))))
plot!(collect(0.01:0.01:3.0), exp.(vec(median(h2[:,:,:], dims = 3))))
plot!(collect(0.01:0.01:3.0), exp.(vec(median(h1_[:,:,:], dims = 3))))
plot!(collect(0.01:0.01:3.0), exp.(vec(median(h2_[:,:,:], dims = 3))))
plot!(collect(0.01:0.01:3.0), exp.(vec(median(h11[:,:,:], dims = 3))))
plot!(collect(0.01:0.01:3.0), exp.(vec(median(h21[:,:,:], dims = 3))))

histogram(h1[1,150,:], alpha = 0.1)
histogram!(h1_[1,150,:], alpha = 0.1)
histogram(h1[1,98,findall(h1[1,98,:] .< -0.5)], alpha = 0.1)
histogram!(h11[1,98,findall(h11[1,98,:] .< -0.5)], alpha = 0.1)


nits = 10_000
for i in 1:exp_its
    priors1 = BasicPrior(1.0, FixedV(0.5), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.1), [RandomWalk()], [], 2)
    x0, v0, s0 = init_params(p, dat)
    v0 = v0./norm(v0)
    t0 = 0.0
    state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
    settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
    out = pem_fit(state0, dat, priors1, settings, test_times, 1_000)
    push!(J_vec,mean(sum(out[1]["Sk_s"],dims = 2)[1,1,:]))
    h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[1]["Sk_θ"], dims = 2), out[1]["Sk_s_loc"], h_test), dims = 3)[1,:,1])
    push!(J_vec,mean(sum(out[2]["Sk_s"],dims = 2)[1,1,:]))
    h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[2]["Sk_θ"], dims = 2), out[2]["Sk_s_loc"], h_test), dims = 3)[1,:,1])
    push!(it, 2*i - 1)
    push!(it, 2*i)
    push!(sampler,"PDMP")
    push!(tuning, 0.0)
    push!(sampler,"PDMP")
    push!(tuning, 0.0)
    push!(h_ess, out[4])
end

println("-----------")
nits = 100_000
tuning_param = [0.01,0.1,0.2,0.5,1.0]
#tuning_param = [0.1]
for σ in tuning_param
    for i in 1:exp_its
        priors2 = BasicPrior(1.0, FixedV(0.5), FixedW([0.5]), 0.0, RJ(10.0, 1, σ, 100.0, 3.1), [RandomWalk()], [], 2)
        x0, v0, s0 = init_params(p, dat)
        v0 = v0./norm(v0)
        t0 = 0.0
        state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
        settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 5.0)
        out = pem_fit(state0, dat, priors2, settings, test_times, 10_000)
        push!(J_vec,mean(sum(out[1]["Sk_s"][:,:,1:10:end] ,dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[1]["Sk_θ"][:,:,1:10:end], dims = 2), out[1]["Sk_s_loc"][:,1:10:end], h_test), dims = 3)[1,:,1])
        push!(J_vec,mean(sum(out[2]["Sk_s"][:,:,1:10:end],dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[2]["Sk_θ"][:,:,1:10:end], dims = 2), out[2]["Sk_s_loc"][:,1:10:end], h_test), dims = 3)[1,:,1])
        push!(it, 2*i - 1)
        push!(it, 2*i)
        push!(sampler,"PDMPRJ")
        push!(tuning, σ)
        push!(sampler,"PDMPRJ")
        push!(tuning, σ)
        push!(h_ess, out[4])
    end
end
println("-----------")
nits = 100_000
for σ in tuning_param
    for i in 1:exp_its
        priors3 = BasicPrior(1.0, FixedV(0.5), FixedW([0.5]), 0.0, RJ(10.0, 1, σ, 100.0, 3.1), [RandomWalk()], [], 2)
        x0, v0, s0 = init_params(p, dat)
        v0 = v0./norm(v0)
        t0 = 0.0
        state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
        settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 50.0)
        out = pem_fit(state0, dat, priors3, settings, test_times, 10_000)
        push!(J_vec,mean(sum(out[1]["Sk_s"][:,:,1:10:end] ,dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[1]["Sk_θ"][:,:,1:10:end], dims = 2), out[1]["Sk_s_loc"][:,1:10:end], h_test), dims = 3)[1,:,1])
        push!(J_vec,mean(sum(out[2]["Sk_s"][:,:,1:10:end],dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[2]["Sk_θ"][:,:,1:10:end], dims = 2), out[2]["Sk_s_loc"][:,1:10:end], h_test), dims = 3)[1,:,1])
        push!(it, 2*i - 1)
        push!(it, 2*i)
        push!(sampler,"MHRJ")
        push!(tuning, σ)
        push!(sampler,"MHRJ")
        push!(tuning, σ)
        push!(h_ess, out[4])
    end
end


df = DataFrame(Sampler = sampler, Iter = it, Tuning = tuning, J = J_vec,  h1 = h_mat[1,:], h2 = h_mat[2,:], h3 = h_mat[3,:])
ess2 = copy(h_ess)
CSV.write(datadir("RJExp2.csv"), df)

Random.seed!(12515)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = collect(0.1:0.1:3.1)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 10_000
nsmp = 1_000
test_times = collect(0.05:0.05:2.95)
sampler = []
J_vec = []
h_mat = Matrix{Float64}(undef, 3, 0)
it = []
tuning = []
exp_its = 5
h_test = [0.5, 1.5, 2.5]
h_ess = Vector{Vector{Float64}}()


Random.seed!(12515)
priors1 = BasicPrior(1.0, FixedV(1.0), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.1), [RandomWalk()], [], 2)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
nits = 20_000
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 5.0, false, true, 0.01, 50.0)
@time out1 = pem_fit(state0, dat, priors1, settings, test_times, 1_000)

Random.seed!(12515)
nits = 1_000_000
priors3 = BasicPrior(1.0, FixedV(1.0), FixedW([0.5]), 0.0, RJ(10.0, 1, 0.01, 100.0, 3.1), [RandomWalk()], [], 2)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 50.0)
@time out2 = pem_fit(state0, dat, priors3, settings, test_times, 10_000)

Random.seed!(12515)
nits = 1_000_000
priors3 = BasicPrior(1.0, FixedV(1.0), FixedW([0.5]), 0.0, RJ(5.0, 1, 1.0, 100.0, 3.1), [RandomWalk()], [], 2)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 50.0)
@time out3 = pem_fit(state0, dat, priors3, settings, test_times, 10_000)

Random.seed!(12515)
nits = 1_000_000
priors3 = BasicPrior(1.0, FixedV(1.0), FixedW([0.5]), 0.0, RJ(5.0, 1, 0.25, 100.0, 3.1), [RandomWalk()], [], 2)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 50.0)
@time out4 = pem_fit(state0, dat, priors3, settings, test_times, 10_000)


c1 = cts_transform(cumsum(out1[1]["Sk_θ"], dims = 2), out1[1]["Sk_s_loc"], h_test)
c2 = cts_transform(cumsum(out1[2]["Sk_θ"], dims = 2), out1[2]["Sk_s_loc"], h_test)
h1 = cts_transform(cumsum(out1[2]["Sk_θ"], dims = 2), out1[2]["Sk_s_loc"], collect(0.01:0.01:3.0))
h2 = cts_transform(cumsum(out1[1]["Sk_θ"], dims = 2), out1[1]["Sk_s_loc"], collect(0.01:0.01:3.0))
c1_ = cts_transform(cumsum(out2[1]["Sk_θ"], dims = 2), out2[1]["Sk_s_loc"], h_test)
c2_ = cts_transform(cumsum(out2[2]["Sk_θ"], dims = 2), out2[2]["Sk_s_loc"], h_test)
h1_ = cts_transform(cumsum(out2[1]["Sk_θ"][:,:,1:50:1_000_000], dims = 2), out2[1]["Sk_s_loc"][:,1:50:1_000_000], collect(0.01:0.01:3.0))
h2_ = cts_transform(cumsum(out2[2]["Sk_θ"][:,:,1:50:1_000_000], dims = 2), out2[2]["Sk_s_loc"][:,1:50:1_000_000], collect(0.01:0.01:3.0))
c11 = cts_transform(cumsum(out3[1]["Sk_θ"], dims = 2), out3[1]["Sk_s_loc"], h_test)
c21 = cts_transform(cumsum(out3[2]["Sk_θ"], dims = 2), out3[2]["Sk_s_loc"], h_test)
h11 = cts_transform(cumsum(out3[2]["Sk_θ"][:,:,1:50:1_000_000], dims = 2), out3[2]["Sk_s_loc"][:,1:50:1_000_000], collect(0.01:0.01:3.0))
h21 = cts_transform(cumsum(out3[1]["Sk_θ"][:,:,1:50:1_000_000], dims = 2), out3[1]["Sk_s_loc"][:,1:50:1_000_000], collect(0.01:0.01:3.0))
c12 = cts_transform(cumsum(out4[1]["Sk_θ"], dims = 2), out4[1]["Sk_s_loc"], h_test)
c22 = cts_transform(cumsum(out4[2]["Sk_θ"], dims = 2), out4[2]["Sk_s_loc"], h_test)
h12 = cts_transform(cumsum(out4[2]["Sk_θ"][:,:,1:50:1_000_000], dims = 2), out4[2]["Sk_s_loc"][:,1:50:1_000_000], collect(0.01:0.01:3.0))
h22 = cts_transform(cumsum(out4[1]["Sk_θ"][:,:,1:50:1_000_000], dims = 2), out4[1]["Sk_s_loc"][:,1:50:1_000_000], collect(0.01:0.01:3.0))


plot(collect(0.01:0.01:3.0), exp.(vec(mean(h1[:,:,:], dims = 3))))
plot!(collect(0.01:0.01:3.0), exp.(vec(mean(h2[:,:,:], dims = 3))))
plot!(collect(0.01:0.01:3.0), exp.(vec(mean(h1_[:,:,:], dims = 3))))
plot!(collect(0.01:0.01:3.0), exp.(vec(mean(h2_[:,:,:], dims = 3))))
plot!(collect(0.01:0.01:3.0), exp.(vec(mean(h11[:,:,:], dims = 3))))
plot!(collect(0.01:0.01:3.0), exp.(vec(mean(h21[:,:,:], dims = 3))))

mean(vec(sum(out1[1]["Sk_s"], dims = 2)))
mean(vec(sum(out3[1]["Sk_s"], dims = 2)))

plot(vec(sum(out1[1]["Sk_s"], dims = 2)))
plot!(vec(sum(out2[1]["Sk_s"], dims = 2)))
plot!(vec(sum(out3[1]["Sk_s"], dims = 2)))

histogram(h1[1,100,:], alpha = 0.1, normalize = true)
histogram!(h1_[1,100,:], alpha = 0.1, normalize = true)

plot(h1[1,120,1_000:end])
plot!(h1_[1,120,1_000:end])

plot(h1_[1,100,1_000:end],h1_[1,120,1_000:end])
plot!(h1[1,100,1_000:end],h1[1,120,1_000:end])


plot(collect(0.01:0.01:3.0), exp.(vec(median(h2[:,:,:], dims = 3))))
plot!(collect(0.01:0.01:3.0), exp.(quantile.(eachrow(h2[1,:,:]), 0.025)))
plot!(collect(0.01:0.01:3.0), exp.(quantile.(eachrow(h2[1,:,:]), 0.975)))
plot!(collect(0.01:0.01:3.0), exp.(vec(median(h2_[:,:,:], dims = 3))), linestyle = :dot)
plot!(collect(0.01:0.01:3.0), exp.(quantile.(eachrow(h2_[1,:,:]), 0.025)), linestyle = :dot)
plot!(collect(0.01:0.01:3.0), exp.(quantile.(eachrow(h2_[1,:,:]), 0.975)), linestyle = :dot)
plot!(collect(0.01:0.01:3.0), exp.(vec(median(h1_[:,:,:], dims = 3))), linestyle = :dot)
plot!(collect(0.01:0.01:3.0), exp.(quantile.(eachrow(h1_[1,:,:]), 0.025)), linestyle = :dot)
plot!(collect(0.01:0.01:3.0), exp.(quantile.(eachrow(h1_[1,:,:]), 0.975)), linestyle = :dot)

plot!(collect(0.01:0.01:3.0), exp.(vec(median(h21[:,:,:], dims = 3))), linestyle = :dot)
plot!(collect(0.01:0.01:3.0), exp.(quantile.(eachrow(h21[1,:,:]), 0.025)), linestyle = :dot)
plot!(collect(0.01:0.01:3.0), exp.(quantile.(eachrow(h21[1,:,:]), 0.975)), linestyle = :dot)
plot!(collect(0.01:0.01:3.0), exp.(vec(median(h22[:,:,:], dims = 3))), linestyle = :dot)
plot!(collect(0.01:0.01:3.0), exp.(quantile.(eachrow(h22[1,:,:]), 0.025)), linestyle = :dot)
plot!(collect(0.01:0.01:3.0), exp.(quantile.(eachrow(h22[1,:,:]), 0.975)), linestyle = :dot)


nits = 10_000
for i in 1:exp_its
    priors1 = BasicPrior(1.0, FixedV(0.5), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.1), [RandomWalk()], [], 2)
    x0, v0, s0 = init_params(p, dat)
    v0 = v0./norm(v0)
    t0 = 0.0
    state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
    settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
    out = pem_fit(state0, dat, priors1, settings, test_times, 1_000)
    push!(J_vec,mean(sum(out[1]["Sk_s"],dims = 2)[1,1,:]))
    h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[1]["Sk_θ"], dims = 2), out[1]["Sk_s_loc"], h_test), dims = 3)[1,:,1])
    push!(J_vec,mean(sum(out[2]["Sk_s"],dims = 2)[1,1,:]))
    h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[2]["Sk_θ"], dims = 2), out[2]["Sk_s_loc"], h_test), dims = 3)[1,:,1])
    push!(it, 2*i - 1)
    push!(it, 2*i)
    push!(sampler,"PDMP")
    push!(tuning, 0.0)
    push!(sampler,"PDMP")
    push!(tuning, 0.0)
    push!(h_ess, out[4])
end

println("-----------")
nits = 100_000
tuning_param = [0.01,0.1,0.2,0.5,1.0]
#tuning_param = [0.1]
for σ in tuning_param
    for i in 1:exp_its
        priors2 = BasicPrior(1.0, FixedV(0.5), FixedW([0.5]), 0.0, RJ(10.0, 1, σ, 100.0, 3.1), [RandomWalk()], [], 2)
        x0, v0, s0 = init_params(p, dat)
        v0 = v0./norm(v0)
        t0 = 0.0
        state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
        settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 5.0)
        out = pem_fit(state0, dat, priors2, settings, test_times, 10_000)
        push!(J_vec,mean(sum(out[1]["Sk_s"][:,:,1:10:end] ,dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[1]["Sk_θ"][:,:,1:10:end], dims = 2), out[1]["Sk_s_loc"][:,1:10:end], h_test), dims = 3)[1,:,1])
        push!(J_vec,mean(sum(out[2]["Sk_s"][:,:,1:10:end],dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[2]["Sk_θ"][:,:,1:10:end], dims = 2), out[2]["Sk_s_loc"][:,1:10:end], h_test), dims = 3)[1,:,1])
        push!(it, 2*i - 1)
        push!(it, 2*i)
        push!(sampler,"PDMPRJ")
        push!(tuning, σ)
        push!(sampler,"PDMPRJ")
        push!(tuning, σ)
        push!(h_ess, out[4])
    end
end
println("-----------")
nits = 100_000
for σ in tuning_param
    for i in 1:exp_its
        priors3 = BasicPrior(1.0, FixedV(0.5), FixedW([0.5]), 0.0, RJ(10.0, 1, σ, 100.0, 3.1), [RandomWalk()], [], 2)
        x0, v0, s0 = init_params(p, dat)
        v0 = v0./norm(v0)
        t0 = 0.0
        state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
        settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 50.0)
        out = pem_fit(state0, dat, priors3, settings, test_times, 10_000)
        push!(J_vec,mean(sum(out[1]["Sk_s"][:,:,1:10:end] ,dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[1]["Sk_θ"][:,:,1:10:end], dims = 2), out[1]["Sk_s_loc"][:,1:10:end], h_test), dims = 3)[1,:,1])
        push!(J_vec,mean(sum(out[2]["Sk_s"][:,:,1:10:end],dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[2]["Sk_θ"][:,:,1:10:end], dims = 2), out[2]["Sk_s_loc"][:,1:10:end], h_test), dims = 3)[1,:,1])
        push!(it, 2*i - 1)
        push!(it, 2*i)
        push!(sampler,"MHRJ")
        push!(tuning, σ)
        push!(sampler,"MHRJ")
        push!(tuning, σ)
        push!(h_ess, out[4])
    end
end


df = DataFrame(Sampler = sampler, Iter = it, Tuning = tuning, J = J_vec,  h1 = h_mat[1,:], h2 = h_mat[2,:], h3 = h_mat[3,:])
ess3 = copy(h_ess)
CSV.write(datadir("RJExp3.csv"), df)

df1 = CSV.read(datadir("RJExp1.csv"), DataFrame)
df2 = CSV.read(datadir("RJExp2.csv"), DataFrame)
df3 = CSV.read(datadir("RJExp3.csv"), DataFrame)
df1[!,"Exp"] .= "Prior"
df2[!,"Exp"] .= "Changepoint"
df3[!,"Exp"] .= "Colon data" 
df = vcat(df1,df2,df3)
df.Tuning
R"""
dat = data.frame($df)
dat = dat %>%
    pivot_longer(J:h3, names_to = "Param", values_to = "Mean_est") %>%
    mutate(Exp = factor(Exp, c("Prior", "Changepoint", "Colon data")))

param_names <- c(
    `J` = "J",
    `h1` = "h(0.5)",
    `h2` = "h(1.5)",
    `h3` = "h(2.5)"
    )

dat %>%
    ggplot(aes(x = as.factor(Tuning), y = Mean_est, fill = Sampler)) + geom_boxplot() +
    theme_classic() + facet_wrap(Param ~ Exp, scales = "free", nrow = 4, labeller = labeller(Param = param_names)) +
    theme(axis.text.x = element_text(angle = 45, hjust=1), legend.position = "bottom") + 
    geom_hline(data = filter(dat, Exp == "Changepoint", Param == "h1"), aes(yintercept = -0.471), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Changepoint", Param == "h2"), aes(yintercept = -0.153), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Changepoint", Param == "h3"), aes(yintercept = -0.072), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Changepoint", Param == "J"), aes(yintercept = 12.824), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Prior", Param == "h1"), aes(yintercept = 0.0), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Prior", Param == "h2"), aes(yintercept = 0.0), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Prior", Param == "h3"), aes(yintercept = 0.0), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Prior", Param == "J"), aes(yintercept = 16), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Colon data", Param == "h1"), aes(yintercept = -1.271), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Colon data", Param == "h2"), aes(yintercept = -1.53), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Colon data", Param == "h3"), aes(yintercept = -2.195), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Colon data", Param == "J"), aes(yintercept = 14.688), linetype = "dashed") + ylab("Estimate") + xlab("Tuning parameter")
    #ggsave($plotsdir("RJ_exp.pdf"), width = 8, height = 10.5)
"""