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

nits = 10_000
for i in 1:exp_its
    priors1 = BasicPrior(1.0, FixedV(0.5), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.1), [RandomWalk()], [], 2)
    x0, v0, s0 = init_params(p, dat)
    v0 = v0./norm(v0)
    t0 = 0.0
    state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
    settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 30.0)
    out = pem_fit(state0, dat, priors1, settings, test_times, 1_000)
    push!(J_vec,mean(sum(out[1]["Sk_s"],dims = 2)[1,1,:]))
    h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[1]["Sk_x"], dims = 2), out[1]["Sk_s_loc"], h_test), dims = 3)[1,:,1])
    push!(J_vec,mean(sum(out[2]["Sk_s"],dims = 2)[1,1,:]))
    h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[1]["Sk_x"], dims = 2), out[2]["Sk_s_loc"], h_test), dims = 3)[1,:,1])
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
        settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 3.0)
        out = pem_fit(state0, dat, priors2, settings, test_times, 10_000)
        push!(J_vec,mean(sum(out[1]["Sk_s"][:,:,1:10:end] ,dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[1]["Sk_x"][:,:,1:10:end], dims = 2), out[1]["Sk_s_loc"][:,1:10:end], h_test), dims = 3)[1,:,1])
        push!(J_vec,mean(sum(out[2]["Sk_s"][:,:,1:10:end],dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[2]["Sk_x"][:,:,1:10:end], dims = 2), out[2]["Sk_s_loc"][:,1:10:end], h_test), dims = 3)[1,:,1])
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
        push!(J_vec,mean(sum(out[1]["Sk_s"][:,:,1:15:end] ,dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[1]["Sk_x"][:,:,1:10:end], dims = 2), out[1]["Sk_s_loc"][:,1:15:end], h_test), dims = 3)[1,:,1])
        push!(J_vec,mean(sum(out[2]["Sk_s"][:,:,1:15:end],dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[2]["Sk_x"][:,:,1:10:end], dims = 2), out[2]["Sk_s_loc"][:,1:15:end], h_test), dims = 3)[1,:,1])
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
priors1 = BasicPrior(1.0, FixedV(0.5), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.1), [RandomWalk()], [], 2)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
nits = 100_000
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.005, 200.0)
@time out1 = pem_fit(state0, dat, priors1, settings, test_times, 1_000)


c1 = cts_transform(cumsum(out1[1]["Sk_θ"], dims = 2), out1[1]["Sk_s_loc"], h_test)
median(vec(sum(out1[1]["Sk_s"], dims = 2)))
median(c1, dims = 3)

nits = 10_000
for i in 1:exp_its
    priors1 = BasicPrior(1.0, FixedV(0.5), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.1), [RandomWalk()], [], 2)
    x0, v0, s0 = init_params(p, dat)
    v0 = v0./norm(v0)
    t0 = 0.0
    state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
    settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 30.0)
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
        settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 3.0)
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
nits = 150_000
for σ in tuning_param
    for i in 1:exp_its
        priors3 = BasicPrior(1.0, FixedV(0.5), FixedW([0.5]), 0.0, RJ(10.0, 1, σ, 100.0, 3.1), [RandomWalk()], [], 2)
        x0, v0, s0 = init_params(p, dat)
        v0 = v0./norm(v0)
        t0 = 0.0
        state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
        settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 50.0)
        out = pem_fit(state0, dat, priors3, settings, test_times, 15_000)
        push!(J_vec,mean(sum(out[1]["Sk_s"][:,:,1:15:end] ,dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[1]["Sk_θ"][:,:,1:15:end], dims = 2), out[1]["Sk_s_loc"][:,1:15:end], h_test), dims = 3)[1,:,1])
        push!(J_vec,mean(sum(out[2]["Sk_s"][:,:,1:15:end],dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[2]["Sk_θ"][:,:,1:15:end], dims = 2), out[2]["Sk_s_loc"][:,1:15:end], h_test), dims = 3)[1,:,1])
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
priors1 = BasicPrior(1.0, FixedV(0.5), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.1), [RandomWalk()], [], 2)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
nits = 100_000
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.005, 200.0)
@time out1 = pem_fit(state0, dat, priors1, settings, test_times, 1_000)


c1 = cts_transform(cumsum(out1[1]["Sk_θ"], dims = 2), out1[1]["Sk_s_loc"], h_test)
mean(vec(sum(out1[1]["Sk_s"], dims = 2)))
median(c1, dims = 3)

nits = 10_000
for i in 1:exp_its
    priors1 = BasicPrior(1.0, FixedV(0.5), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.1), [RandomWalk()], [], 2)
    x0, v0, s0 = init_params(p, dat)
    v0 = v0./norm(v0)
    t0 = 0.0
    state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
    settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 30.0)
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
        settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 3.0)
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
nits = 150_000
for σ in tuning_param
    for i in 1:exp_its
        priors3 = BasicPrior(1.0, FixedV(0.5), FixedW([0.5]), 0.0, RJ(10.0, 1, σ, 100.0, 3.1), [RandomWalk()], [], 2)
        x0, v0, s0 = init_params(p, dat)
        v0 = v0./norm(v0)
        t0 = 0.0
        state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
        settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 50.0)
        out = pem_fit(state0, dat, priors3, settings, test_times, 15_000)
        push!(J_vec,mean(sum(out[1]["Sk_s"][:,:,1:15:end] ,dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[1]["Sk_θ"][:,:,1:15:end], dims = 2), out[1]["Sk_s_loc"][:,1:15:end], h_test), dims = 3)[1,:,1])
        push!(J_vec,mean(sum(out[2]["Sk_s"][:,:,1:15:end],dims = 2)[1,1,:]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out[2]["Sk_θ"][:,:,1:15:end], dims = 2), out[2]["Sk_s_loc"][:,1:15:end], h_test), dims = 3)[1,:,1])
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
    ggplot(aes(x = as.factor(Tuning), y = 0.5*Mean_est, fill = Sampler)) + geom_boxplot() +
    theme_classic() + facet_wrap(Param ~ Exp, scales = "free", nrow = 4, labeller = labeller(Param = param_names)) +
    theme(axis.text.x = element_text(angle = 45, hjust=1), legend.position = "bottom") 
"""
R"""
    + 
    geom_hline(data = filter(dat, Exp == "Changepoint", Param == "h1"), aes(yintercept = -0.41), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Changepoint", Param == "h2"), aes(yintercept = -0.2), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Changepoint", Param == "h3"), aes(yintercept = -0.1), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Changepoint", Param == "J"), aes(yintercept = 8.3), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Prior", Param == "h1"), aes(yintercept = 0.0), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Prior", Param == "h2"), aes(yintercept = 0.0), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Prior", Param == "h3"), aes(yintercept = 0.0), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Prior", Param == "J"), aes(yintercept = 8.5), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Colon data", Param == "h1"), aes(yintercept = -1.2), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Colon data", Param == "h2"), aes(yintercept = -1.57), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Colon data", Param == "h3"), aes(yintercept = -1.96), linetype = "dashed") + 
    geom_hline(data = filter(dat, Exp == "Colon data", Param == "J"), aes(yintercept = 10.8), linetype = "dashed") + ylab("Estimate") + xlab("Tuning parameter")
    #ggsave($plotsdir("RJ_exp.pdf"), width = 8, height = 10.5)
"""