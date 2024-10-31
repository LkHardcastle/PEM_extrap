using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random, Optim, Roots, SpecialFunctions
using Plots, CSV, DataFrames, RCall, Interpolations

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))

R"""
library(ggplot2)
library(dplyr)
library(tidyr)
library(cowplot)
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
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
nits = 10_000
nsmp = 50_000

Random.seed!(14124)
grid = [0.5, 1.5, 2.5]
exp_its = 10
priors1 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 1.0, Cts(5.0, 100.0, 3.1), [RandomWalk()])
priors2 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, 0.2, 100.0, 3.1), [RandomWalk()])
priors3 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, 0.2, 100.0, 3.1), [RandomWalk()])

sampler = []
J_vec = []
h_mat = Matrix{Float64}(undef, 3, 0)
it = []
tuning = []
for i in 1:exp_its
    x0, v0, s0 = init_params(p, dat)
    v0 = v0./norm(v0)
    t0 = 0.0
    state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
    settings = Settings(nits, nsmp, 1_000_000, 1.0, 1.0, 1.0, false, true)
    out = pem_sample(state0, dat, priors1, settings)
    push!(J_vec,mean(sum(out["Smp_s"],dims = 2)[1,1,:]))
    h_mat = hcat(h_mat, mean(cts_transform(cumsum(out["Smp_x"], dims = 2), out["Smp_s_loc"], grid), dims = 3)[1,:,1])
    push!(it, i)
    push!(sampler,"PDMP")
    push!(tuning, 0.0)
end


tuning_param = [0.01,0.1,0.2,0.5,1.0]
for σ in tuning_param
    for i in 1:exp_its
        priors2 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, σ, 100.0, 3.1), [RandomWalk()])
        x0, v0, s0 = init_params(p, dat)
        v0 = v0./norm(v0)
        t0 = 0.0
        state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
        settings = Settings(nits, nsmp, 1_000_000, 1.0, 5.0, 1.0, false, true)
        out = pem_sample(state0, dat, priors2, settings)
        push!(J_vec,mean(out["Smp_J"]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out["Smp_x"], dims = 2), out["Smp_s_loc"], grid), dims = 3)[1,:,1])
        push!(it, i)
        push!(sampler,"PDMPRJ")
        push!(tuning, σ)
    end
end
for σ in tuning_param
    for i in 1:exp_its
        priors3 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, σ, 100.0, 3.1), [RandomWalk()])
        x0, v0, s0 = init_params(p, dat)
        v0 = v0./norm(v0)
        t0 = 0.0
        state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
        settings = Settings(nits, nsmp, 1_000_000, 1.0, 1.0, 1.0, false, true)
        out = pem_sample(state0, dat, priors3, settings)
        push!(J_vec,mean(out["Sk_J"]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out["Smp_x"], dims = 2), out["Smp_s_loc"], grid), dims = 3)[1,:,1])
        push!(it, i)
        push!(sampler,"MHRJ")
        push!(tuning, σ)
    end
end

df = DataFrame(Sampler = sampler, Iter = it, Tuning = tuning, J = J_vec,  h1 = h_mat[1,:], h2 = h_mat[2,:], h3 = h_mat[3,:])
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
histogram(y)
sum(cens)
y1
cens
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
nits = 10_000
nsmp = 50_000

Random.seed!(14124)
grid = [0.5, 1.5, 2.5]
exp_its = 10
priors1 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 1.0, Cts(5.0, 100.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, 0.2, 100.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, 0.2, 100.0, 3.2), [RandomWalk()])

sampler = []
J_vec = []
h_mat = Matrix{Float64}(undef, 3, 0)
it = []
tuning = []
for i in 1:exp_its
    x0, v0, s0 = init_params(p, dat)
    v0 = v0./norm(v0)
    t0 = 0.0
    state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
    settings = Settings(nits, nsmp, 1_000_000, 1.0, 1.0, 1.0, false, true)
    out = pem_sample(state0, dat, priors1, settings)
    push!(J_vec,mean(sum(out["Smp_s"],dims = 2)[1,1,:]))
    h_mat = hcat(h_mat, mean(cts_transform(cumsum(out["Smp_x"], dims = 2), out["Smp_s_loc"], grid), dims = 3)[1,:,1])
    push!(it, i)
    push!(sampler,"PDMP")
    push!(tuning, 0.0)
end


tuning_param = [0.01,0.1,0.2,0.5,1.0]
for σ in tuning_param
    for i in 1:exp_its
        priors2 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, σ, 100.0, 3.2), [RandomWalk()])
        x0, v0, s0 = init_params(p, dat)
        v0 = v0./norm(v0)
        t0 = 0.0
        state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
        settings = Settings(nits, nsmp, 1_000_000, 1.0, 5.0, 1.0, false, true)
        out = pem_sample(state0, dat, priors2, settings)
        push!(J_vec,mean(out["Smp_J"]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out["Smp_x"], dims = 2), out["Smp_s_loc"], grid), dims = 3)[1,:,1])
        push!(it, i)
        push!(sampler,"PDMPRJ")
        push!(tuning, σ)
    end
end
for σ in tuning_param
    for i in 1:exp_its
        priors3 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, σ, 100.0, 3.2), [RandomWalk()])
        x0, v0, s0 = init_params(p, dat)
        v0 = v0./norm(v0)
        t0 = 0.0
        state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
        settings = Settings(nits, nsmp, 1_000_000, 1.0, 1.0, 1.0, false, true)
        out = pem_sample(state0, dat, priors3, settings)
        push!(J_vec,mean(out["Sk_J"]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out["Smp_x"], dims = 2), out["Smp_s_loc"], grid), dims = 3)[1,:,1])
        push!(it, i)
        push!(sampler,"MHRJ")
        push!(tuning, σ)
    end
end

df = DataFrame(Sampler = sampler, Iter = it, Tuning = tuning, J = J_vec,  h1 = h_mat[1,:], h2 = h_mat[2,:], h3 = h_mat[3,:])
CSV.write(datadir("RJExp2.csv"), df)


Random.seed!(9102)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = collect(0.1:0.1:3.1)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
nits = 50_000
nsmp = 50_000

Random.seed!(14124)
grid = [0.5, 1.5, 2.5]
exp_its = 1
priors1 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 1.0, Cts(5.0, 100.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, 0.2, 100.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, 0.2, 100.0, 3.2), [RandomWalk()])

sampler = []
J_vec = []
h_mat = Matrix{Float64}(undef, 3, 0)
it = []
tuning = []
for i in 1:exp_its
    x0, v0, s0 = init_params(p, dat)
    v0 = v0./norm(v0)
    t0 = 0.0
    state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
    settings = Settings(nits, nsmp, 1_000_000, 1.0, 1.0, 1.0, false, true)
    out1 = pem_sample(state0, dat, priors1, settings)
    push!(J_vec,mean(sum(out1["Smp_s"],dims = 2)[1,1,:]))
    h_mat = hcat(h_mat, mean(cts_transform(cumsum(out1["Smp_x"], dims = 2), out1["Smp_s_loc"], grid), dims = 3)[1,:,1])
    push!(it, i)
    push!(sampler,"PDMP")
    push!(tuning, 0.0)
end

tuning_param = [0.01,0.1,0.2,0.5,1.0]
for σ in tuning_param
    for i in 1:exp_its
        priors2 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, σ, 100.0, 3.2), [RandomWalk()])
        x0, v0, s0 = init_params(p, dat)
        v0 = v0./norm(v0)
        t0 = 0.0
        state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
        settings = Settings(nits, nsmp, 1_000_000, 1.0, 5.0, 1.0, false, true)
        out2 = pem_sample(state0, dat, priors2, settings)
        push!(J_vec,mean(out1["Smp_J"]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out1["Smp_x"], dims = 2), out1["Smp_s_loc"], grid), dims = 3)[1,:,1])
        push!(it, i)
        push!(sampler,"PDMPRJ")
        push!(tuning, σ)
    end
end
Random.seed!(3463)
nsmp = 100_000
for σ in tuning_param
    for i in 1:exp_its
        priors3 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, σ, 100.0, 3.2), [RandomWalk()])
        x0, v0, s0 = init_params(p, dat)
        v0 = v0./norm(v0)
        t0 = 0.0
        state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
        settings = Settings(nits, nsmp, 1_000_000, 1.0, 1.0, 1.0, false, true)
        out3 = pem_sample(state0, dat, priors3, settings)
        push!(J_vec,mean(out1["Sk_J"]))
        h_mat = hcat(h_mat, mean(cts_transform(cumsum(out1["Smp_x"], dims = 2), out1["Smp_s_loc"], grid), dims = 3)[1,:,1])
        push!(it, i)
        push!(sampler,"MHRJ")
        push!(tuning, σ)
    end
end

df = DataFrame(Sampler = sampler, Iter = it, Tuning = tuning, J = J_vec,  h1 = h_mat[1,:], h2 = h_mat[2,:], h3 = h_mat[3,:])
CSV.write(datadir("RJExp3.csv"), df)
plot(sum(out1["Smp_s"],dims = 2)[1,1,1:3900])
plot!(out2["Smp_J"][1:3900])
plot!(out3["Sk_J"][1:3900])
mean(sum(out1["Smp_s"],dims = 2)[1,1,:])
mean(out2["Smp_J"])
mean(out3["Sk_J"])

df1 = CSV.read(datadir("RJExp1.csv"), DataFrame)
df2 = CSV.read(datadir("RJExp2.csv"), DataFrame)
df3 = CSV.read(datadir("RJExp3.csv"), DataFrame)

R"""
dat1 = data.frame($df1)
dat1 %>%
    ggplot(aes(x =  interaction(Sampler,as.factor(Tuning)), y = J, fill = Sampler)) + geom_boxplot() +
        theme_classic() + geom_hline(yintercept = 8.5, linetype = "dotted")
"""
R"""
dat1 = data.frame($df1)
dat1 %>%
    ggplot(aes(x =  interaction(Sampler,as.factor(Tuning)), y = h1, fill = Sampler)) + geom_boxplot() +
        theme_classic() 
"""
R"""
dat1 = data.frame($df1)
dat1 %>%
    ggplot(aes(x =  interaction(Sampler,as.factor(Tuning)), y = h2, fill = Sampler)) + geom_boxplot() +
        theme_classic() 
"""
R"""
dat1 = data.frame($df1)
dat1 %>%
    ggplot(aes(x =  interaction(Sampler,as.factor(Tuning)), y = h3, fill = Sampler)) + geom_boxplot() +
        theme_classic() 
"""


R"""
dat1 = data.frame($df2)
dat1 %>%
    ggplot(aes(x =  interaction(Sampler,as.factor(Tuning)), y = J, fill = Sampler)) + geom_boxplot() +
        theme_classic() 
"""
R"""
dat1 = data.frame($df2)
dat1 %>%
    ggplot(aes(x =  interaction(Sampler,as.factor(Tuning)), y = h1, fill = Sampler)) + geom_boxplot() +
        theme_classic() 
"""
R"""
dat1 = data.frame($df2)
dat1 %>%
    ggplot(aes(x =  interaction(Sampler,as.factor(Tuning)), y = h2, fill = Sampler)) + geom_boxplot() +
        theme_classic() 
"""
R"""
dat1 = data.frame($df2)
dat1 %>%
    ggplot(aes(x =  interaction(Sampler,as.factor(Tuning)), y = h3, fill = Sampler)) + geom_boxplot() +
        theme_classic() 
"""


R"""
dat1 = data.frame($df3)
dat1 %>%
    ggplot(aes(x =  interaction(Sampler,as.factor(Tuning)), y = J, fill = Sampler)) + geom_boxplot() +
        theme_classic() 
"""
R"""
dat1 = data.frame($df3)
dat1 %>%
    ggplot(aes(x =  interaction(Sampler,as.factor(Tuning)), y = h1, fill = Sampler)) + geom_boxplot() +
        theme_classic() 
"""
R"""
dat1 = data.frame($df3)
dat1 %>%
    ggplot(aes(x =  interaction(Sampler,as.factor(Tuning)), y = h2, fill = Sampler)) + geom_boxplot() +
        theme_classic() 
"""
R"""
dat1 = data.frame($df3)
dat1 %>%
    ggplot(aes(x =  interaction(Sampler,as.factor(Tuning)), y = h3, fill = Sampler)) + geom_boxplot() +
        theme_classic() 
"""