using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random, Optim, Roots, SpecialFunctions, Statistics
using Plots, CSV, DataFrames, RCall, Interpolations, MCMCDiagnosticTools, ParetoSmooth

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

Random.seed!(2352)
df = CSV.read(datadir("TA174.csv"), DataFrame)
y = df.death_ty[findall(df.treat .== 0)]
maximum(y)
n = length(y)
breaks = vcat(0.1,collect(0.26:0.25:4.01))
p = 1
cens = df.death[findall(df.treat .== 0.0)]
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 5000
nsmp = 10
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)

test_Gamma = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0]
Random.seed!(23521)
Gamma_used = []
WAIC = []
test_times = [0.5, 1.5, 2.5]
loo_est = []
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
for Γ_ in test_Gamma
    x0, v0, s0 = init_params(p, dat)
    v0 = v0./norm(v0)
    priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(Γ_, 1.0, 100.0, 4.1), [RandomWalk()], [0.1], 2.0)
    @time out = pem_fit(state0, dat, priors, settings, test_times)
    println(out[3]);println(out[4])
    llhood = get_llhood(out[1], dat, 1_000)
    est = ParetoSmooth.psis_loo(llhood, chain_index = ones(size(llhood, 2))).estimates
    push!(loo_est, est[1,1])
    llhood = get_llhood(out[2], dat, 1_000)
    est = ParetoSmooth.psis_loo(llhood, chain_index = ones(size(llhood, 2))).estimates
    push!(loo_est, est[1,1])
    push!(Gamma_used, Γ_)
    push!(Gamma_used, Γ_)
end 
plot(scatter(Gamma_used, -2*loo_est))

Random.seed!(2352)
df = CSV.read(datadir("TA174.csv"), DataFrame)
y = df.death_ty[findall(df.treat .== 1.0)]
maximum(y)
n = length(y)
breaks = vcat(0.1,collect(0.26:0.25:4.01))
p = 1
cens = df.death[findall(df.treat .== 1.0)]
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 10_000
nsmp = 10
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)

test_Gamma = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0]
Random.seed!(23521)
Gamma_used = []
WAIC = []
test_times = [0.5, 1.5, 2.5]
loo_est = []
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
for Γ_ in test_Gamma
    x0, v0, s0 = init_params(p, dat)
    v0 = v0./norm(v0)
    priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(Γ_, 1.0, 100.0, 4.1), [RandomWalk()], [0.1], 2.0)
    @time out = pem_fit(state0, dat, priors, settings, test_times)
    println(out[3]);println(out[4])
    llhood = get_llhood(out[1], dat, 1_000)
    est = ParetoSmooth.psis_loo(llhood, chain_index = ones(size(llhood, 2))).estimates
    push!(loo_est, est[1,1])
    llhood = get_llhood(out[2], dat, 1_000)
    est = ParetoSmooth.psis_loo(llhood, chain_index = ones(size(llhood, 2))).estimates
    push!(loo_est, est[1,1])
    push!(Gamma_used, Γ_)
    push!(Gamma_used, Γ_)
end 
plot(scatter(Gamma_used, -2*loo_est))


Random.seed!(2352)
df = CSV.read(datadir("TA174.csv"), DataFrame)
y = df.death_ty
maximum(y)
n = length(y)
breaks = vcat(0.1,collect(0.26:1.0:4.26))
p = 1
cens = df.death
covar = fill(1.0, 1, n)
trt = (df.treat .== 1)
covar = [covar; transpose(trt)]
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
x0[2,:] = vcat(x0[2,1], zeros(size(breaks) .-1))
v0[2,:] = vcat(v0[2,1], 1.0, zeros(size(breaks) .-2))
s0[2,:] = vcat(s0[2,1], true, zeros(Int,size(breaks) .-2))
v0 = v0./norm(v0)
t0 = 0.0
nits = 20_000
nsmp = 10
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
test_Gamma = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
Random.seed!(23521)
Gamma_used = []
WAIC = []
test_times = [0.5, 1.5, 2.5]
loo_est = []
for Γ_ in test_Gamma
    x0, v0, s0 = init_params(p, dat)
    v0 = v0./norm(v0)
    priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5,0.5]), 1.0, CtsPois(Γ_, 1.0, 100.0, 4.1), [RandomWalk(), RandomWalk()], [0.01], 2)
    @time out = pem_fit(state0, dat, priors, settings, test_times)
    println(out[3]);println(out[4])
    llhood = get_llhood(out[1], dat, 1_000)
    est = ParetoSmooth.psis_loo(llhood, chain_index = ones(size(llhood, 2))).estimates
    push!(loo_est, est[1,1])
    llhood = get_llhood(out[2], dat, 1_000)
    est = ParetoSmooth.psis_loo(llhood, chain_index = ones(size(llhood, 2))).estimates
    push!(loo_est, est[1,1])
    push!(Gamma_used, Γ_)
    push!(Gamma_used, Γ_)
end 

plot(scatter(Gamma_used, -2*loo_est))

Random.seed!(2352)
df = CSV.read(datadir("TA174.csv"), DataFrame)
y = df.death_ty
maximum(y)
n = length(y)
breaks = vcat(0.1,collect(0.26:1.0:4.26))
p = 1
cens = df.death
covar = fill(1.0, 1, n)
trt = (df.treat .== 1)
covar = [covar; transpose(trt)]
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
x0[2,:] = vcat(x0[2,1], zeros(size(breaks) .-1))
v0[2,:] = vcat(v0[2,1], 1.0, zeros(size(breaks) .-2))
s0[2,:] = vcat(s0[2,1], true, zeros(Int,size(breaks) .-2))
v0 = v0./norm(v0)
nits = 10_000
nsmp = 10
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5,0.5]), 1.0, CtsPois(15.0, 1.0, 100.0, 4.1), [RandomWalk(), RandomWalk()], [0.01], 2)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
out = pem_fit(state0, dat, priors, settings, test_times)
println(out[3]);println(out[4])

grid = sort(unique(out[1]["Sk_s_loc"][cumsum(out[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:1:length(grid)]
test_smp = cts_transform(cumsum(out[1]["Sk_θ"], dims = 2), out[1]["Sk_s_loc"], grid)[:,:,2_500:end]
s1 = view(exp.(test_smp), 1, :, :)
df1 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

s2 = exp.(view(test_smp, 1, :, :) .+ view(test_smp, 2, :, :))
df2 = DataFrame(hcat(grid, median(s2, dims = 2), quantile.(eachrow(s2), 0.025), quantile.(eachrow(s2), 0.25), quantile.(eachrow(s2), 0.75), quantile.(eachrow(s2), 0.975)), :auto)

R"""
dat1 = data.frame($df1)
dat1$Arm = "Placebo"
dat2 = data.frame($df2)
dat2$Arm = "Treatment"
dat1 = rbind(dat1, dat2)
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Arm") 
p1 <- dat1 %>%
    subset(Time < 4.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Arm, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.3) + xlim(0,4) 
"""

###########################
# Placebo - Gamma fixed
###########################

Random.seed!(35673)
df = CSV.read(datadir("TA174.csv"), DataFrame)
y = df.death_ty[findall(df.treat .== 0)]
maximum(y)
n = length(y)
breaks = vcat(0.01,collect(0.26:0.25:4.01))
p = 1
cens = df.death[findall(df.treat .== 0.0)]
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 20_000
burn_in = 10_000
test_times = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]
nsmp = 10
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 4.1), [GammaLangevin(10,10,1)], [0.1], 2.0)
@time out1 = pem_fit(state0, dat, priors, settings, test_times, burn_in)
println(out1[3]);println(out1[4])
###########################
# Treatment - Gamma fixed
###########################

Random.seed!(16346)
df = CSV.read(datadir("TA174.csv"), DataFrame)
y = df.death_ty[findall(df.treat .== 1)]
n = length(y)
breaks = vcat(0.01,collect(0.26:0.25:4.01))
p = 1
cens = df.death[findall(df.treat .== 1)]
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 20_000
burn_in = 10_000
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 4.1), [GammaLangevin(10,10,1)], [0.1], 2.0)
@time out2 = pem_fit(state0, dat, priors, settings, test_times, burn_in)
println(out2[3]);println(out2[4])
###########################
# Placebo - Gamma converging
###########################

Random.seed!(2876)
df = CSV.read(datadir("TA174.csv"), DataFrame)
y = df.death_ty[findall(df.treat .== 0)]
maximum(y)
n = length(y)
breaks = vcat(0.01,collect(0.26:0.25:4.01))
p = 1
cens = df.death[findall(df.treat .== 0.0)]
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 10_000
burn_in = 5_000
nsmp = 10
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 4.1), [GammaLangevin(10,10,10)], [0.1], 2.0)
@time out3 = pem_fit(state0, dat, priors, settings, test_times, burn_in)
println(out3[3]);println(out3[4])
###########################
# Treatment - Gamma converging
###########################

Random.seed!(2876)
df = CSV.read(datadir("TA174.csv"), DataFrame)
y = df.death_ty[findall(df.treat .== 1)]
maximum(y)
n = length(y)
breaks = vcat(0.01,collect(0.26:0.25:4.01))
p = 1
cens = df.death[findall(df.treat .== 1)]
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 10_000
burn_in = 5_000
nsmp = 10
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 4.1), [GammaLangevin(10,10,10)], [0.1], 2.0)
@time out4 = pem_fit(state0, dat, priors, settings, test_times, burn_in)
println(out4[3]);println(out4[4])
###########################
# Placebo - Gompertz baseline
###########################

Random.seed!(2876)
df = CSV.read(datadir("TA174.csv"), DataFrame)
y = df.death_ty[findall(df.treat .== 0)]
maximum(y)
n = length(y)
breaks = vcat(0.01,collect(0.26:0.25:4.01))
p = 1
cens = df.death[findall(df.treat .== 0.0)]
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 10_000
burn_in = 5_000
nsmp = 10
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 4.1), [GompertzBaseline(0.4)], [0.1], 2.0)
@time out5 = pem_fit(state0, dat, priors, settings, test_times, burn_in)
println(out5[3]);println(out5[4])
###########################
# Treatment - Gompertz baseline
###########################

Random.seed!(2876)
df = CSV.read(datadir("TA174.csv"), DataFrame)
y = df.death_ty[findall(df.treat .== 1)]
maximum(y)
n = length(y)
breaks = vcat(0.01,collect(0.26:0.25:4.01))
p = 1
cens = df.death[findall(df.treat .== 1)]
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 10_000
burn_in = 5_000
nsmp = 10
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 4.1), [GompertzBaseline(0.4)], [0.1], 2.0)
@time out6 = pem_fit(state0, dat, priors, settings, test_times, burn_in)
println(out6[3]);println(out6[4])
###########################
# Placebo - Gompertz centred
###########################

Random.seed!(2876)
df = CSV.read(datadir("TA174.csv"), DataFrame)
y = df.death_ty[findall(df.treat .== 0)]
maximum(y)
n = length(y)
breaks = vcat(0.01,collect(0.26:0.25:4.01))
p = 1
cens = df.death[findall(df.treat .== 0.0)]
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 10_000
burn_in = 5_000
nsmp = 10
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 4.1), [GaussLangevin(t -> (log(0.1) + log(0.5)) .+ 0.1 .*t, t -> 1.0)], [0.1], 2.0)
@time out7 = pem_fit(state0, dat, priors, settings, test_times, burn_in)
println(out7[3]);println(out7[4])
###########################
# Treatment - Gompertz centred
###########################

Random.seed!(2876)
df = CSV.read(datadir("TA174.csv"), DataFrame)
y = df.death_ty[findall(df.treat .== 1)]
maximum(y)
n = length(y)
breaks = vcat(0.01,collect(0.26:0.25:4.01))
p = 1
cens = df.death[findall(df.treat .== 1)]
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 10_000
burn_in = 5_000
nsmp = 10
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 4.1), [GaussLangevin(t -> (log(0.1) + log(0.5)) .+ 0.1 .*t, t-> 1.0)], [0.1], 2.0)
@time out8 = pem_fit(state0, dat, priors, settings, test_times, burn_in)
println(out8[3]);println(out8[4])


m_obs = []
m_obsq1 = []
m_obsq2 = []
m_total = []
m_totalq1 = []
m_totalq2 = []
breaks_extrap = collect(4.12:0.02:15)
nits = 20_000
burn_in = 10_000
grid = sort(unique(out1[1]["Sk_s_loc"][cumsum(out1[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out1[1]["Sk_θ"], dims = 2), out1[1]["Sk_s_loc"], grid)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 4.1), [GammaLangevin(10,10,1)], [0.1], 2)
extrap1 = barker_extrapolation(out1[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.1)
s1 = vcat(view(exp.(test_smp), 1, :, burn_in:nits), view(exp.(extrap1), :, burn_in:nits))
df1 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
push!(m_obs, mean(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1])))
push!(m_obsq1, quantile(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1]), 0.025))
push!(m_obsq2, quantile(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1]), 0.975))
push!(m_total, mean(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1])))
push!(m_totalq1, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.025))
push!(m_totalq2, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.975))
s1 = pem_survival(s1,  vcat(0.0,grid, breaks_extrap))
df1_ = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

grid = sort(unique(out2[1]["Sk_s_loc"][cumsum(out2[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 4.1), [GammaLangevin(10,10,1)], [0.1], 2)
test_smp = cts_transform(cumsum(out2[1]["Sk_θ"], dims = 2), out2[1]["Sk_s_loc"], grid)
extrap1 = barker_extrapolation(out2[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.1)
s1 = vcat(view(exp.(test_smp), 1, :, burn_in:nits), view(exp.(extrap1), :, burn_in:nits))
df2 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
push!(m_obs, mean(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1])))
push!(m_obsq1, quantile(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1]), 0.025))
push!(m_obsq2, quantile(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1]), 0.975))
push!(m_total, mean(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1])))
push!(m_totalq1, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.025))
push!(m_totalq2, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.975))
s1 = pem_survival(s1,  vcat(0.0,grid, breaks_extrap))
df2_ = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

nits = 10_000
burn_in = 5_000
grid = sort(unique(out3[1]["Sk_s_loc"][cumsum(out3[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out3[1]["Sk_θ"], dims = 2), out3[1]["Sk_s_loc"], grid)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 4.1), [GammaLangevin(10,10,10)], [0.1], 2)
extrap1 = barker_extrapolation(out3[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.1)
s1 = vcat(view(exp.(test_smp), 1, :, burn_in:nits), view(exp.(extrap1), :, burn_in:nits))
df3 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
push!(m_obs, mean(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1])))
push!(m_obsq1, quantile(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1]), 0.025))
push!(m_obsq2, quantile(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1]), 0.975))
push!(m_total, mean(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1])))
push!(m_totalq1, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.025))
push!(m_totalq2, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.975))
s1 = pem_survival(s1,  vcat(0.0,grid, breaks_extrap))
df3_ = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

grid = sort(unique(out4[1]["Sk_s_loc"][cumsum(out4[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out4[1]["Sk_θ"], dims = 2), out4[1]["Sk_s_loc"], grid)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 4.1), [GammaLangevin(10,10,10)], [0.1], 2)
extrap1 = barker_extrapolation(out4[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.1)
s1 = vcat(view(exp.(test_smp), 1, :, burn_in:nits), view(exp.(extrap1), :, burn_in:nits))
df4 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
push!(m_obs, mean(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1])))
push!(m_obsq1, quantile(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1]), 0.025))
push!(m_obsq2, quantile(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1]), 0.975))
push!(m_total, mean(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1])))
push!(m_totalq1, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.025))
push!(m_totalq2, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.975))
s1 = pem_survival(s1,  vcat(0.0,grid, breaks_extrap))
df4_ = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

grid = sort(unique(out5[1]["Sk_s_loc"][cumsum(out5[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out5[1]["Sk_θ"], dims = 2), out5[1]["Sk_s_loc"], grid)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 4.1), [GompertzBaseline(0.1)], [0.1], 2)
extrap1 = barker_extrapolation(out5[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.1)
s1 = vcat(view(exp.(test_smp), 1, :, burn_in:nits), view(exp.(extrap1), :, burn_in:nits))
df5 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
push!(m_obs, mean(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1])))
push!(m_obsq1, quantile(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1]), 0.025))
push!(m_obsq2, quantile(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1]), 0.975))
push!(m_total, mean(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1])))
push!(m_totalq1, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.025))
push!(m_totalq2, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.975))
s1 = pem_survival(s1,  vcat(0.0,grid, breaks_extrap))
df5_ = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

grid = sort(unique(out6[1]["Sk_s_loc"][cumsum(out6[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out6[1]["Sk_θ"], dims = 2), out6[1]["Sk_s_loc"], grid)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 4.1), [GompertzBaseline(0.1)], [0.1], 2)
extrap1 = barker_extrapolation(out6[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.1)
s1 = vcat(view(exp.(test_smp), 1, :, burn_in:nits), view(exp.(extrap1), :, burn_in:nits))
df6 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
push!(m_obs, mean(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1])))
push!(m_obsq1, quantile(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1]), 0.025))
push!(m_obsq2, quantile(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1]), 0.975))
push!(m_total, mean(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1])))
push!(m_totalq1, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.025))
push!(m_totalq2, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.975))
s1 = pem_survival(s1,  vcat(0.0,grid, breaks_extrap))
df6_ = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)


grid = sort(unique(out7[1]["Sk_s_loc"][cumsum(out7[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out7[1]["Sk_θ"], dims = 2), out7[1]["Sk_s_loc"], grid)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 4.1), [GaussLangevin(t -> (log(0.1) + log(0.5)) .+ 0.1 .*t, t -> 1.0)], [0.1], 2)
extrap1 = barker_extrapolation(out7[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.1)
s1 = vcat(view(exp.(test_smp), 1, :, burn_in:nits), view(exp.(extrap1), :, burn_in:nits))
df7 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
push!(m_obs, mean(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1])))
push!(m_obsq1, quantile(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1]), 0.025))
push!(m_obsq2, quantile(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1]), 0.975))
push!(m_total, mean(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]) .+ 0.0000001))
push!(m_totalq1, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.025))
push!(m_totalq2, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.975))
s1 = pem_survival(s1,  vcat(0.0,grid, breaks_extrap))
df7_ = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

grid = sort(unique(out8[1]["Sk_s_loc"][cumsum(out8[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out8[1]["Sk_θ"], dims = 2), out8[1]["Sk_s_loc"], grid)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 4.1), [GaussLangevin(t -> (log(0.1) + log(0.5)) .+ 0.1 .*t, t -> 1.0)], [0.1], 2)
extrap1 = barker_extrapolation(out8[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.1)
s1 = vcat(view(exp.(test_smp), 1, :, burn_in:nits), view(exp.(extrap1), :, burn_in:nits))
df8 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
push!(m_obs, mean(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1])))
push!(m_obsq1, quantile(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1]), 0.025))
push!(m_obsq2, quantile(get_meansurv(test_smp[1,:,burn_in:nits], grid, [1]), 0.975))
push!(m_total, mean(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1])))
push!(m_totalq1, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.025))
push!(m_totalq2, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.975))
s1 = pem_survival(s1,  vcat(0.0,grid, breaks_extrap))
df8_ = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

df_meanS = DataFrame(m_obs = m_obs, m_obsq1 = m_obsq1, m_obsq2 = m_obsq2, m_total = m_total, m_totalq1 = m_totalq1, m_totalq2 = m_totalq2)

round.(df_meanS, digits = 2)

#R"""
#library(VGAM)
#pgompertz(20, 0.1,0.2)
#"""
#plot(collect(0.01:0.01:20.0), 0.4*0.005 .*exp.(0.4 .* collect(0.01:0.01:20.0)))
#s1
#plot(vcat(grid, breaks_extrap), median(pem_survival(s1,  vcat(0.0,grid, breaks_extrap)), dims = 2))
#plot!(grid, median(pem_survival(exp.(test_smp[1,:,:]),  vcat(0.0,grid)), dims = 2))

CSV.write(datadir("TA174Models","MeanSurvival.csv"),df_meanS)
CSV.write(datadir("TA174Models","GammaNonWanePlacebo.csv"),df1)
CSV.write(datadir("TA174Models","GammaNonWaneTreat.csv"),df2)
CSV.write(datadir("TA174Models","GammaWanePlacebo.csv"),df3)
CSV.write(datadir("TA174Models","GammaWaneTreat.csv"),df4)
CSV.write(datadir("TA174Models","GompBasePlacebo.csv"),df5)
CSV.write(datadir("TA174Models","GompBaseTreat.csv"),df6)
CSV.write(datadir("TA174Models","GompCentPlacebo.csv"),df7)
CSV.write(datadir("TA174Models","GompCentTreat.csv"),df8)

CSV.write(datadir("TA174Models","GammaNonWanePlaceboSurv.csv"),df1_)
CSV.write(datadir("TA174Models","GammaNonWaneTreatSurv.csv"),df2_)
CSV.write(datadir("TA174Models","GammaWanePlaceboSurv.csv"),df3_)
CSV.write(datadir("TA174Models","GammaWaneTreatSurv.csv"),df4_)
CSV.write(datadir("TA174Models","GompBasePlaceboSurv.csv"),df5_)
CSV.write(datadir("TA174Models","GompBaseTreatSurv.csv"),df6_)
CSV.write(datadir("TA174Models","GompCentPlaceboSurv.csv"),df7_)
CSV.write(datadir("TA174Models","GompCentTreatSurv.csv"),df8_)

df1 = CSV.read(datadir("TA174Models","GammaNonWanePlacebo.csv"),DataFrame)
df2 = CSV.read(datadir("TA174Models","GammaNonWaneTreat.csv"),DataFrame)
df3 = CSV.read(datadir("TA174Models","GammaWanePlacebo.csv"),DataFrame)
df4 = CSV.read(datadir("TA174Models","GammaWaneTreat.csv"),DataFrame)
df5 = CSV.read(datadir("TA174Models","GompBasePlacebo.csv"),DataFrame)
df6 = CSV.read(datadir("TA174Models","GompBaseTreat.csv"),DataFrame)
df7 = CSV.read(datadir("TA174Models","GompCentPlacebo.csv"),DataFrame)
df8 = CSV.read(datadir("TA174Models","GompCentTreat.csv"),DataFrame)

df1 = CSV.read(datadir("TA174Models","GammaNonWanePlaceboSurv.csv"),DataFrame)
df2 = CSV.read(datadir("TA174Models","GammaNonWaneTreatSurv.csv"),DataFrame)
df3 = CSV.read(datadir("TA174Models","GammaWanePlaceboSurv.csv"),DataFrame)
df4 = CSV.read(datadir("TA174Models","GammaWaneTreatSurv.csv"),DataFrame)
df5 = CSV.read(datadir("TA174Models","GompBasePlaceboSurv.csv"),DataFrame)
df6 = CSV.read(datadir("TA174Models","GompBaseTreatSurv.csv"),DataFrame)
df7 = CSV.read(datadir("TA174Models","GompCentPlaceboSurv.csv"),DataFrame)
df8 = CSV.read(datadir("TA174Models","GompCentTreatSurv.csv"),DataFrame)

R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "Gamma fixed")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df3)
dat2 = cbind(dat2, "Gamma - converging")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df5)
dat3 = cbind(dat3, "Gompertz baseline")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df7)
dat4 = cbind(dat4, "Gompertz centred")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_1 <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
dat1 = data.frame($df2)
dat1 = cbind(dat1, "Gamma fixed")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df4)
dat2 = cbind(dat2, "Gamma - converging")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df6)
dat3 = cbind(dat3, "Gompertz baseline")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df8)
dat4 = cbind(dat4, "Gompertz centred")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_2 <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
p1 <- dat_1 %>%
    subset(Time < 4.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)")

p2 <- dat_2 %>%
    subset(Time < 4.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)")

p3 <- dat_1 %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)")

p4 <- dat_2 %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)")
plot_grid(p1,p2,p3,p4, nrow = 2)
ggsave($plotsdir("TA174Survival.pdf"), width = 8, height = 6)
"""





Random.seed!(2352)
df = CSV.read(datadir("TA174.csv"), DataFrame)
y = df.death_ty
maximum(y)
n = length(y)
breaks = vcat(0.1,collect(0.26:1.0:4.26))
p = 1
cens = df.death
covar = fill(1.0, 1, n)
trt = (df.treat .== 1)
covar = [covar; transpose(trt)]
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
x0[2,:] = vcat(x0[2,1], zeros(size(breaks) .-1))
v0[2,:] = vcat(v0[2,1], 1.0, zeros(size(breaks) .-2))
s0[2,:] = vcat(s0[2,1], true, zeros(Int,size(breaks) .-2))
v0 = v0./norm(v0)
nits = 10_000
nsmp = 10
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5,0.5]), 1.0, CtsPois(15.0, 1.0, 100.0, 4.1), [GammaLangevin(1,2,1), GaussLangevin(t -> 0.0,t -> 1.0)], [0.01], 2)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
out = pem_fit(state0, dat, priors, settings, test_times)
println(out[3]);println(out[4])

grid = sort(unique(out[1]["Sk_s_loc"][cumsum(out[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:1:length(grid)]
test_smp = cts_transform(cumsum(out[1]["Sk_θ"], dims = 2), out[1]["Sk_s_loc"], grid)[:,:,2_500:end]
s1 = view(exp.(test_smp), 1, :, :)
df1 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

s2 = exp.(view(test_smp, 1, :, :) .+ view(test_smp, 2, :, :))
df2 = DataFrame(hcat(grid, median(s2, dims = 2), quantile.(eachrow(s2), 0.025), quantile.(eachrow(s2), 0.25), quantile.(eachrow(s2), 0.75), quantile.(eachrow(s2), 0.975)), :auto)

R"""
dat1 = data.frame($df1)
dat1$Arm = "Placebo"
dat2 = data.frame($df2)
dat2$Arm = "Treatment"
dat1 = rbind(dat1, dat2)
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Arm") 
p1 <- dat1 %>%
    subset(Time < 4.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Arm, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.3) + xlim(0,4) 
"""

