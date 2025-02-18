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

Random.seed!(3453)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = collect(0.25:0.25:3.25)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 20_000
nsmp = 10_000
settings1 = Exact(nits, nsmp, 1_000_000, 1.0, 10.0, 0.5, false, true)
nits = 300_000
nsmp = 1_000
settings2 = Splitting(nits, nsmp, 1_000_000, 1.0, 0.0, 0.1, false, true, 0.005, 1.0)

priors1 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 0.0, CtsPois(5.0, 5.0, 150.0, 3.2), [RandomWalk()], [])
priors2 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 0.0, CtsPois(5.0, 5.0, 150.0, 3.2), [RandomWalk()], [0.1])

Random.seed!(24562)
test_times = collect(0.1:0.5:3.1)
@time out1 = pem_fit(state0, dat, priors1, settings1, test_times)
Random.seed!(24562)
@time out2 = pem_fit(state0, dat, priors2, settings2, test_times)


Random.seed!(34124)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
#n = 0
breaks = collect(0.1:0.1:3.1)
p = 1
cens = df.status
#cens = []
#y = []
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 500_000
nsmp = 10_000
settings1 = Exact(nits, nsmp, 1_000_000, 0.1, 1.0, 0.1, false, true)
nits = 10_000
nsmp = 1_000
settings2 = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 50.0)

priors1 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 150.0, 3.2), [RandomWalk()], [])
priors2 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 150.0, 3.2), [RandomWalk()], [0.1])

Random.seed!(3463)
test_times = collect(0.1:0.5:3.0)
@time out1 = pem_fit(state0, dat, priors1, settings1, test_times)
Random.seed!(24562)
@time out2 = pem_fit(state0, dat, priors2, settings2, test_times)

histogram(unique(out1[2]["Sk_s_loc"]), alpha = 0.5, normalize = true, bins = 0.0:0.1:5)
histogram!(unique(out2[2]["Sk_s_loc"]), alpha = 0.5, normalize = true, bins = 0.0:0.1:5)

plot(out2[1]["Sk_θ"][1,1,:])
plot(out2[1]["Sk_σ"][1,:])
plot(out1[3])
plot!(out2[3])
plot!(out3[3])

plot(out1[1]["Smp_Γ"])
plot(out2[1]["Sk_Γ"])
plot(out1[1]["Smp_ω"][1,:])
plot(out2[1]["Sk_ω"][1,:])
out1[3]
out2[3]
out3[3]

plot(out1[4])
plot!(out2[4])
plot!(out3[4])

histogram(out1[1]["Smp_θ"][1,3,:], alpha = 0.5, normalize = true)
histogram!(out2[1]["Sk_θ"][1,3,:], alpha = 0.5, normalize = true)
plot(out1[1]["Sk_x"][1,2,:], log.(out1[1]["Sk_σ"][1,:]))
plot(out2[1]["Sk_x"][1,2,:], log.(out2[1]["Sk_σ"][1,:]))
plot!(out3[1]["Sk_x"][1,2,:], log.(out3[1]["Sk_σ"][1,:]))

plot(out1[1]["Sk_x"][1,2,:],out1[1]["Sk_x"][1,3,:])
plot!(out2[1]["Sk_x"][1,2,:],out2[1]["Sk_x"][1,3,:])

j = 40
mean(out1[1]["Smp_x"][1,j,:][.!isinf.(out1[1]["Smp_x"][1,j,:])].== 0.0)
mean(out2[1]["Sk_x"][1,j,:][.!isinf.(out2[1]["Sk_x"][1,j,:])].== 0.0)

mean(out1[1]["Smp_x"][1,30,:] .== 0.0)
mean(out2[1]["Sk_x"][1,30,:] .== 0.0)
plot(out2[1]["Sk_x"][1,2,:])
plot(out1[1]["Sk_θ"][1,2,:],out1[1]["Sk_θ"][1,3,:])
plot!(out2[1]["Sk_θ"][1,2,:],out2[1]["Sk_θ"][1,3,:])

plot(out1[1]["Sk_t"],out1[1]["Sk_θ"][1,2,:])
plot!(out2[1]["Sk_t"],out2[1]["Sk_θ"][1,2,:])

plot(out1[1]["Sk_t"], log.(out1[1]["Sk_σ"][1,:]))
plot!(out2[1]["Sk_t"], log.(out2[1]["Sk_σ"][1,:]))
plot(out2[2]["Sk_t"], log.(out2[2]["Sk_σ"][1,:]))
plot(out3[1]["Sk_t"], log.(out3[1]["Sk_σ"][1,:]))
plot(out3[2]["Sk_t"], log.(out3[2]["Sk_σ"][1,:]))

histogram(out1[1]["Smp_θ"][1,2,:], alpha = 0.5, normalize = true, bins = -2:0.1:2)
histogram!(out2[1]["Sk_θ"][1,2,:], alpha = 0.5, normalize = true, bins = -2:0.1:2)


histogram(log.(out1[1]["Smp_σ"][1,:]), alpha = 0.5, normalize = true, bins = -2:0.1:1)
histogram!(log.(out2[2]["Sk_σ"][1,:]), alpha = 0.5, normalize = true, bins = -2:0.1:1)
plot!(collect(0:0.1:2), (cdf.(Exponential(0.5),0.1:0.1:2.1) .-cdf.(Exponential(0.5),0:0.1:2)).*10)

plot(out1[1]["Sk_σ"][1,:])
plot(out2[1]["Sk_σ"][1,:])
plot(out2[1]["Sk_σ"][1,1:15_000],out2[1]["Sk_x"][1,1,1:15_000])
histogram!(out2[1]["Sk_x"][1,2,1:200:end], alpha = 0.1, normalize = true)

histogram(out1[1]["Smp_J"], alpha = 0.3, normalize = true, bins = 0:1:50)
histogram!(out2[1]["Sk_J"], alpha = 0.3, normalize = true, bins = 0:1:50)
plot!(collect(0:1:50), pdf.(Poisson(32), 0:1:50))

S1 = vec(sum(out1[1]["Smp_s"],dims = 2))
S2 = vec(sum(out2[1]["Sk_s"],dims = 2))
histogram(S1, alpha = 0.3, normalize = true, bins = 0:1:50)
histogram!(S2, alpha = 0.3, normalize = true, bins = 0:1:50)
plot!(collect(0:1:50), pdf.(Poisson(16), 0:1:50))

Random.seed!(1237)
grid = sort(unique(out1[1]["Smp_s_loc"][cumsum(out1[1]["Smp_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
breaks_extrap = collect(3.02:0.25:4.02)
@time extrap1 = barker_extrapolation(out1[1], priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1; typeof = "Smp")
@time test_smp1 = cts_transform(cumsum(out1[1]["Smp_θ"], dims = 2), out1[1]["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp1), 1, :, :), view(exp.(extrap1), :, :))
df1 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
#plot(scatter(log.(out1[1]["Smp_σ"][1,:]),test_smp[1,10,:], alpha = 0.1))
#histogram(test_smp[1,10,:], bins = -5:0.1:5, alpha = 0.5, normalize = true)
#histogram!(test_smp1[1,10,:], bins = -5:0.1:5, alpha = 0.5, normalize = true)

grid = sort(unique(out2[2]["Sk_s_loc"][cumsum(out2[2]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = sort(unique(out1[1]["Smp_s_loc"][cumsum(out1[1]["Smp_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
@time extrap2 = barker_extrapolation(out2[2], priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
@time test_smp2 = cts_transform(cumsum(out2[2]["Sk_θ"][:,:,1:1:end], dims = 2), out2[2]["Sk_s_loc"][:,:,1:1:end], grid)
s1 = vcat(view(exp.(test_smp2), 1, :, :), view(exp.(extrap2), :, 1:1:9999))
df2 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

plot(test_smp[1,1,:])
plot(scatter(log.(out2[2]["Sk_σ"][1,1:10:end]),test_smp1[1,10,:], alpha = 0.1))

plot(scatter(log.(out1[1]["Smp_σ"][1,:]),test_smp[1,100,:], alpha = 0.1))
plot(scatter(log.(out1[1]["Smp_σ"][1,:]),test_smp[1,100,:], alpha = 0.1))

test_smp3 = cts_transform(out2[2]["Sk_θ"][:,:,1:10:end], out2[2]["Sk_s_loc"][:,:,1:10:end], grid)
test_smp4 = cts_transform(out2[2]["Sk_x"][:,:,1:10:end], out2[2]["Sk_s_loc"][:,:,1:10:end], grid)
histogram(test_smp3[1,6_000,:], normalize = true, alpha = 0.5, bins = -3:0.1:3.0)
histogram(test_smp4[1,6_000,:], normalize = true, alpha = 0.5, bins = -3:0.1:3.0)
plot(test_smp4[1,6_000,:])
histogram(test_smp1[1,6_000,:], normalize = true, alpha = 0.5, bins = -10:0.1:0.5)
histogram!(test_smp2[1,6_000,:], normalize = true, alpha = 0.5, bins = -10:0.1:0.5)
grid[6_000]


R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "Exact")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df2)
dat2 = cbind(dat2, "Splitting")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat_1 <- rbind(dat1, dat2)
"""

R"""
p1 <- dat_1 %>%
    subset(Time < 3.0) %>%
    pivot_longer(c(Mean, LCI, UCI)) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") #+ ylim(0,0.5) + xlim(0,3) 
    p1
"""