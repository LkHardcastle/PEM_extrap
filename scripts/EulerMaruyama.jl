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

Random.seed!(34124)
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
nits = 5_000
nsmp = 5_000
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 50.0)


priors1 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 150.0, 3.2), [GaussLangevin(2.0,0.2)], [0.1])
priors2 = EulerMaruyama(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 150.0, 3.2), [GaussLangevin(2.0,0.2)], [0.1])

Random.seed!(3463)
test_times = collect(0.1:0.5:3.0)
@time out1 = pem_fit(state0, dat, priors1, settings, test_times)
Random.seed!(24562)
@time out2 = pem_fit(state0, dat, priors2, settings, test_times)

plot(out1[3])
plot!(out2[3])
plot(out1[4])
plot!(out2[4])

Random.seed!(1237)
grid = sort(unique(out1[1]["Sk_s_loc"][cumsum(out1[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
breaks_extrap = collect(3.02:0.25:4.02)
@time extrap1 = barker_extrapolation(out1[1], priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
@time test_smp1 = cts_transform(cumsum(out1[1]["Sk_θ"], dims = 2), out1[1]["Sk_s_loc"], grid)
s1 = vcat(view(exp.(test_smp1), 1, :, :), view(exp.(extrap1), :, :))
df1 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
#plot(scatter(log.(out1[1]["Smp_σ"][1,:]),test_smp[1,10,:], alpha = 0.1))
#histogram(test_smp[1,10,:], bins = -5:0.1:5, alpha = 0.5, normalize = true)
#histogram!(test_smp1[1,10,:], bins = -5:0.1:5, alpha = 0.5, normalize = true)

grid = sort(unique(out2[2]["Sk_s_loc"][cumsum(out2[2]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = sort(unique(out1[1]["Sk_s_loc"][cumsum(out1[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
@time extrap2 = barker_extrapolation(out2[2], priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
@time test_smp2 = cts_transform(cumsum(out2[2]["Sk_θ"][:,:,1:1:end], dims = 2), out2[2]["Sk_s_loc"][:,:,1:1:end], grid)
s1 = vcat(view(exp.(test_smp2), 1, :, :), view(exp.(extrap2), :, :))
df2 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)


R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "Barker")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df2)
dat2 = cbind(dat2, "Euler-Maruyama")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat_1 <- rbind(dat1, dat2)
"""

R"""
p1 <- dat_1 %>%
    subset(Time < 3.0) %>%
    pivot_longer(c(Mean, LCI, UCI)) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 1), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") #+ ylim(0,0.5) + xlim(0,3) 
    p1
"""

Random.seed!(12515)
n = 0
y = rand(Exponential(1.0),n)
breaks = collect(1:1:100)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 5_000
nsmp = 5_000
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 0.0, 0.1, false, true, 0.05, 10.0)
test_times = [10,50,90.0]
var0 = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]
Barker1 = Vector{Vector{Float64}}()
for i in eachindex(var0)
    priors = BasicPrior(1.0, FixedV([i]), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.2), [GaussLangevin(2.0,2.0)], [])
    out1 = pem_fit(state0, dat, priors, settings, test_times)
    push!(Barker1, (sum(out1[1]["Sk_s"][1,:,:], dims = 2)/length(out1[1]["Sk_s"][1,5,:]))[2:end])
end

EM1 = Vector{Vector{Float64}}()
for i in eachindex(var0)
    priors = EulerMaruyama(1.0, FixedV([i]), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.2), [GaussLangevin(2.0,2.0)], [])
    out1 = pem_fit(state0, dat, priors, settings, test_times)
    push!(EM1, (sum(out1[1]["Sk_s"][1,:,:], dims = 2)/length(out1[1]["Sk_s"][1,5,:]))[2:end])
end

Barker2 = Vector{Vector{Float64}}()
for i in eachindex(var0)
    priors = BasicPrior(1.0, FixedV([i]), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.2), [GaussLangevin(2.0,0.2)], [])
    out1 = pem_fit(state0, dat, priors, settings, test_times)
    push!(Barker2, (sum(out1[1]["Sk_s"][1,:,:], dims = 2)/length(out1[1]["Sk_s"][1,5,:]))[2:end])
end
EM2 = Vector{Vector{Float64}}()
for i in eachindex(var0)
    priors = EulerMaruyama(1.0, FixedV([i]), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.2), [GaussLangevin(2.0,0.2)], [])
    out1 = pem_fit(state0, dat, priors, settings, test_times)
    push!(EM2, (sum(out1[1]["Sk_s"][1,:,:], dims = 2)/length(out1[1]["Sk_s"][1,5,:]))[2:end])
end

dfb1 = DataFrame(Barker1, :auto)
rename!(dfb1, Symbol.(var0))
dfb1.Method .= "Barker"

dfem1 = DataFrame(EM1, :auto)
rename!(dfem1, Symbol.(var0))
dfem1.Method .= "Euler-Maruyama"

df1 = vcat(dfb1,dfem1)

dfb2 = DataFrame(Barker2, :auto)
rename!(dfb2, Symbol.(var0))
dfb2.Method .= "Barker"

dfem2 = DataFrame(EM2, :auto)
rename!(dfem2, Symbol.(var0))
dfem2.Method .= "Euler-Maruyama"
df2 = vcat(dfb2,dfem2)

R"""
$df1 %>%
    pivot_longer("0.001":"0.5", names_to  = "step_size") %>%
    ggplot(aes(x = step_size, y = value, col = Method)) + geom_boxplot() +
    theme_classic() + scale_colour_manual(values = cbPalette[6:7]) + geom_hline(yintercept = 0.5, linetype = "dotted")
"""

R"""
$df2 %>%
    pivot_longer("0.001":"0.5", names_to  = "step_size") %>%
    ggplot(aes(x = step_size, y = value, col = Method)) + geom_boxplot() +
    theme_classic() + scale_colour_manual(values = cbPalette[6:7]) + geom_hline(yintercept = 0.5, linetype = "dotted")
"""