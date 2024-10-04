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
library(ggsurvfit)
library(survival)
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
"""

Random.seed!(123)
n = 500
prob = [1 - exp(-0.05*.5),  exp(-0.05*.5)*(1-exp(-0.1*2.0)), exp(-0.05*.5-0.1*2.0)]
prob_norm = prob/sum(prob)
int = rand(Categorical(prob_norm), n)
y = Vector{Float64}()
for i in 1:n
    if int[i] == 1
        push!(y, rand(Uniform(0,.5)))
    elseif int[i] == 2
        push!(y, rand(Uniform(.5,2.5)))
    else
       push!(y,2.5) 
    end
end


cens = (int .!= 3).*1.0

breaks = collect(0.1:0.1:2.6)
p = 1
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
nits = 100_000
nsmp = 20_000

Random.seed!(23462)
settings = Settings(nits, nsmp, 1_000_000, 2.0, 2.0, 5.0, false, true)
priors = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]) , 1.0, Cts(3.0, 50.0, 2.6), [RandomWalk()])
@time out1 = pem_sample(state0, dat, priors, settings)

Random.seed!(23462)
settings = Settings(nits, nsmp, 1_000_000, 2.0, 2.0, 5.0, false, true)
priors = BasicPrior(1.0, InvGamma([0.5],[0.01],[0.01]), Beta([0.4], [10.0], [10.0]) , 1.0, Cts(3.0, 50.0, 2.6), [RandomWalk()])
@time out2 = pem_sample(state0, dat, priors, settings)

grid = sort(unique(out2["Smp_s_loc"]))[1:(end -1)]
test_smp = cts_transform(cumsum(out2["Smp_x"], dims = 2), out2["Smp_s_loc"], grid)
s2 = view(exp.(test_smp), 1, :, :)
df2 = DataFrame(hcat(grid, median(s2, dims = 2), quantile.(eachrow(s2), 0.025), quantile.(eachrow(s2), 0.25), quantile.(eachrow(s2), 0.75), quantile.(eachrow(s2), 0.975)), :auto)



R"""
dat2 = data.frame($df2)
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
"""

R"""
p2 <- dat2 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name, linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,4,4,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dashed","dashed","dotdash")) + ylab("h(t)") + xlab("Time")
p2
#ggsave($plotsdir("GammaPrior.pdf"), width = 8, height = 6)
"""