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
breaks = collect(0.5:0.5:1.0)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0

Random.seed!(3463)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 20
nsmp = 10000
settings = Settings(nits, nsmp, 1_000_000, 0.01,0.0, 0.1, false, true)
priors = BasicPrior(1.0, FixedV([1.0]), FixedW([0.5]), 1.0, Fixed(0.1), [RandomWalk()])
@time out3 = pem_sample(state0, dat, priors, settings)


t = vec(out3["Sk_t"])
y1 = vec(out3["Sk_x"][1,1,:])
y2 = vec(out3["Sk_x"][1,2,:])
x1 = copy(y1)
x2 = y1 + y2
df = DataFrame([t, x1, x2, y1, y2], [:t, :x1, :x2, :y1, :y2])

R"""
p1 <- $df %>%
    pivot_longer(x2:x1) %>%
    ggplot(aes(x = t, y = value, col = factor(name, levels = c("x2","x1")))) + geom_line() +
    theme_classic() + scale_colour_manual(values = cbPalette[7:6]) + ylab("State") + xlab("Sampler time (arbitrary units)") +
    theme(legend.position = "none", text = element_text(size = 20))
p2 <- $df %>%
    pivot_longer(y1:y2) %>%
    ggplot(aes(x = t, y = value, col = name)) + geom_line() +
    theme_classic() + scale_colour_manual(values = cbPalette[6:7]) + ylab("State") + xlab("Sampler time (arbitrary units)") +
    theme(legend.position = "none", text = element_text(size = 20))
plot_grid(p1,p2)
ggsave($plotsdir("SM.pdf"), width = 8, height = 4)
"""