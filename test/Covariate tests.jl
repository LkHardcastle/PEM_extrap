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
breaks = collect(0.5:0.5:5)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0

Random.seed!(34734)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
state0 = ECMC2(x0, v0, s0, t0, true, findall(s0))
nits = 100
nsmp = 100000
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.0, 0.5, false, true)
priors = BasicPrior(1.0, FixedV(1.0), FixedW(0.5), 0.0)
@time out3 = pem_sample(state0, dat, priors, settings)
out3["Sk_x"]
plot(out3["Sk_t"],out3["Sk_x"][1,10,:])

Random.seed!(123)
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
t0 = 0.0
priors = BasicPrior(1.0, FixedV(1.0), FixedW(0.5), 0.0)
state0 = BPS(x0, v0, s0, t0, findall(s0))
nits = 100_000
nsmp = 100000
Random.seed!(123)
state0 = ECMC2(x0, v0, s0, t0, true, findall(s0))
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 1.0, false, true)
@time out4 = pem_sample(state0, dat, priors, settings)

smps1 = out4["Smp_trans"]
plot(vcat(0,breaks), vec(hcat(mean(exp.(smps1), dims = 3), mean(exp.(smps1), dims = 3)[end])),linetype=:steppost, xlims = (0,3.5), ylim = (0,0.5))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1[])), 0.025),quantile.(eachrow(exp.(smps1)), 0.025)[end]),linetype=:steppost, xlims = (0,3.5), ylim = (0,1))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.975),quantile.(eachrow(exp.(smps1)), 0.975)[end]),linetype=:steppost, xlims = (0,3.5), ylim = (0,1))

Random.seed!(123)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = collect(0.1:0.1:3.1)
p = 2
cens = df.status
covar = fill(1.0, 1, n)
covar = [covar; transpose(df.sex)]
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
#x0[2,:] = vcat(x0[2,1],zeros(30))
#v0[2,:] = vcat(v0[2,1],zeros(30))
#s0[2,:] = vcat(s0[2,1],zeros(Int,30))
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, t0, true, findall(s0))
nits = 100000
nsmp = 100000
settings = Settings(nits, nsmp, 100000, 0.5,0.5, 0.2, false, true)
Random.seed!(123)
priors = BasicPrior(1.0, PC(0.2, 2, 0.5, 1, Inf), Beta(0.2, 5.0, 15.0), 1.0)
@time out1 = pem_sample(state0, dat, priors, settings)
Random.seed!(123)
priors = BasicPrior(1.0, FixedV(1.0), Beta(0.2, 5.0, 15.0), 1.0)
@time out2 = pem_sample(state0, dat, priors, settings)

s1 = view(exp.(out1["Smp_trans"]), 1, :, :)
df1 = DataFrame(hcat(breaks, mean(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
s2 = view(exp.(out2["Smp_trans"]), 1, :, :)
s3 = exp.(out2["Smp_trans"][1,:,:] .+ out2["Smp_trans"][2,:,:])
plot(out2["Smp_t"],out2["Smp_x"][1,1,:])
plot(out2["Smp_t"],out2["Smp_x"][2,2,:])
plot(out2["Sk_t"],out2["Sk_x"][2,1,:])
plot(out2["Smp_t"],out2["Smp_x"][1,15,:])
plot(out2["Smp_t"],out2["Smp_v"][2,1,:])
plot(out2["Smp_t"],out2["Smp_v"][1,12,:])

df2 = DataFrame(hcat(breaks, mean(s2, dims = 2), mean(s3, dims = 2), quantile.(eachrow(s2), 0.025), quantile.(eachrow(s2), 0.25), quantile.(eachrow(s2), 0.75), quantile.(eachrow(s2), 0.975)), :auto)
df3 = DataFrame(hcat(breaks, mean(s3, dims = 2), quantile.(eachrow(s3), 0.025), quantile.(eachrow(s3), 0.25), quantile.(eachrow(s3), 0.75), quantile.(eachrow(s3), 0.975)), :auto)
R"""
dat1 = data.frame($df1)
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
"""
R"""
dat2 = data.frame($df2)
colnames(dat2) <- c("Time","Mean", "Mean2","LCI","Q1","Q4","UCI") 
"""

R"""
dat3 = data.frame($df3)
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
"""

R"""
p1 <- dat1 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name, linetype = name)) + geom_step() +
    theme_classic() + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,4,4,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dashed","dashed","dotdash")) + ylab("h(t)") + xlab("Time (years)")
"""
R"""    
p2 <- dat2 %>%
    pivot_longer(Mean:Mean2) %>%
    ggplot(aes(x = Time, y = value, col = name, linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,4,4,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dashed","dashed","dotdash")) + ylab("h(t)") + xlab("Time (years)")
p2
#plot_grid(p1,p2)
#ggsave($plotsdir("SnS.pdf"), width = 14, height = 6)
"""
R"""    
p3 <- dat3 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name, linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,4,4,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dashed","dashed","dotdash")) + ylab("h(t)") + xlab("Time (years)")
p3
#plot_grid(p1,p2)
#ggsave($plotsdir("SnS.pdf"), width = 14, height = 6)
"""

[6.131003788582998 0.04400607305463744 -3.6356758743584683 -6.893631683919485 -9.484552883242461 -9.412271848839364 -10.924707694915545 -5.992967451604308 -4.45330491140081 -6.318391995806599 -4.487233725769351 -5.8415771757960036 -6.1695897953305945 -7.181892784182396 -4.961131363425247 -3.237874661923904 -4.84259845680615 -4.929304469358444 -4.799930456819041 -3.9993956485716557 -4.981742334872203 -2.6789170714131254 -2.3320974558282543 -3.0844028898470457 -1.5058050196377226 -1.932079412164772 -0.2101625114408881 -0.3786973805771142 -0.5761493054403941 -0.7639955835231766 0.0; 20.33334353348656 12.881356574925373 7.856736046853967 3.107881246865592 1.7111969949772776 1.1696187102011586 2.4069938147621524 4.186325618256596 4.443010558610491 3.2195455143526024 2.6126580602213774 2.834557951122198 1.2835313738564786 0.09226358252491995 0.16282638519100145 0.597835808369676 3.7014897138199707 2.4583894912151276 1.4566513566953352 0.04003607500529971 1.8950588906434065 1.0760936751978654 0.2807270846352944 -0.6374572331613917 -0.1698607679144839 1.3143929726002255 0.9800130897677882 0.7711171156397112 0.5263789650173673 0.29354684281291904 0.0]
[26.464347322069557 12.92536264798001 4.221060172495498 -3.785750437053893 -7.773355888265184 -8.242653138638206 -8.517713880153392 -1.8066418333477117 -0.010294352790318761 -3.0988464814539967 -1.8745756655479737 -3.0070192246738054 -4.886058421474116 -7.089629201657476 -4.7983049782342455 -2.640038853554228 -1.1411087429861793 -2.470914978143316 -3.3432791001237057 -3.9593595735663563 -3.0866834442287967 -1.60282339621526 -2.05137037119296 -3.7218601230084376 -1.6756657875522065 -0.6176864395645465 0.7698505783269001 0.392419735062597 -0.04977034042302686 -0.4704487407102576 0.0; 6.131003788582998 0.04400607305463744 -3.6356758743584683 -6.893631683919485 -9.484552883242461 -9.412271848839364 -10.924707694915545 -5.992967451604308 -4.45330491140081 -6.318391995806599 -4.487233725769351 -5.8415771757960036 -6.1695897953305945 -7.181892784182396 -4.961131363425247 -3.237874661923904 -4.84259845680615 -4.929304469358444 -4.799930456819041 -3.9993956485716557 -4.981742334872203 -2.6789170714131254 -2.3320974558282543 -3.0844028898470457 -1.5058050196377226 -1.932079412164772 -0.2101625114408881 -0.3786973805771142 -0.5761493054403941 -0.7639955835231766 0.0]