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

######## Run models for different Poisson intensities
######## Select best model comprimising between minimising LOOCIC and appearance of curve

Random.seed!(2352)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = vcat(0.01,collect(0.26:0.25:3.01))
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 10_000
nsmp = 10

settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)

test_Gamma = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0]
Random.seed!(23521)
Gamma_used = []
WAIC = []
test_times = [0.5, 1.5, 2.5]
loo_est = []
for Γ_ in test_Gamma
    x0, v0, s0 = init_params(p, dat)
    v0 = v0./norm(v0)
    priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(Γ_, 1.0, 100.0, 3.1), [RandomWalk()], [0.1], 2.0)
    @time out = pem_fit(state0, dat, priors, settings, test_times)
    println(out[3]);println(out[4])
    push!(WAIC, get_WAIC(out[1], dat, 1_000)[1])
    push!(WAIC, get_WAIC(out[2], dat, 1_000)[1])
    llhood = get_llhood(out[1], dat, 1_000)
    est = ParetoSmooth.psis_loo(llhood, chain_index = ones(size(llhood, 2))).estimates
    push!(loo_est, est[1,1])
    llhood = get_llhood(out[2], dat, 1_000)
    est = ParetoSmooth.psis_loo(llhood, chain_index = ones(size(llhood, 2))).estimates
    push!(loo_est, est[1,1])
    push!(Gamma_used, Γ_)
    push!(Gamma_used, Γ_)
end 

plot(scatter(Gamma_used, WAIC))
scatter!(Gamma_used, -2*loo_est)

x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 100.0, 3.1), [RandomWalk()], [0.1], 2)
out = pem_fit(state0, dat, priors, settings, test_times)

grid = sort(unique(out[1]["Sk_s_loc"][cumsum(out[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
test_smp = cts_transform(cumsum(out[1]["Sk_θ"], dims = 2), out[1]["Sk_s_loc"], grid)
s1 = view(exp.(test_smp), 1, :, :)
df11 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

R"""
dat1 = data.frame($df11)
dat1 = cbind(dat1, "Pois(5)")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
p1 <- dat1 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) 
"""


Random.seed!(2352)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = vcat(0.01,collect(0.26:0.25:3.01))
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 10_000
nsmp = 10

settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)

Random.seed!(23521)
test_a = [1.0, 2.5, 5.0, 10.0]
test_b = [0.1, 0.5, 1.0, 2.5]
WAIC = []
test_times = [0.5, 1.5, 2.5]
loo_est = []
a_used = []
b_used = []
df_vec = Vector{DataFrame}()
Random.seed!(23524)
for a_ in test_a
    for b_ in test_b
        x0, v0, s0 = init_params(p, dat)
        v0 = v0./norm(v0)
        priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(a_, b_, a_/b_, 1.0, 150.0, 3.02), [RandomWalk()], [0.1])
        out = pem_fit(state0, dat, priors, settings, test_times)
        push!(a_used, a_)
        push!(b_used, b_)
        push!(a_used, a_)
        push!(b_used, b_)
        println(out[3]);println(out[4])
        push!(WAIC, get_WAIC(out[1], dat, 1_000)[1])
        push!(WAIC, get_WAIC(out[2], dat, 1_000)[1])
        llhood = get_llhood(out[1], dat, 1_000)
        est = ParetoSmooth.psis_loo(llhood, chain_index = ones(size(llhood, 2))).estimates
        push!(loo_est, est[1,1])
        llhood = get_llhood(out[2], dat, 1_000)
        est = ParetoSmooth.psis_loo(llhood, chain_index = ones(size(llhood, 2))).estimates
        push!(loo_est, est[1,1])

        grid = sort(unique(out[1]["Sk_s_loc"][cumsum(out[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
        grid = grid[1:10:length(grid)]
        test_smp = cts_transform(cumsum(out[1]["Sk_θ"], dims = 2), out[1]["Sk_s_loc"], grid)
        s1 = view(exp.(test_smp), 1, :, :)
        push!(df_vec, DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto))
    end
end

df1 = df_vec[1]
df2 = df_vec[2]
df3 = df_vec[3]
df4 = df_vec[4]
df5 = df_vec[5]
df6 = df_vec[6]
df7 = df_vec[7]
df8 = df_vec[8]
df9 = df_vec[9]
df10 = df_vec[10]
df11 = df_vec[11]
df12 = df_vec[12]
df13 = df_vec[13]
df14 = df_vec[14]
df15 = df_vec[15]
df16 = df_vec[16]

test_a = [1.0, 2.5, 5.0, 10.0]
test_b = [0.1, 0.5, 1.0, 2.5]
R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "(1, 0.1)")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df2)
dat2 = cbind(dat2, "(1, 0.5)")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df3)
dat3 = cbind(dat3, "(1, 1.0)")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df4)
dat4 = cbind(dat4, "(1, 2.5)")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_1 <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
dat1 = data.frame($df5)
dat1 = cbind(dat1, "(2.5, 0.1)")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df6)
dat2 = cbind(dat2, "(2.5, 0.5)")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df7)
dat3 = cbind(dat3, "(2.5, 1.0)")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df8)
dat4 = cbind(dat4, "(2.5, 2.5)")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_2 <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
dat1 = data.frame($df9)
dat1 = cbind(dat1, "(5, 0.1)")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df10)
dat2 = cbind(dat2, "(5, 0.5)")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df11)
dat3 = cbind(dat3, "(5, 1.0)")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df12)
dat4 = cbind(dat4, "(5, 2.5)")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_3 <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
dat1 = data.frame($df13)
dat1 = cbind(dat1, "(10, 0.1)")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df14)
dat2 = cbind(dat2, "(10, 0.5)")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df15)
dat3 = cbind(dat3, "(10, 1.0)")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df16)
dat4 = cbind(dat4, "(10, 2.5)")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_4 <- rbind(dat1, dat2, dat3, dat4)
"""


R"""
p1 <- dat_1 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) 
p2 <- dat_2 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7,2)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) 
p3 <- dat_3 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) 
p4 <- dat_4 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) 
plot_grid(p1,p2,p3,p4, nrow = 2)
#ggsave($plotsdir("CovariateColon.pdf"), width = 8, height = 6)
"""

## For optimal models re-run for Gauss/Gamma/Gompertz
## Report mean survival for optimal Poisson/NB model for each set of priors

Random.seed!(2352)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = vcat(0.01,collect(0.26:0.25:3.01))
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 10_000
nsmp = 10
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
test_times = collect(0.2:0.2:3.0)

priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 100.0, 3.1), [RandomWalk()], [0.1], 2)
out1 = pem_fit(state0, dat, priors, settings, test_times)
println(out1[3]);println(out1[4])
#priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 100.0, 3.1), [GaussLangevin(log(0.2),1.0)], [0.1], 2)
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 100.0, 3.1), [GaussLangevin(log(0.29),0.4)], [0.1], 2)
out2 = pem_fit(state0, dat, priors, settings, test_times)
println(out2[3]);println(out2[4])
#priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 100.0, 3.1), [GammaLangevin(2,7)], [0.1], 2)
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 100.0, 3.1), [GammaLangevin(2,7)], [0.1], 2)
out3 = pem_fit(state0, dat, priors, settings, test_times)
println(out3[3]);println(out3[4])
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 100.0, 3.1), [GompertzBaseline(0.1)], [0.1], 2)
out4 = pem_fit(state0, dat, priors, settings, test_times)
println(out4[3]);println(out4[4])

priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(7.0, 1.0, 7.0, 1.0, 100.0, 3.1), [RandomWalk()], [0.1], 2)
out5 = pem_fit(state0, dat, priors, settings, test_times)
println(out5[3]);println(out5[4])
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(7.0, 1.0, 7.0, 1.0, 100.0, 3.1), [GaussLangevin(log(0.29),0.4)], [0.1], 2)
out6 = pem_fit(state0, dat, priors, settings, test_times)
println(out6[3]);println(out6[4])
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(7.0, 1.0, 7.0, 1.0, 100.0, 3.1), [GammaLangevin(2,7)], [0.1], 2)
out7 = pem_fit(state0, dat, priors, settings, test_times)
println(out7[3]);println(out7[4])
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(7.0, 1.0, 7.0, 1.0, 100.0, 3.1), [GompertzBaseline(0.1)], [0.1], 2)
out8 = pem_fit(state0, dat, priors, settings, test_times)
println(out8[3]);println(out8[4])

m_obs = []
m_obsq1 = []
m_obsq2 = []
m_total = []
m_totalq1 = []
m_totalq2 = []
breaks_extrap = collect(3.12:0.02:15)
grid = sort(unique(out1[1]["Sk_s_loc"][cumsum(out1[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out1[1]["Sk_θ"], dims = 2), out1[1]["Sk_s_loc"], grid)
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 100.0, 3.1), [RandomWalk()], [0.1], 2)
extrap1 = barker_extrapolation(out1[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.01)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df1 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
push!(m_obs, mean(get_meansurv(test_smp[1,:,:], grid, [1])))
push!(m_obsq1, quantile(get_meansurv(test_smp[1,:,:], grid, [1]), 0.025))
push!(m_obsq2, quantile(get_meansurv(test_smp[1,:,:], grid, [1]), 0.975))
push!(m_total, mean(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1])))
push!(m_totalq1, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.025))
push!(m_totalq2, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.975))

grid = sort(unique(out2[1]["Sk_s_loc"][cumsum(out2[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 100.0, 3.1), [GaussLangevin(log(0.29),0.4)], [0.1], 2)
test_smp = cts_transform(cumsum(out2[1]["Sk_θ"], dims = 2), out2[1]["Sk_s_loc"], grid)
extrap1 = barker_extrapolation(out2[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.01)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df2 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
push!(m_obs, mean(get_meansurv(test_smp[1,:,:], grid, [1])))
push!(m_obsq1, quantile(get_meansurv(test_smp[1,:,:], grid, [1]), 0.025))
push!(m_obsq2, quantile(get_meansurv(test_smp[1,:,:], grid, [1]), 0.975))
push!(m_total, mean(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1])))
push!(m_totalq1, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.025))
push!(m_totalq2, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.975))

grid = sort(unique(out3[1]["Sk_s_loc"][cumsum(out3[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out3[1]["Sk_θ"], dims = 2), out3[1]["Sk_s_loc"], grid)
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 100.0, 3.1), [GammaLangevin(2,7)], [0.1], 2)
extrap1 = barker_extrapolation(out3[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.01)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df3 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
push!(m_obs, mean(get_meansurv(test_smp[1,:,:], grid, [1])))
push!(m_obsq1, quantile(get_meansurv(test_smp[1,:,:], grid, [1]), 0.025))
push!(m_obsq2, quantile(get_meansurv(test_smp[1,:,:], grid, [1]), 0.975))
push!(m_total, mean(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1])))
push!(m_totalq1, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.025))
push!(m_totalq2, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.975))

grid = sort(unique(out4[1]["Sk_s_loc"][cumsum(out4[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out4[1]["Sk_θ"], dims = 2), out4[1]["Sk_s_loc"], grid)
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 100.0, 3.1), [GompertzBaseline(0.1)], [0.1], 2)
extrap1 = barker_extrapolation(out4[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.01)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df4 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
push!(m_obs, mean(get_meansurv(test_smp[1,:,:], grid, [1])))
push!(m_obsq1, quantile(get_meansurv(test_smp[1,:,:], grid, [1]), 0.025))
push!(m_obsq2, quantile(get_meansurv(test_smp[1,:,:], grid, [1]), 0.975))
push!(m_total, mean(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1])))
push!(m_totalq1, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.025))
push!(m_totalq2, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.975))

grid = sort(unique(out5[1]["Sk_s_loc"][cumsum(out5[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out5[1]["Sk_θ"], dims = 2), out5[1]["Sk_s_loc"], grid)
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(7.0, 1.0, 7.0, 1.0, 100.0, 3.1), [RandomWalk()], [0.1], 2)
extrap1 = barker_extrapolation(out5[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.01)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df5 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
push!(m_obs, mean(get_meansurv(test_smp[1,:,:], grid, [1])))
push!(m_obsq1, quantile(get_meansurv(test_smp[1,:,:], grid, [1]), 0.025))
push!(m_obsq2, quantile(get_meansurv(test_smp[1,:,:], grid, [1]), 0.975))
push!(m_total, mean(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1])))
push!(m_totalq1, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.025))
push!(m_totalq2, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.975))

grid = sort(unique(out6[1]["Sk_s_loc"][cumsum(out6[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out6[1]["Sk_θ"], dims = 2), out6[1]["Sk_s_loc"], grid)
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(7.0, 1.0, 7.0, 1.0, 100.0, 3.1), [GaussLangevin(log(0.29),0.4)], [0.1], 2)
extrap1 = barker_extrapolation(out6[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.01)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df6 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
push!(m_obs, mean(get_meansurv(test_smp[1,:,:], grid, [1])))
push!(m_obsq1, quantile(get_meansurv(test_smp[1,:,:], grid, [1]), 0.025))
push!(m_obsq2, quantile(get_meansurv(test_smp[1,:,:], grid, [1]), 0.975))
push!(m_total, mean(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1])))
push!(m_totalq1, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.025))
push!(m_totalq2, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.975))

grid = sort(unique(out7[1]["Sk_s_loc"][cumsum(out7[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out7[1]["Sk_θ"], dims = 2), out7[1]["Sk_s_loc"], grid)
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(7.0, 1.0, 7.0, 1.0, 100.0, 3.1), [GammaLangevin(2,7)], [0.1], 2)
extrap1 = barker_extrapolation(out7[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.01)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df7 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
push!(m_obs, mean(get_meansurv(test_smp[1,:,:], grid, [1])))
push!(m_obsq1, quantile(get_meansurv(test_smp[1,:,:], grid, [1]), 0.025))
push!(m_obsq2, quantile(get_meansurv(test_smp[1,:,:], grid, [1]), 0.975))
push!(m_total, mean(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1])))
push!(m_totalq1, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.025))
push!(m_totalq2, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.975))

grid = sort(unique(out8[1]["Sk_s_loc"][cumsum(out8[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out8[1]["Sk_θ"], dims = 2), out8[1]["Sk_s_loc"], grid)
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(7.0, 1.0, 7.0, 1.0, 100.0, 3.1), [GompertzBaseline(0.1)], [0.1], 2)
extrap1 = barker_extrapolation(out8[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.01)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df8 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
push!(m_obs, mean(get_meansurv(test_smp[1,:,:], grid, [1])))
push!(m_obsq1, quantile(get_meansurv(exp.(test_smp[1,:,:]), grid, [1]), 0.025))
push!(m_obsq2, quantile(get_meansurv(test_smp[1,:,:], grid, [1]), 0.975))
push!(m_total, mean(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1])))
push!(m_totalq1, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.025))
push!(m_totalq2, quantile(get_meansurv(log.(s1), vcat(grid,breaks_extrap), [1]), 0.975))

df_meanS = DataFrame(m_obs = m_obs, m_obsq1 = m_obsq1, m_obsq2 = m_obsq2, m_total = m_total, m_totalq1 = m_totalq1, m_totalq2 = m_totalq2)

round.(df_meanS, digits = 2)

CSV.write(datadir("ColonModels","MeanSurvival.csv"),df_meanS)
CSV.write(datadir("ColonModels","RW_Pois.csv"),df1)
CSV.write(datadir("ColonModels","Gauss_Pois.csv"),df2)
CSV.write(datadir("ColonModels","Gamma_Pois.csv"),df3)
CSV.write(datadir("ColonModels","Gomp_Pois.csv"),df4)
CSV.write(datadir("ColonModels","RW_NB.csv"),df5)
CSV.write(datadir("ColonModels","Gauss_NB.csv"),df6)
CSV.write(datadir("ColonModels","Gamma_NB.csv"),df7)
CSV.write(datadir("ColonModels","Gomp_NB.csv"),df8)

df1 = CSV.read(datadir("ColonModels","RW_Pois.csv"),DataFrame)
df2 = CSV.read(datadir("ColonModels","Gauss_Pois.csv"),DataFrame)
df3 = CSV.read(datadir("ColonModels","Gamma_Pois.csv"),DataFrame)
df4 = CSV.read(datadir("ColonModels","Gomp_Pois.csv"),DataFrame)
df5 = CSV.read(datadir("ColonModels","RW_NB.csv"),DataFrame)
df6 = CSV.read(datadir("ColonModels","Gauss_NB.csv"),DataFrame)
df7 = CSV.read(datadir("ColonModels","Gamma_NB.csv"),DataFrame)
df8 = CSV.read(datadir("ColonModels","Gomp_NB.csv"),DataFrame)

R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "RW")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df2)
dat2 = cbind(dat2, "Gaussian")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df3)
dat3 = cbind(dat3, "Gamma")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df4)
dat4 = cbind(dat4, "Gompertz")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_1 <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
dat1 = data.frame($df5)
dat1 = cbind(dat1, "RW")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df6)
dat2 = cbind(dat2, "Gaussian")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df7)
dat3 = cbind(dat3, "Gamma")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df8)
dat4 = cbind(dat4, "Gompertz")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_2 <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
p1 <- dat_1 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) 

p2 <- dat_2 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7,2)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3)

p3 <- dat_1 %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = log(value), col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + geom_hline(yintercept = log(0.5)) + geom_hline(yintercept = 0.1)#+ ylim(0,2.0) + 

p4 <- dat_2 %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = log(value), col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7,2)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + geom_hline(yintercept = log(0.5))+ geom_hline(yintercept = 0.1)#+ ylim(0,2.0) 
plot_grid(p1,p2,p3,p4, nrow = 2)
#ggsave($plotsdir("CovariateColon.pdf"), width = 8, height = 6)
"""

## TODO - Run comparators and compare mean survival/hazards

df1 = CSV.read(datadir("ColonSmps","spline.csv"), DataFrame)
df2 = CSV.read(datadir("ColonSmps","spline_ext.csv"), DataFrame)
df3 = CSV.read(datadir("ColonSmps","DSM.csv"), DataFrame)
df4 = CSV.read(datadir("ColonSmps","PW.csv"), DataFrame)

R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "Spline - no external")
colnames(dat1) <- c("Col","Time","Mean","LCI","UCI","Model") 
dat2 = data.frame($df2)
dat2 = cbind(dat2, "Spline - external")
colnames(dat2) <- c("Col","Time","Mean","LCI","UCI","Model") 
dat2 = rbind(dat1, dat2)
dat3 = data.frame($df3)
colnames(dat3) <- c("Col", "Time", "Model", "Mean")  
dat3 = bind_rows(dat2,dat3)
dat3 <- dat3[,c(1,2,3,5,6)]
dat4 = data.frame($df4)
dat4 = cbind(dat4, "Independent PWEXP")
colnames(dat4) <- c("Col","Time","Mean","UCI","LCI","Model") 
dat_2 <- bind_rows(dat4,dat3)
"""

R"""
p1 <- dat_2 %>%
    subset(Time < 3.1) %>%
    pivot_longer(cols = Mean:LCI) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7,1,2)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) 
p2 <- dat_2 %>%
    pivot_longer(cols = Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7,1,2)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,2) + xlim(0,15)
plot_grid(p1,p2)
"""

Random.seed!(2352)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = vcat(0.01,collect(0.26:0.25:3.01))
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 10_000
nsmp = 10
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
test_times = collect(0.2:0.2:3.0)

priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 100.0, 3.1), [GammaLangevin(4,14,5)], [0.1], 2)
out1 = pem_fit(state0, dat, priors, settings, test_times)
println(out1[3]);println(out1[4])
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(7.0, 1.0, 7.0, 1.0, 100.0, 3.1), [GammaLangevin(4,14,5)], [0.1], 2)
out2 = pem_fit(state0, dat, priors, settings, test_times)
println(out2[3]);println(out2[4])

breaks_extrap = collect(3.12:0.02:15)
grid = sort(unique(out1[1]["Sk_s_loc"][cumsum(out1[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out1[1]["Sk_θ"], dims = 2), out1[1]["Sk_s_loc"], grid)
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 100.0, 3.1), [GammaLangevin(4,14,10)], [0.1], 2)
extrap1 = barker_extrapolation(out1[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.01)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df1 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

breaks_extrap = collect(3.12:0.02:15)
grid = sort(unique(out2[1]["Sk_s_loc"][cumsum(out2[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out2[1]["Sk_θ"], dims = 2), out2[1]["Sk_s_loc"], grid)
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 100.0, 3.1), [GammaLangevin(4,14,10)], [0.1], 2)
extrap1 = barker_extrapolation(out2[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.01)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df2 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)


df3 = CSV.read(datadir("ColonModels","Gamma_Pois.csv"),DataFrame)
df4 = CSV.read(datadir("ColonModels","Gamma_NB.csv"),DataFrame)


R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "Gamma (Pois) - waning")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df2)
dat2 = cbind(dat2, "Gamma (NB) - waning")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df3)
dat3 = cbind(dat3, "Gamma (Pois) - non-waning")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df4)
dat4 = cbind(dat4, "Gamma (NB) - non-waning")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_1 <- rbind(dat1, dat2, dat3, dat4)
"""


R"""
p1 <- dat_1 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) 

p2 <- dat_1 %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)")+ ylim(0,2.0)
plot_grid(p1,p2, nrow = 1)
#ggsave($plotsdir("CovariateColon.pdf"), width = 8, height = 6)
"""
