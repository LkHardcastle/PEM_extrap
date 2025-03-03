using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random, Optim, Roots, SpecialFunctions
using Plots, CSV, DataFrames, RCall, Interpolations, MCMCDiagnosticTools

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

########### Extrapolation ############

Random.seed!(12515)
n = 0
y = rand(Exponential(1.0),n)
breaks = collect(1:1:300)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
nits = 100_000
nsmp = 20

Random.seed!(23462)
test_times = [10.0]
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 0.0, 0.1, false, true, 0.01, 10.0)
priors1 = BasicPrior(1.0, FixedV([0.2]), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 300.0, 3.1), [RandomWalk()], [], 2)
#@time out1 = pem_fit(state0, dat, priors1, settings, test_times)
priors2 = BasicPrior(1.0, FixedV([0.2]), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 300.0, 3.1), [GaussLangevin(2.0,1.0)], [], 1)
#@time out2 = pem_fit(state0, dat, priors2, settings, test_times)
priors3 =  BasicPrior(1.0, FixedV([0.2]), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 300.0, 3.1), [GompertzBaseline(0.1)], [], 1)
#@time out3 = pem_fit(state0, dat, priors3, settings, test_times)

s1 = view(exp.(cumsum(out1[1]["Sk_θ"], dims = 2)), 1, :, :)
df1 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

s1 = view(exp.(cumsum(out2[1]["Sk_θ"], dims = 2)), 1, :, :)
df2 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

s1 = view(exp.(cumsum(out3[1]["Sk_θ"], dims = 2)), 1, :, :)
df3 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "Random Walk")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df2)
dat2 = cbind(dat2, "Log-Normal stationary")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat3 = data.frame($df3)
dat3 = cbind(dat3, "Gompertz dynamics")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat_diffusion <- rbind(dat1, dat2, dat3)
"""

R"""
p1 <- dat_diffusion %>%
    pivot_longer(c(Mean, Q1, Q4)) %>%
    subset(Model == "Random Walk") %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 1), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6)]) +
    scale_linetype_manual(values = c("solid","dotdash","dotdash")) + ylab("h(y)") + xlab("Time (arbitrary units)") + ylim(0,17)
p2 <- dat_diffusion %>%
    pivot_longer(c(Mean, Q1, Q4)) %>%
    subset(Model == "Log-Normal stationary") %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 1), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(7)]) +
    scale_linetype_manual(values = c("solid","dotdash","dotdash")) + ylab("h(y)") + xlab("Time (arbitrary units)") + ylim(0,17)
p3 <- dat_diffusion %>%
    pivot_longer(c(Mean, Q1, Q4)) %>%
    subset(Model == "Gompertz dynamics") %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 1), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(4)]) +
    scale_linetype_manual(values = c("solid","dotdash","dotdash")) + ylab("h(y)") + xlab("Time (arbitrary units)") + ylim(0,17)
plot_grid(p1,p2,p3,ncol = 3)
#ggsave($plotsdir("Talks","Diffusions.pdf"), width = 14, height = 6)
"""


########### Discretisation ###########


x = -2:0.005:7
y1 = (1 .+ tanh.(x.*3.0)).*pdf.(Normal(0,1), x)
y2 = pdf.(Normal(3,1), x)

df = DataFrame(x = x, Barker = y1, EM = y2)

R"""
$df %>%
    pivot_longer(Barker:EM, names_to = "Method", values_to = "y") %>%
    ggplot(aes(x = x, y = y, col = Method)) + geom_line(size = 0.8) + theme_classic() +
    scale_colour_manual(values = cbPalette[6:7]) + xlab(expression(theta)) + ylab(expression("f("*theta*")")) + theme(legend.title = element_blank(), legend.position = "bottom", text = element_text(size = 20)) + 
    geom_vline(xintercept = 0, linetype = "dashed")
    ggsave($plotsdir("Paper","Discretisation.pdf"), width = 14, height = 5)
"""

########### Split-Merge figure #########

Random.seed!(3452)
n = 0
y = rand(Exponential(1.0),n)
breaks = collect(0.5:0.5:1.0)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
nits = 2_000
nsmp = 1_000
test_times = collect(0.05:0.05:2.95)
priors1 = BasicPrior(1.0, FixedV([0.5]), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.1), [RandomWalk()], [], 1)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 0.0, 0.1, false, true, 0.01, 1.0)
out = pem_fit(state0, dat, priors1, settings, test_times)

t = vec(out[1]["Sk_t"])
y1 = vec(out[1]["Sk_x"][1,1,:])
y2 = vec(out[1]["Sk_x"][1,2,:])
v = vec(out[1]["Sk_v"][1,2,:])
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
ggsave($plotsdir("Paper","SM.pdf"), width = 14, height = 6)
"""

########### Efficiency vs reversible jump & Paramerisation experiment #########

df1 = CSV.read(datadir("RJExp1.csv"), DataFrame)
df2 = CSV.read(datadir("RJExp2.csv"), DataFrame)
df3 = CSV.read(datadir("RJExp3.csv"), DataFrame)
df1[!,"Exp"] .= "Prior"
df2[!,"Exp"] .= "Changepoint"
df3[!,"Exp"] .= "Colon data" 
df_rj = vcat(df1,df2,df3)

df1 = CSV.read(datadir("ParamExp1.csv"), DataFrame)
df2 = CSV.read(datadir("ParamExp2.csv"), DataFrame)

df1[!, "Exp"] .= "Diffusion 1"
df2[!, "Exp"] .= "Diffusion 2"

df_param = vcat(df1, df2)

R"""
dat = data.frame($df_rj)
dat = dat %>%
    pivot_longer(J:h3, names_to = "Param", values_to = "Mean_est") %>%
    mutate(Exp = factor(Exp, c("Prior", "Changepoint", "Colon data")))

param_names <- c(
    `J` = "J",
    `h1` = "h(0.5)",
    `h2` = "h(1.5)",
    `h3` = "h(2.5)"
    )

p1 <- dat %>%
    subset(Exp == "Changepoint") %>%
    ggplot(aes(x = as.factor(Tuning), y = 0.5*Mean_est, col = Sampler)) + geom_boxplot() +
    theme_classic() + facet_wrap(Param ~ ., scales = "free", nrow = 1, labeller = labeller(Param = param_names)) +
    theme(axis.text.x = element_text(angle = 45, hjust=1), legend.position = "bottom", text = element_text(size = 20)) + ylab(expression("E["*theta*"]")) + xlab("Step size")
p2 <- $df_param %>%
    pivot_longer("0.001":"0.5", names_to  = "step_size") %>%
    ggplot(aes(x = step_size, y = value, col = Method)) + geom_boxplot() +
    theme_classic() + facet_wrap(Exp ~ ., scales = "free", nrow = 1, labeller = labeller(Param = param_names)) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[6:7]) + geom_hline(yintercept = 0.5, linetype = "dotted") + xlab("Step size") + ylab(expression("P["*theta*"=0]"))
plot_grid(p1,p2, nrow = 2, labels = c("A", "B"), label_size = 30)
ggsave($plotsdir("Paper","Eff_exp.pdf"), width = 14, height = 12)
"""   

#### Colon hazards ####
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
    theme(text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) + guides(colour = guide_legend(nrow = 1))

leg <- get_legend(p1)

p1 <- dat_1 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) 

p2 <- dat_2 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7,2)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3)

p3 <- dat_1 %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,2.0)

p4 <- dat_2 %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7,2)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,2.0) 
p5 <- plot_grid(p1,p2,p3,p4, nrow = 2)
p6 <- plot_grid(p5, leg, nrow = 2, rel_heights = c(0.9, 0.1))
#ggsave($plotsdir("Paper", "ColonModels.pdf"), width = 14, height = 12)
"""
