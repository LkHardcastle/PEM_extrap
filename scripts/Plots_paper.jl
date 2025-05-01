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
breaks = collect(1:1:3)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
nits = 10_000
nsmp = 20

Random.seed!(23462)
test_times = [10.0]
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 0.0, 0.1, false, true, 0.01, 10.0)
priors1 = BasicPrior(1.0, FixedV(0.05), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 300.0, 3.1), [RandomWalk()], [], 2)
@time out1 = pem_fit(state0, dat, priors1, settings, test_times, 1_000)
priors2 = BasicPrior(1.0, FixedV(0.05), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 300.0, 3.1), [GaussLangevin(t -> 2.0,t -> 1.0)], [], 1)
@time out2 = pem_fit(state0, dat, priors2, settings, test_times, 1_000)
priors3 =  BasicPrior(1.0, FixedV(0.05), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 300.0, 3.1), [GompertzBaseline(0.5)], [], 1)
@time out3 = pem_fit(state0, dat, priors3, settings, test_times, 1_000)

breaks_extrap = collect(4.0:500.0)
extrap1 = barker_extrapolation(out2[1], priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.05)
extrap2 = barker_extrapolation(out2[1], priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.05)
extrap3 = barker_extrapolation(out2[1], priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.05)

s1 = exp.(extrap1)
df1 = DataFrame(hcat(breaks_extrap./100, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

s1 = exp.(extrap2)
df2 = DataFrame(hcat(breaks_extrap./100, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

s1 = exp.(extrap3)
df3 = DataFrame(hcat(breaks_extrap./100, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

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
ggsave($plotsdir("Paper","Diffusions.pdf"), width = 14, height = 6)
"""


########### Discretisation ###########


x = -2:0.005:7
y1 = (1 .+ tanh.(x.*1)).*pdf.(Normal(0,1), x)
y2 = pdf.(Normal(1,1), x)
df1 = DataFrame(x = x, Barker = y1, EM = y2, ind = "= 1")

x = -2:0.005:7
y1 = (1 .+ tanh.(x.*2)).*pdf.(Normal(0,1), x)
y2 = pdf.(Normal(2,1), x)
df2 = DataFrame(x = x, Barker = y1, EM = y2, ind = "= 2")

x = -2:0.005:7
y1 = (1 .+ tanh.(x.*3)).*pdf.(Normal(0,1), x)
y2 = pdf.(Normal(3,1), x)
df3 = DataFrame(x = x, Barker = y1, EM = y2, ind = "= 3")

x = -2:0.005:7
y1 = (1 .+ tanh.(x.*4)).*pdf.(Normal(0,1), x)
y2 = pdf.(Normal(4,1), x)
df4 = DataFrame(x = x, Barker = y1, EM = y2, ind = "= 4")

df = vcat(df1, df2, df3, df4)

R"""
$df %>%
    pivot_longer(Barker:EM, names_to = "Method", values_to = "y") %>%
    mutate(Method = case_when(
        Method == "Barker" ~ "Skew-Symmetric",
        Method == "EM" ~ "Euler-Maruyama"
    )) %>%
    ggplot(aes(x = x, y = y, col = ind, linetype = Method)) + geom_line(size = 0.8) + theme_classic() +
    scale_colour_manual(values = cbPalette[c(2,4,6,7)]) + xlab(expression(theta)) + ylab(expression("f("*theta*")")) + 
    theme(legend.title = element_blank(), legend.position = "bottom", text = element_text(size = 20)) + 
    geom_vline(xintercept = 0, linetype = "dashed") + scale_linetype_manual(values = c("dashed", "solid")) 
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
priors1 = BasicPrior(1.0, FixedV(0.5), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.1), [RandomWalk()], [], 1)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 0.0, 0.1, false, true, 0.01, 1.0)
out = pem_fit(state0, dat, priors1, settings, test_times, 100)

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
    theme_classic() + scale_colour_manual(values = cbPalette[7:6]) + ylab(expression(alpha*"-space")) + xlab("Sampler time (arbitrary units)") +
    theme(legend.position = "none", text = element_text(size = 20))
p2 <- $df %>%
    pivot_longer(y1:y2) %>%
    ggplot(aes(x = t, y = value, col = name)) + geom_line() +
    theme_classic() + scale_colour_manual(values = cbPalette[6:7]) + ylab(expression(theta*"-space")) + xlab("Sampler time (arbitrary units)") +
    theme(legend.position = "none", text = element_text(size = 20))
plot_grid(p1,p2)
ggsave($plotsdir("Paper","SM.pdf"), width = 14, height = 6)
"""

########### Paramerisation experiment #########

df1 = CSV.read(datadir("ParamExp1.csv"), DataFrame)
df2 = CSV.read(datadir("ParamExp2.csv"), DataFrame)

df1[!, "Exp"] .= "Diffusion 1"
df2[!, "Exp"] .= "Diffusion 2"

df_param = vcat(df1, df2)

R"""
p1 <- $df_param %>%
    mutate(Method = case_when(
       Method == "Barker" ~ "Skew-Symmetric",
       Method == "Euler-Maruyama" ~ "Euler-Maruyama"
    )) %>%
    pivot_longer("0.001":"0.5", names_to  = "step_size") %>%
    ggplot(aes(x = step_size, y = value, col = Method)) + geom_boxplot() +
    theme_classic() + facet_wrap(Exp ~ ., scales = "free", nrow = 1) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[6:7]) + geom_hline(yintercept = 0.5, linetype = "dotted") + xlab("Step size") + ylab(expression("P["*theta*"=0]"))
p1
ggsave($plotsdir("Paper", "Param_Eff.pdf"), width = 14, height = 6)
"""   

########### Reversible jump experiment ###########

df1 = CSV.read(datadir("RJexp", "trace_plots_main.csv"), DataFrame)
df2 = CSV.read(datadir("RJexp", "hazards.csv"), DataFrame)

R"""
p1 <- $df1 %>%
    mutate(Iteration = 1:n()) %>%
    subset(Iteration < 10000) %>%
    pivot_longer(PDMP:RJ, names_to = "Method", values_to = "log(h(1.2))") %>%
    ggplot(aes(x = Iteration, y = `log(h(1.2))`, col = Method)) + geom_path() + theme_classic() + 
    scale_colour_manual(values = cbPalette[6:7]) + theme(legend.position = "bottom", text = element_text(size = 20)) 
p2 <- $df2 %>% 
    pivot_longer(median:UCI, values_to = "h(y)", names_to = "Quantity") %>%
    ggplot(aes(x = Time, y = `h(y)`, col = Method, linetype = Quantity)) + geom_line() + theme_classic() + guides(linetype = FALSE) + 
    scale_colour_manual(values = cbPalette[6:7]) + scale_linetype_manual(values = c("dashed", "solid", "dashed")) + 
    theme(legend.position = "bottom", text = element_text(size = 20))
plot_grid(p2, p1, nrow = 1)
ggsave($plotsdir("Paper", "RJ.pdf"), width = 14, height = 6)
"""

df2 = CSV.read(datadir("RJexp", "hazards_supp.csv"), DataFrame)

R"""
p2 <- $df2 %>% 
    pivot_longer(median:UCI, values_to = "h(y)", names_to = "Quantity") %>%
    ggplot(aes(x = Time, y = `h(y)`, linetype = Quantity)) + geom_line(col = cbPalette[6]) + theme_classic() + guides(linetype = FALSE) + 
    scale_linetype_manual(values = c("dashed", "solid", "dashed")) + xlim(0.1,NA) +
    theme(legend.position = "bottom", text = element_text(size = 20))
p2
ggsave($plotsdir("Paper", "RJ_supp.pdf"), width = 14, height = 6)
"""

#### Colon LOOIC ####

df_LOOIC = CSV.read(datadir("ColonModels","LOOIC.csv"),DataFrame)

R"""
$df_LOOIC %>%
    ggplot(aes(x = Gamma, y = LOOIC)) + geom_point(col = cbPalette[6], size = 2) + theme_classic() + 
    theme(text = element_text(size = 20))
ggsave($plotsdir("Paper", "ColonLOOIC.pdf"), width = 14, height = 6)
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
dat1 = cbind(dat1, "Random Walk")
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
dat1 = cbind(dat1, "Random Walk")
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
    subset(Time > 0.01) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) + guides(colour = guide_legend(nrow = 1))

leg <- get_legend(p1)

p1 <- dat_1 %>%
    subset(Time < 3.1) %>%
    subset(Time > 0.01) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(y)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) 

p2 <- dat_2 %>%
    subset(Time < 3.1) %>%
    subset(Time > 0.01) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7,2)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(y)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3)

p3 <- dat_1 %>%
    subset(Time > 0.01) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(y)") + xlab("Time (years)") + ylim(0,2.0)

p4 <- dat_2 %>%
    subset(Time > 0.01) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7,2)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(y)") + xlab("Time (years)") + ylim(0,2.0) 
p5 <- plot_grid(p1,p2,p3,p4, nrow = 2)
p6 <- plot_grid(p5, leg, nrow = 2, rel_heights = c(0.9, 0.1))
ggsave($plotsdir("Paper", "ColonModels.pdf"), width = 14, height = 8)
"""


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
dat1 = cbind(dat1, "Gamma (fixed)")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df3)
dat2 = cbind(dat2, "Gamma (converging)")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df5)
dat3 = cbind(dat3, "Gompertz (baseline)")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df7)
dat4 = cbind(dat4, "Gompertz (centred)")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_1 <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
dat1 = data.frame($df2)
dat1 = cbind(dat1, "Gamma (fixed)")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df4)
dat2 = cbind(dat2, "Gamma (converging)")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df6)
dat3 = cbind(dat3, "Gompertz (baseline)")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df8)
dat4 = cbind(dat4, "Gompertz (centred)")
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
    subset(Time < 5.0) %>%
    subset(Time > 0.01) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("S(y)") + xlab("Time (years)") + ylim(0,1) + xlim(0,5) + geom_vline(xintercept = 4, linetype = "dotted") 

p2 <- dat_2 %>%
    subset(Time < 5) %>%
    subset(Time > 0.01) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7,2)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("S(y)") + xlab("Time (years)") + ylim(0,1) + xlim(0,5)+ geom_vline(xintercept = 4, linetype = "dotted") 

p3 <- dat_1 %>%
    subset(Time > 0.01) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("S(y)") + xlab("Time (years)") + ylim(0,1.0)+ geom_vline(xintercept = 4, linetype = "dotted") 

p4 <- dat_2 %>%
    subset(Time > 0.01) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7,2)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("S(y)") + xlab("Time (years)") + ylim(0,1.0) + geom_vline(xintercept = 4, linetype = "dotted") 
p5 <- plot_grid(p1,p2,p3,p4, nrow = 2)
p6 <- plot_grid(p5, leg, nrow = 2, rel_heights = c(0.9, 0.1))
ggsave($plotsdir("Paper", "TA174Survival.pdf"), width = 14, height = 8)
"""


df1_ = CSV.read(datadir("TA174Models","CovCtrl.csv"),DataFrame)
df2_ = CSV.read(datadir("TA174Models","CovTrt.csv"),DataFrame)
df3_ = CSV.read(datadir("TA174Models","CovWane.csv"),DataFrame)

R"""
dat1 = data.frame($df1_)
dat1$Arm = "Control"
dat2 = data.frame($df2_)
dat2$Arm = "Treatment"
dat1 = rbind(dat1, dat2)
dat3 = data.frame($df3_)
dat3$Arm = "Treatment (Waning)"
dat3 = rbind(dat1,dat3)
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Arm") 
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Arm") 
p1 <- dat1 %>%
    subset(Time < 4.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Arm, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(y)") + xlab("Time (years)") + ylim(0,0.3) + xlim(0.01,4) 
p2 <- dat3 %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    subset(Time > 4.0) %>%
    ggplot(aes(x = Time, y = log(value), col = Arm, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,4)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("log(h(y))") + xlab("Time (years)") + xlim(4.01,NA) #+ ylim(0,2)
plot_grid(p1,p2)
ggsave($plotsdir("Paper", "TA174covariate.pdf"), width = 14, height = 6)
"""

#### TA174 LOOIC

df_LOOIC1 = CSV.read(datadir("TA174Models","LOOIC_trt.csv"),DataFrame)
df_LOOIC2 = CSV.read(datadir("TA174Models","LOOIC_control.csv"),DataFrame)

R"""
p1 <- $df_LOOIC1 %>%
    ggplot(aes(x = Gamma, y = LOOIC)) + geom_point(col = cbPalette[6], size = 2) + theme_classic() + 
    theme(text = element_text(size = 20))
p2 <- $df_LOOIC2 %>%
    ggplot(aes(x = Gamma, y = LOOIC)) + geom_point(col = cbPalette[6], size = 2) + theme_classic() + 
    theme(text = element_text(size = 20))
plot_grid(p1,p2, nrow = 1)
ggsave($plotsdir("Paper", "TA174LOOIC.pdf"), width = 14, height = 6)
"""