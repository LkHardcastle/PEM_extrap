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

df1 = CSV.read(datadir("ColonSmps","RW.csv"), DataFrame)
df2 = CSV.read(datadir("ColonSmps","Gaussian.csv"), DataFrame)
df3 = CSV.read(datadir("ColonSmps","Gamma.csv"), DataFrame)
df4 = CSV.read(datadir("ColonSmps","Gompertz.csv"), DataFrame)

R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "Random Walk")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df2)
dat2 = cbind(dat2, "Log-Normal Langevin")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df3)
dat3 = cbind(dat3, "Gamma Langevin")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df4)
dat4 = cbind(dat4, "Gompertz dynamics")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_diffusion <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
p1 <- dat_diffusion %>%
    subset(Time < 3.1) %>%
    pivot_longer(Mean:UCI,) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3)
"""

df1 = CSV.read(datadir("ColonSmps","RW.csv"), DataFrame)
df2 = CSV.read(datadir("ColonSmps","PW.csv"), DataFrame)
df3 = CSV.read(datadir("ColonSmps","spline.csv"), DataFrame)
df4 = CSV.read(datadir("ColonSmps","DSM.csv"), DataFrame)
df5 = CSV.read(datadir("ColonSmps","spline_ext.csv"), DataFrame)
df6 = CSV.read(datadir("ColonSmps","Gaussian.csv"), DataFrame)

R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "Random Walk")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat1 = data.frame(Time = dat1$Time, hazard = dat1$Mean, Model = dat1$Model)

dat2 = data.frame($df2)
dat2 = cbind(dat2, "Piecewise independent")
colnames(dat2) <- c("ID", "Time","Mean","UCI","LCI","Model") 
dat2 = data.frame(Time = dat2$Time, hazard = dat2$Mean, Model = dat2$Model)

dat3 = data.frame($df3)
dat3 = cbind(dat3, "M-spline")
colnames(dat3) <- c("ID","Time","Mean","UCI","LCI","Model") 
dat3 = data.frame(Time = dat3$Time, hazard = dat3$Mean, Model = dat3$Model)

dat4 = data.frame($df4)
colnames(dat4) <- c("ID","Time", "Model", "Mean") 
dat4 = data.frame(Time = dat4$Time, hazard = dat4$Mean, Model = dat4$Model)

dat5 = data.frame($df5)
dat5 = cbind(dat5, "M-spline - external")
colnames(dat5) <- c("ID","Time","Mean","UCI","LCI","Model") 
dat5 = data.frame(Time = dat5$Time, hazard = dat5$Mean, Model = dat5$Model)

dat6 = data.frame($df6)
dat6 = cbind(dat6, "Gaussian Langevin")
colnames(dat6) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat6 = data.frame(Time = dat6$Time, hazard = dat6$Mean, Model = dat6$Model)

dat_comp <- rbind(dat1, dat2, dat3, dat4, dat5, dat6)
"""

R"""
p2 <- dat_comp %>%
    ggplot(aes(x = Time, y = hazard, col = Model)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[1:8]) +
    ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3)
plot_grid(p1,p2)
#ggsave($plotsdir("WithinColon.pdf"), width = 8, height = 6)
"""

df1 = CSV.read(datadir("ColonSmps","RW.csv"), DataFrame)
df2 = CSV.read(datadir("ColonSmps","Gaussian.csv"), DataFrame)
df3 = CSV.read(datadir("ColonSmps","Gamma.csv"), DataFrame)
df4 = CSV.read(datadir("ColonSmps","Gompertz.csv"), DataFrame)

R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "Random Walk")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df2)
dat2 = cbind(dat2, "Log-Normal Langevin")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df3)
dat3 = cbind(dat3, "Gamma Langevin")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df4)
dat4 = cbind(dat4, "Gompertz dynamics")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_diffusion <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
p3 <- dat_diffusion %>%
    pivot_longer(c(Mean, Q1, Q4)) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("solid","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,15)
"""

df1 = CSV.read(datadir("ColonSmps","Gompertz.csv"), DataFrame)
df2 = CSV.read(datadir("ColonSmps","PW.csv"), DataFrame)
df3 = CSV.read(datadir("ColonSmps","spline.csv"), DataFrame)
df4 = CSV.read(datadir("ColonSmps","DSM.csv"), DataFrame)
df5 = CSV.read(datadir("ColonSmps","spline_ext.csv"), DataFrame)
df6 = CSV.read(datadir("ColonSmps","Gaussian.csv"), DataFrame)

R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "Random Walk")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat1 = data.frame(Time = dat1$Time, hazard = dat1$Mean, Model = dat1$Model)

dat2 = data.frame($df2)
dat2 = cbind(dat2, "Piecewise independent")
colnames(dat2) <- c("ID", "Time","Mean","UCI","LCI","Model") 
dat2 = data.frame(Time = c(dat2$Time,15), hazard = c(dat2$Mean, dat2$Mean[length(dat2$Mean)]), Model = c(dat2$Model,"Piecewise independent"))

dat3 = data.frame($df3)
dat3 = cbind(dat3, "M-spline")
colnames(dat3) <- c("ID","Time","Mean","UCI","LCI","Model") 
dat3 = data.frame(Time = dat3$Time, hazard = dat3$Mean, Model = dat3$Model)

dat4 = data.frame($df4)
colnames(dat4) <- c("ID","Time", "Model", "Mean") 
dat4 = data.frame(Time = dat4$Time, hazard = dat4$Mean, Model = dat4$Model)

dat5 = data.frame($df5)
dat5 = cbind(dat5, "M-spline - external")
colnames(dat5) <- c("ID","Time","Mean","UCI","LCI","Model") 
dat5 = data.frame(Time = dat5$Time, hazard = dat5$Mean, Model = dat5$Model)

dat6 = data.frame($df6)
dat6 = cbind(dat6, "Gaussian Langevin")
colnames(dat6) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat6 = data.frame(Time = dat6$Time, hazard = dat6$Mean, Model = dat6$Model)

dat_comp <- rbind(dat1, dat2, dat3, dat4, dat5, dat6)
"""

R"""
p4 <- dat_comp %>%
    ggplot(aes(x = Time, y = hazard, col = Model)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[1:8]) +
    ylab("h(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,15)
plot_grid(p3,p4)
ggsave($plotsdir("ExtrapColon.pdf"), width = 8, height = 6)
"""
