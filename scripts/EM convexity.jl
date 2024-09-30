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

function EM_convexity(t, θ, α)
    U1 = []
    U2 = []
    for i in eachindex(t)
        push!(U1, -logpdf(Normal(-(0.5^2)*0.5*(α + t[i] - 0.0) , 0.5), θ + t[i]))
        push!(U2, -logpdf(Normal(0.5*0.1^2 , 0.1), θ + t[i]))
    end
    return DataFrame(t = t, U1 = U1, U2 = U2)
end

function Barker_convexity(t, θ, α)
    U1 = []
    U2 = []
    U3 = []
    for i in eachindex(t)
        push!(U1, -logpdf(Normal(0.0 , 0.5), θ + t[i]))
        push!(U2, -log(1 + tanh(-0.5*(α + t[i] - 0.0)*(θ + t[i]))))
        push!(U3, -log(1 + tanh(0.5*(θ + t[i]))))
    end
    return DataFrame(t = t, U1 = U1, U2 = U2, U3 = U3)
end

dat1 = EM_convexity(collect(-5:0.01:5), -1.0, 1.0)
dat2 = Barker_convexity(collect(-5:0.01:5), -1.0, 1.0)
plot(dat1.t, dat1.U1)
plot!(dat2.t, dat2.U1 + dat2.U2)

plot(dat1.t, dat1.U2)
plot!(dat2.t,  dat2.U1 + dat2.U3)