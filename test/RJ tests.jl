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
breaks = collect(0.1:0.1:3.0)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
nits = 10_000
nsmp = 50_000



priors1 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 1.0, Cts(5.0, 100.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, 0.2, 100.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(10.0, 0.2, 100.0, 3.2), [RandomWalk()])
priors4 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, 2.0, 100.0, 3.2), [RandomWalk()])

Random.seed!(9102)
settings = Settings(nits, nsmp, 1_000_000, 5.0, 1.0, 1.0, false, true)
@time out1 = pem_sample(state0, dat, priors1, settings)
Random.seed!(9102)
settings = Settings(nits, nsmp, 1_000_000, 5.0, 1.0, 1.0, false, true)
@time out2 = pem_sample(state0, dat, priors2, settings)
Random.seed!(9102)
state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.1)
@time out3 = pem_sample(state0, dat, priors3, settings)
Random.seed!(9102)
@time out4 = pem_sample(state0, dat, priors4, settings)

out2["Smp_x"]
logpdf(Normal(0,))

histogram(out1["Smp_J"])
histogram(out2["Smp_J"])

mean(out1["Smp_J"])
plot(out1["Smp_J"])
mean(sum(out1["Smp_s"],dims = 2))
plot(sum(out1["Smp_s"],dims = 2)[1,1,:])
mean(sum(out1["Smp_s"],dims = 2)[1,1,:])
mean(out2["Smp_J"])
mean(out3["Smp_J"])
plot!(out2["Smp_J"])
plot!(out3["Smp_J"])
plot(out1["Smp_x"][1,3,:])
plot(out3["Smp_x"][1,3,:])

plot(pdf.(Poisson(15.3/2),1:20))
sum(pdf.(Poisson(15.3/2),1:20).*collect(1:20))
plot(sum(out1["Smp_s"],dims = 2)[1,1,:])
plot(out3["Smp_J"])
mean(sum(out1["Smp_s"],dims = 2))
mean(out2["Smp_J"])
mean(out3["Smp_J"])
mean(out4["Smp_J"])
mean(out5["Smp_J"])
mean(out6["Smp_J"])

plot(sphere_area.(1:30))

plot(log.(sum(out1["Smp_s"],dims = 2)[1,1,:]))
plot!(log.(out2["Smp_J"]))
plot!(log.(out3["Smp_J"]))
plot!(out4["Smp_J"])
plot!(out5["Smp_J"])

plot(out1["Smp_x"][1,2,:])
plot(out2["Smp_x"][1,2,:])

plot(out1["Smp_x"][1,5,:])
plot(out2["Smp_x"][1,5,:])

plot(out3["Smp_x"][1,4,:])

out3["Smp_s_loc"]

sum(out2["Smp_v"][1,:,5_000])
plot(sum(out2["Smp_v"][1,1:out2["Smp_J"][1:end],1:end].^2))

out1["Smp_s_loc"]

histogram(sum(out1["Smp_s"],dims = 2)[1,1, 1_000:end], normalize = :probability)
histogram!(out2["Smp_J"], normalize = :probability)
plot!(collect(2:31),pdf.(Poisson(3.1*2.5),1:30)/(1-pdf(Poisson(3.1*2.5),0)))

println(state_split.J);println(u)
println(state_curr.x);println(state_split.x)
println(-U2 + U1)
println(exp(logpriors));println(exp(prop_terms));println(J);
println(exp(A))
println("---------")

4
-0.1565129508206829
[0.10728479941805574 0.004722428086760912 -0.06510442516431941]
[0.10728479941805574 0.004722428086760912 -0.1565129508206829 0.09140852565636348]
0.33283392721328875
1.9374999999999993
0.7036277976002522
0.4244131815783876
0.8070891360945895

(pdf(Normal(0, priors2.grid.σ), 0.09140852565636348)/pdf(Normal(0, priors2.grid.σ), -0.06510442516431941))*(3.1/3)
exp(0.33283392721328875)*0.7036277976002522
exp(sum(logpdf.(Normal(0, priors2.σ.σ[1]), [ -0.1565129508206829 0.09140852565636348])) - 
sum(logpdf.(Normal(0, priors2.σ.σ[1]), [-0.06510442516431941])))*exp(-logpdf(Normal(0, priors2.grid.σ), -0.1565129508206829) -log(4 - 1)  + log(3.1))
exp(sum(logpdf.(Normal(0, priors2.σ.σ[1]), [ -0.1565129508206829 0.09140852565636348])) - 
sum(logpdf.(Normal(0, priors2.σ.σ[1]), [-0.06510442516431941])))
exp(-logpdf(Normal(0, priors2.grid.σ), -0.1565129508206829) -log(4 - 1)  + log(3.1))

exp(0.33283392721328875)
