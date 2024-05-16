using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random
using Plots

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))

Random.seed!(123)
n = 0
y = rand(Exponential(1.0),n)
breaks = collect(0.5:0.5:1)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = BPS(x0, v0, s0, t0, findall(s0))
priors = BasicPrior(1.0, 1.0)
nits = 1000
nsmp = 5000
settings = Settings(nits, nsmp, 100000, 1.0,0.0, 0.0, false, true)
Random.seed!(123)
@time out1 = pem_sample(state0, dat, priors, settings)

Random.seed!(123)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
state0 = ECMC(x0, v0, s0, t0, findall(s0))
settings = Settings(nits, nsmp, 100000, 1.0,0.0, 0.0, false, true)
@time out2 = pem_sample(state0, dat, priors, settings)

x_plot = out1["Sk_x"][:,:,1:50]
v_plot = out1["Sk_v"][:,:,1:50]
plot(x_plot[1,1,:], x_plot[1,2,:])
plot(out1["Sk_t"][1:50],x_plot[1,1,:])
plot!(out1["Sk_t"][1:50],v_plot[1,1,:], linetype=:steppost)
plot!(out1["Sk_t"][1:50],x_plot[1,2,:])
x_plot = out2["Sk_x"][:,:,1:50]
plot!(x_plot[1,1,:], x_plot[1,2,:])
plot(out2["Sk_t"][1:50],x_plot[1,1,:])
plot!(out2["Sk_t"][1:50],x_plot[1,2,:])
out1["Sk_t"]
out2["Sk_t"]

norm([0.721191,0.385616])
a = [CartesianIndex(1,1),CartesianIndex(1,3)]
b = [1 1 5]
b[a]
vec(b[a])

CartesianIndex{2}[CartesianIndex(1, 1), CartesianIndex(1, 2), CartesianIndex(1, 3), CartesianIndex(1, 4)]


state = BPS([0.0 0.0 0.0 0.0], [0.4922456865251828 0.9809798121241488 0.0799568295050599 1.5491245530427917], Bool[1 1 1 1], 0.0, CartesianIndex{2}[CartesianIndex(1, 1), CartesianIndex(1, 2), CartesianIndex(1, 3), CartesianIndex(1, 4)])
t = 1.22467442854755
dyn = Dynamics(2, 1, Inf, 1, [0.0 0.0 0.0 0.0], [0.0 0.0 0.0 0.0], [0.0 0.0 0.0 0.0], [0.4922456865251828 1.4732254986493316 1.5531823281543915 3.1023068811971832], [0.0 0.0 0.0 0.0], SamplerEval([0.0, 0.0], 0))
priors = BasicPrior(1.0, 1.0)
V = 0.5
f = copy()
out["Sk_v"][:,:,5]

x_plot = out["Sk_x"][:,:,1:500]

plot(x_plot[1,1,:], x_plot[1,2,:])
plot!(scatter!(out["Smp_t"],vec(out["Smp_x"])))
out["Smp_t"]

x_smp = vec(out["Smp_x"][1,1,:])
mean(x_smp)
quantile(x_smp, 0.025)
quantile(x_smp, 0.975)

x_smp = vec(out["Smp_x"][1,2,:])
mean(x_smp)
quantile(x_smp, 0.025)
quantile(x_smp, 0.975)

out["Eval"]