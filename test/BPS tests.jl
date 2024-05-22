using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random, Roots, SpecialFunctions
using Plots

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))

Random.seed!(123)
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
state0 = BPS(x0, v0, s0, t0, findall(s0))
priors = BasicPrior(1.0, 1.0, 0.5, 1.0)
nits = 100_000
nsmp = 10000000
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 1.0, false, true)
Random.seed!(123)
@time out1 = pem_sample(state0, dat, priors, settings)
@time out11 = pem_sample(state0, dat, priors, settings)

Random.seed!(123)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
state0 = ECMC(x0, v0, s0, t0, findall(s0))
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 0.1, false, true)
@time out2 = pem_sample(state0, dat, priors, settings)
@time out21 = pem_sample(state0, dat, priors, settings)

Random.seed!(34734)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
state0 = ECMC2(x0, v0, s0, t0, true, findall(s0))
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.0, 0.2, false, true)
priors = BasicPrior(1.0, 1.0, 0.5, 1.0)
@time out3 = pem_sample(state0, dat, priors, settings)
Random.seed!(123)
@time out31 = pem_sample(state0, dat, priors, settings)


x_smp = view(out1["Smp_x"], 1, 2:10, :)
x_ss = mean.(eachrow(x_smp .== 0.0))
x_smp = view(out11["Smp_x"], 1, 2:10, :)
x_ss = mean.(eachrow(x_smp .== 0.0))
x_smp = view(out3["Smp_x"], 1, 2:10, :)
x_ss = mean.(eachrow(x_smp .== 0.0))
x_smp = view(out31["Smp_x"], 1, 2:10, :)
x_ss = mean.(eachrow(x_smp .== 0.0))

norm([0.4258813188821129, 0.3079213709060054])
x_plot = out1["Sk_x"][:,2:3,1:100]
v_plot = out1["Sk_v"][:,:,1:50]
plot(x_plot[1,1,:], x_plot[1,2,:])
plot(out1["Sk_t"][1:50],x_plot[1,1,:])
plot(out1["Sk_t"][1:50],v_plot[1,2,:], linetype=:steppost)
plot!(out1["Sk_t"][1:50],x_plot[1,2,:])

x_plot = out2["Sk_x"][:,1:2,1:100]
plot(x_plot[1,1,:], x_plot[1,2,:])
plot(out2["Sk_t"][1:50],x_plot[1,1,1:50])
plot!(out2["Sk_t"][1:50],x_plot[1,2,1:50])
plot(out2["Sk_t"][1:50],x_plot[1,3,1:50])

x_plot = out31["Sk_x"][:,1:4,1:50]
plot(x_plot[1,1,:], x_plot[1,2,:])
plot(out31["Sk_t"][1:50],x_plot[1,1,1:50])
plot!(out31["Sk_t"][1:50],x_plot[1,2,1:50])
plot!(out31["Sk_t"][1:50],x_plot[1,3,1:50])
plot!(out31["Sk_t"][1:50],x_plot[1,4,1:50])
plot(out3["Sk_t"][1:50],x_plot[1,3,1:50])
out31["Sk_x"][1,1:4,10]


x_smp = vec(out1["Smp_x"][1,2,:])
mean(x_smp .== 0.0)
mean(x_smp)
quantile(x_smp, 0.025)
quantile(x_smp, 0.975)
x_smp = vec(out11["Smp_x"][1,1,:])
mean(x_smp)
quantile(x_smp, 0.025)
quantile(x_smp, 0.975)
norm([0.4395687085512257, 0.5126540905643848, 0.0, 0.7375399202010771])
x_smp = vec(out2["Smp_x"][1,2,:])
mean(x_smp)
quantile(x_smp, 0.025)
quantile(x_smp, 0.975)
x_smp = vec(out21["Smp_x"][1,2,:])
mean(x_smp)
quantile(x_smp, 0.025)
quantile(x_smp, 0.975)

mean(x_smp)
mean(x_smp .== 0.0)
quantile(x_smp[findall(x_smp .!= 0.0)], 0.025)
quantile(x_smp[findall(x_smp .!= 0.0)], 0.975)
x_smp = vec(out31["Smp_x"][1,2,:])
mean(x_smp)
quantile(x_smp, 0.025)
quantile(x_smp, 0.975)

s_out = vec(sum(out3["Smp_s"], dims = 2)) .- 1
histogram(s_out, normalize=:probability)

pdf(Binomial(3,0.42))

out["Eval"]
0.371158^2 +  0.62572^2

0.991133^2  + 0.132874^2
a = [1,2]
a /= 2
a *= 3