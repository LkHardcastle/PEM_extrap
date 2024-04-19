Random.seed!(123)
n = 100
y = rand(Exponential(1.0),n)
maximum(y)
#breaks = collect(0.05:0.05:(maximum(y) + 0.1))
#breaks = collect(0.05:0.05:1.0)
breaks = collect(0.25:0.25:5.5)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
v_abs = vcat(1.0,collect(0.05:0.05:1.05))
x0, v0, s0 = init_params(p, dat, v_abs)
t0 = 0.0
priors = FixedPrior(fill(0.2, size(x0)), 0.5, 1.0, 0.0, 1.0)
nits = 1_000_000
nsmp = 50_000
settings = Settings(nits, nsmp, 0.9, 0.5, 0.0, v0, false)
Random.seed!(23653)
out1 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

Random.seed!(3546232)
priors = FixedPrior(fill(0.5, size(x0)), 0.5, 1.0, 0.0, 1.0)
out2 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

Random.seed!(2222)
priors = FixedPrior(fill(0.8, size(x0)), 0.5, 1.0, 0.0, 1.0)
out3 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

smps1 = out1["Smp_x"]
smps2 = out2["Smp_x"]
smps3 = out3["Smp_x"]

smps1 = out1["Smp_trans"]
plot(vcat(0,breaks), vcat(mean(exp.(smps1), dims = 2), mean(exp.(smps1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.025),quantile.(eachrow(exp.(smps1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.975),quantile.(eachrow(exp.(smps1)), 0.975)[end]),linetype=:steppost)
hline!([1,1])

smps1 = out2["Smp_trans"]
plot(vcat(0,breaks), vcat(mean(exp.(smps1), dims = 2), mean(exp.(smps1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.025),quantile.(eachrow(exp.(smps1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.975),quantile.(eachrow(exp.(smps1)), 0.975)[end]),linetype=:steppost)
hline!([1,1])

smps1 = out3["Smp_trans"]
plot(vcat(0,breaks), vcat(mean(exp.(smps1), dims = 2), mean(exp.(smps1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.025),quantile.(eachrow(exp.(smps1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.975),quantile.(eachrow(exp.(smps1)), 0.975)[end]),linetype=:steppost)
hline!([1,1])

histogram(y)

smps1 = out1["Smp_trans"]
plot(vcat(0,breaks), vcat(mean(smps1, dims = 2), mean(smps1, dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(smps1), 0.025),quantile.(eachrow(smps1), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(smps1), 0.975),quantile.(eachrow(smps1), 0.975)[end]),linetype=:steppost)


plot(vec(smps1[22,:]))
plot(exp.(vec(smps1[22,:])))

plot(vec(sum(out1["Smp_s"], dims = 1)))
plot(vec(sum(out2["Smp_s"], dims = 1)))
plot(vec(sum(out3["Smp_s"], dims = 1)))

n_plot = 10_000
n_start = 5000
plot(out1["t"][n_start:n_plot], vec(out1["Sk_x"][:,1,:])[n_start:n_plot])
plot!(out1["t"][n_start:n_plot], vec(out1["Sk_x"][:,1,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,2,:])[n_start:n_plot])
plot!(out1["t"][n_start:n_plot], vec(out1["Sk_x"][:,1,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,2,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,3,:])[n_start:n_plot])

plot(out1["t"][n_start:n_plot], vec(out1["Sk_x"][:,1,:])[n_start:n_plot])
plot(out1["t"][n_start:n_plot], vec(out1["Sk_x"][:,2,:])[n_start:n_plot])
plot!(out1["t"][n_start:n_plot], vec(out1["Sk_x"][:,3,:])[n_start:n_plot])


plot(vec(out1["Sk_x"][:,1,:])[n_start:n_plot], vec(out1["Sk_x"][:,2,:])[n_start:n_plot])
plot(vec(out1["Sk_x"][:,2,:])[n_start:n_plot], vec(out1["Sk_x"][:,3,:])[n_start:n_plot])


plot(vec(out1["Sk_x"][:,1,:])[n_start:n_plot], vec(out1["Sk_x"][:,1,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,2,:])[n_start:n_plot])
plot(vec(out1["Sk_x"][:,1,:])[n_start:n_plot], vec(out1["Sk_x"][:,1,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,2,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,3,:])[n_start:n_plot])
plot(vec(out1["Sk_x"][:,1,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,2,:])[n_start:n_plot], vec(out1["Sk_x"][:,1,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,2,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,3,:])[n_start:n_plot])
