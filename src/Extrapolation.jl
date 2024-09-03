function barker_extrapolation(out::Dict, diffs::Diffusion, grid::Grid, t_start::Float64, t_end::Float64, plot_grid::Vector{Float64}, k::Int64)
    ## Get end points
    n_smp = size(out["Smp_t"],1)
    Σθ = cumsum(out["Smp_x"], dims = 2)
    initial = Σθ[k,end,:]
    if sum(isinf.(initial)) > 0.0
        for i in 1:n_smp
            initial[i] = Σθ[k, findlast(isinf.(Σθ[k,:,i]) .== false), i]
        end
    end
    ## Get time points
    times = extrapolation_times(out, grid, t_start, t_end, n_smp, out["Smp_ω"][k,:])
    ## Simulate dynamics
    paths = Vector{Vector{Float64}}()
    for i in 1:n_smp
        push!(paths, barker_dynamics(initial[i], size(times[i],1), diffs, out["Smp_σ"][k,i]))
    end
    output = fill(Inf, length(plot_grid), n_smp)
    for i in 1:n_smp
        for j in 1:length(plot_grid)
            if isnothing(findlast(times[i] .<= plot_grid[j]))
                ind = 1
            else
                ind = findlast(times[i] .<= plot_grid[j])
            end
            output[j, i] = paths[i][ind]
        end
    end
    return output
end

function extrapolation_times(out::Dict, grid::Fixed, t_start::Float64, t_end::Float64, n_smp::Int64, ω::Vector{Float64})
    times = Vector{Vector{Float64}}()
    # Find candidate times
    cand_times = collect(t_start:grid.step:t_end)
    # Thin
    for i in 1:n_smp
        push!(times, vcat(t_start - grid.step, cand_times[findall(rand(Bernoulli(ω[i]),size(cand_times,1)) .== 1)]))
    end
    return times
end

function extrapolation_times(out::Dict, grid::Cts, t_start::Float64, t_end::Float64, n_smp::Int64, ω::Vector{Float64})
    times = Vector{Vector{Float64}}()
    for i in 1:n_smp
        times_inner = [t_start]
        while times_inner[end] < t_end
            push!(times_inner, times_inner[end] + rand(Exponential(1/(grid.Γ*ω[i]))))
        end
        push!(times, times_inner)
    end
    return times
end

function barker_dynamics(init::Float64, iters::Int64, diff::Diffusion, step_size::Float64)
    out_vec = zeros(iters)
    out_vec[1] = init
    for i in 2:iters
        ξ = rand(Normal(0,step_size))
        if rand() < 1/(1 + exp.(-2*ξ*drift(out_vec[i-1], diff)))
            out_vec[i] = out_vec[i-1] + ξ
        else
            out_vec[i] = out_vec[i-1] - ξ
        end
    end
    return out_vec
end
