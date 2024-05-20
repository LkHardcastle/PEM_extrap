function merge_time(state::State, j::CartesianIndex)
    if (state.x[j] > 0.0 && state.v[j] < 0.0) || (state.x[j] < 0.0 && state.v[j] > 0.0)
        return abs(state.x[j])/abs(state.v[j])
    else
        return Inf
    end 
end

function split_rate(state::State, j::CartesianIndex, Priors::BasicPrior)
    rand(Exponential())
end