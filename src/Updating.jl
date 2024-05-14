function flip!(state::ZZS, dat::PEMData, priors::Prior)
    ### Calculate gradient
    ### Select component to flip
    ### Flip
    λ = max.(0.0, ∇U(state, dat, priors))
    Λ = sum(λ)
    j = rand(Categorical(λ./Λ))
    if j == 1
        state.v[CartesianIndex(1,1):state.active[j]] = -state.v[CartesianIndex(1,1):state.active[j]]
    else
        state.v[(state.active[j-1] + CartesianIndex(0,1)):state.active[j]] = -state.v[(state.active[j-1] + CartesianIndex(0,1)):state.active[j]]
    end
end

function refresh!(state::ZZS)
    println("No ZZS refresh mechanism")
end

function flip!(state::BPS)
    ### Calculate gradient

    ### Flip
end

function flip!(state::ECMC)
    ### Calculate gradient

    ### Gradient update


    ### Orthogonal update
end

function update!(state::State, t::Float64)
    state.x += state.v*t
    state.t += t
end

function event!(state::State, dyn::Dynamics, priors::Prior)
    if dyn.next_event == 1
        # Split
        error("Shouldn't be here yet")
    end
    if dyn.next_event == 2
        # Merge 
        error("Shouldn't be here yet")
    end
    if dyn.next_event == 3
        # Refresh
        refresh!(state)
    end
    if dyn.next_event == 4
        # Hyperparameter update
        hyper_update!(priors)
    end
end

function split!()

end

function merge!()

end

function hyper_update!(priors::Priors)

end

