#function flip!(state::ZZS, dat::PEMData, priors::Prior)
    ### Calculate gradient
    ### Select component to flip
    ### Flip
#    λ = max.(0.0, ∇U(state, dat, priors))
#    Λ = sum(λ)
#    j = rand(Categorical(λ./Λ))
#    if j == 1
#        state.v[CartesianIndex(1,1):state.active[j]] = -state.v[CartesianIndex(1,1):state.active[j]]
#    else
#        state.v[(state.active[j-1] + CartesianIndex(0,1)):state.active[j]] = -state.v[(state.active[j-1] + CartesianIndex(0,1)):state.active[j]]
#    end
#end

#function refresh!(state::ZZS)
#    println("No ZZS refresh mechanism")
#end

function flip!(state::BPS, dat::PEMData, priors::Prior)
    ### Calculate gradient
    U_grad = ∇U(state, dat, priors)
    ### Flip
    #println(state.v[state.active] .- 2*dot(state.v[state.active], U_grad)*U_grad/norm(U_grad)^2)
    #println(dot(state.v[state.active], U_grad))
    #println(U_grad)
    #println(norm(U_grad))
    state.v[state.active] = state.v[state.active] - 2*dot(state.v[state.active], U_grad)*U_grad/norm(U_grad)^2
    
end

function refresh!(state::BPS)
    state.v[state.active] = rand(Normal(0,1),size(state.active))
end

#function flip!(state::ECMC)
    ### Calculate gradient

    ### Gradient update


    ### Orthogonal update
#end
function update!(state::State, t::Float64)
    if t < 0.0
        error("Time travel")
    end
    state.x += state.v*t
    state.t += t
end

function event!(state::State, dyn::Dynamics, priors::Prior, times::Times)
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
        deleteat!(times.refresh, 1)
        dyn.next_event = 0
    end
    if dyn.next_event == 4
        # Hyperparameter update
        hyper_update!(priors)
        deleteat!(times.hyper, 1)
    end
end

function split!()

end

function merge!()

end

function hyper_update!(priors::Prior)

end

