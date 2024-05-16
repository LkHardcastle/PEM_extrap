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
    state.v[state.active] = state.v[state.active]/norm(state.v[state.active])
end

function flip!(state::ECMC, dat::PEMData, priors::Prior)
    ### Calculate gradient
    U_grad = ∇U(state, dat, priors)
    ### Gradient update
    #v_0 = gradient_update(state)
    #state.v[state.active] = -v_0*U_grad .+ (state.v[state.active] .- dot(state.v[state.active], U_grad)*U_grad/norm(U_grad)^2)
    #state.v[state.active] = state.v[state.active]/norm(state.v[state.active])
    ### Orthogonal update
    println(U_grad)
    O = orthogonal_update(state, U_grad)
    println(O)
    v_grad = dot(state.v[state.active], U_grad)/norm(U_grad)^2
    println(v_grad)
    v_perp = (state.v[state.active] .- dot(state.v[state.active], U_grad)*U_grad/norm(U_grad)^2)
    println(v_perp)
    v_perp_new = ((1-v_grad^2)^0.5)*sign(dot(v_perp, O*v_perp))*(O*(v_perp/norm(v_perp)))
    state.v[state.active] = v_grad*U_grad + v_perp_new
end

function refresh!(state::ECMC)
    state.v[state.active] = rand(Normal(0,1),size(state.active))
    state.v[state.active] = state.v[state.active]/norm(state.v[state.active])
end

function gradient_update(state::ECMC)
    u = rand()
    return (1 - u^(2/(size(state.active,1)-1)))^0.5
end

function orthogonal_update(state::ECMC, U_grad::Vector{Float64})
    g1 = (I - U_grad*transpose(U_grad))*rand(MvNormal(zeros(size(state.active,1)),I))
    g2 = (I - U_grad*transpose(U_grad))*rand(MvNormal(zeros(size(state.active,1)),I))
    e1 = g1/norm(g1)
    e2 = (g2 - dot(e1,g2)*e1)/norm((g2 - dot(e1,g2)*e1))
    O = e2*transpose(e1) + e1*transpose(e2)
    return O
end

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

