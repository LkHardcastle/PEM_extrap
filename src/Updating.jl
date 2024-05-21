
function flip!(state::BPS, dat::PEMData, priors::Prior)
    ### Calculate gradient
    U_grad = ∇U(state, dat, priors)
    ### Flip
    state.v[state.active] -= 2*dot(state.v[state.active], U_grad)*U_grad/norm(U_grad)^2
    
end

function refresh!(state::BPS)
    state.v[state.active] = rand(Normal(0,1),size(state.active))
    state.v[state.active] /= norm(state.v[state.active])
end

function flip!(state::ECMC, dat::PEMData, priors::Prior)
    ### Calculate gradient
    U_grad = ∇U(state, dat, priors)
    state.v[state.active] -= 2*dot(state.v[state.active], U_grad)*U_grad/norm(U_grad)^2
    U_grad = U_grad/norm(U_grad)
    ### Gradient update
    v_0_new = -(1 - rand()^(2/(size(state.active,1)-1)))^0.5
    v_0_old = dot(U_grad, state.v[state.active])
    state.v[state.active] -= v_0_old*U_grad
    state.v[state.active] /= norm(state.v[state.active])
    state.v[state.active] = ((1-v_0_new^2)^0.5)*state.v[state.active] .+ v_0_new*U_grad
    ### Orthogonal update
    #refresh!(state)
end

function gram_schmidt(state::State, U_grad::Vector{Float64})
    #g1 = rand(Normal(0,1),size(state.active,1))
    #g2 = rand(Normal(0,1),size(state.active,1))
    g1 = (I - U_grad*transpose(U_grad))*rand(Normal(0,1),size(state.active,1))
    g2 = (I - U_grad*transpose(U_grad))*rand(Normal(0,1),size(state.active,1))
    e1 = g1/norm(g1)
    e2 = (g2 - dot(g1,g2)*g1)
    e2 = e2/norm(e2)
    return e1, e2
end

function refresh!(state::ECMC)
    #state.v[state.active] = rand(Normal(0,1),size(state.active))
    #state.v[state.active] = state.v[state.active]/norm(state.v[state.active])
    U_grad = ∇U(state, dat, priors)
    U_grad = U_grad/norm(U_grad)
    v_perp = state.v[state.active] - dot(state.v[state.active], U_grad)*U_grad
    g1, g2 = gram_schmidt(state, U_grad)
    g1_, g2_ = dot(g1, v_perp), dot(g2, v_perp)
    v_perp += g1_*(g2 - g1) + g2_*(g1 - g2)
    v_perp *= dot(state.v[state.active] - dot(state.v[state.active], U_grad)*U_grad, v_perp)
    v_perp /= norm(v_perp)
    b = dot(state.v[state.active], U_grad)
    a = (1 - b^2)^0.5
    state.v[state.active] = a*v_perp + b*U_grad
 end

 function flip!(state::ECMC2, dat::PEMData, priors::Prior)
    ### Calculate gradient
    U_grad = ∇U(state, dat, priors)
    state.v[state.active] -= 2*dot(state.v[state.active], U_grad)*U_grad/norm(U_grad)^2
    U_grad = U_grad/norm(U_grad)
    ### Gradient update
    v_0_new = -(1 - rand()^(2/(size(state.active,1)-1)))^0.5
    v_0_old = dot(U_grad, state.v[state.active])
    state.v[state.active] -= v_0_old*U_grad
    state.v[state.active] /= norm(state.v[state.active])
    state.v[state.active] = ((1-v_0_new^2)^0.5)*state.v[state.active] .+ v_0_new*U_grad
    ### Orthogonal update
    if state.b
        U_grad = ∇U(state, dat, priors)
        U_grad = U_grad/norm(U_grad)
        v_perp = state.v[state.active] - dot(state.v[state.active], U_grad)*U_grad
        g1, g2 = gram_schmidt(state, U_grad)
        g1_, g2_ = dot(g1, v_perp), dot(g2, v_perp)
        v_perp += g1_*(g2 - g1) + g2_*(g1 - g2)
        v_perp *= dot(state.v[state.active] - dot(state.v[state.active], U_grad)*U_grad, v_perp)
        v_perp /= norm(v_perp)
        b = dot(state.v[state.active], U_grad)
        a = (1 - b^2)^0.5
        state.v[state.active] = a*v_perp + b*U_grad
        state.b = false
    end
end

function refresh!(state::ECMC2)
    state.b = true
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
        split!(state)
        split_time!(state, times, priors)
        merge_time!(state, times, priors)
        error("Shouldn't be here yet")
    end
    if dyn.next_event == 2
        # Merge 
        merge!(state, times.next_merge_index)
        split_time!()
        merge_time!()
        error("Shouldn't be here yet")
    end
    if dyn.next_event == 3
        # Refresh
        refresh!(state)
        merge_time_ref!(state, times, priors)
        deleteat!(times.refresh, 1)
        dyn.next_event = 0
    end
    if dyn.next_event == 4
        # Hyperparameter update
        hyper_update!(priors)
        split_time!(state, times, priors)
        deleteat!(times.hyper, 1)
    end
end

function hyper_update!(priors::Prior)

end

