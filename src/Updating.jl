function flip!(ZZS)
    ### Calculate gradient
    ### Select component to flip
    ### Flip
end

function flip!(BPS)
    ### Calculate gradient

    ### Flip
end

function flip!(ECMC)
    ### Calculate gradient

    ### Gradient update


    ### Orthogonal update
end

function update!(state::State, t::Float64)
    state.x += state.v*t
    state.t += t
end

function event!()
    if dyn.next_event == 1
        # Split
    end
    if dyn.next_event == 2
        # Merge 
    end
    if dyn.next_event == 3
        # Refresh
    end
end

function split!()

end

function merge!()

end

