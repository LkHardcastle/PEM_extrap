function h_track_init(priors::FixedPrior, settings::Settings)
    return 0.0
end

function h_store!(h_track, priors::FixedPrior, dyn::Dynamics)

end

function w_order_init(s::Matrix{Bool}, priors::GeomPrior)
    w_order = zeros(size(s,1),size(s,2))
    k = 0
    for i in axes(s,1)
        for j in axes(s,2)
            if j == 1
                k = priors.geom_max
            end
            if s[i,j]
                w_order[i,j] = 1
                k = 1
            else
                w_order[i,j] = k
            end
            k += 1
        end
    end
    for i in eachindex(priors.ω)
        priors.ω[i] ω0
    end
    return w_order, priors
end