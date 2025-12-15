using Plots, ImplicitPlots

function analyze_result(
    ∇r::RoutingGradient,
    pts::Vector{Vector{Float64}},
    G::Vector{Vector{Int}},
    idx::Vector{Int};
    root_counting_system::Union{System,Nothing}=nothing,
    h::Union{Function,Nothing}=nothing,
    RR::Union{Function,Nothing}=nothing,
    root_count_condition::Union{Function,Nothing}=nothing,
    arrowstyle=:closed,
    markersize=3,
    legend=false,
    flow_linewidth=2,
    discriminant_linewidth=2,
    flow_breakpoint_ratio=3,
    plot_contour=true,
    contour_stepsize=0.01,
    M_x_max=nothing,
    M_x_min=nothing,
    M_y_max=nothing,
    M_y_min=nothing
)


    # Analyze root counts
    if !isnothing(root_counting_system)
        for (i, comp) in enumerate(G)
            println("Connected component #$i")
            root_counts = Int[]
            for j in comp
                real_zeros = HC.solve(root_counting_system, target_parameters=pts[j]) |> real_solutions
                filtered_solutions = isnothing(root_count_condition) ? real_zeros : filter(root_count_condition, real_zeros)
                rc = length(filtered_solutions)
                push!(root_counts, rc)
            end
            isnothing(root_count_condition) ? println("Real root counts: $(root_counts)\n") : println("Filtered real root counts: $(root_counts)\n")
        end
    end



    isnothing(M_x_max) && (M_x_max = maximum(p -> p[1], pts) * 1.8)
    isnothing(M_x_min) && (M_x_min = minimum(p -> p[1], pts) * 1.8)
    isnothing(M_y_max) && (M_y_max = maximum(p -> p[2], pts) * 1.8)
    isnothing(M_y_min) && (M_y_min = minimum(p -> p[2], pts) * 1.8)

    if isnothing(RR) && !isnothing(h)
        RR = (x,y) -> log(abs(h(x, y) / (1 + (x - center[1])^2 + (y - center[2])^2)^e))
    end
    @show plot_contour
    # Plot the result
    if !isnothing(RR) 
        e = denominator_exponent(∇r)
        center = ∇r.r.c

        if plot_contour
            pl = contour(
                (M_x_min):contour_stepsize:M_x_max,
                (M_y_min):contour_stepsize:M_y_max,
                RR,
                levels=50,
                color=:plasma,
                clabels=false,
                cbar=false,
                lw=1
            )
        else
            pl = plot()
        end

        implicit_plot!(pl, 
            h;
            xlims=(M_x_min, M_x_max),
            ylims=(M_y_min, M_y_max),
            linecolor=:black,
            linewidth=discriminant_linewidth,
            label="Discriminant",
            legend=false,
            resolution=3000
        )
    else
        pl = plot()
    end

    # Plot flows from the critical points
    idx0 = findall(iszero, idx)
    idx1 = findall(!iszero, idx)
    pts1 = pts[idx1]
    g(x, param, t) = real(evaluate(∇r, x))
    tspan = (0.0, 1e4)
    for u0 in pts1
        jac = real(evaluate_and_jacobian(∇r, u0)[2])
        eigen_data = LinearAlgebra.eigen(jac)
        eigenvalues = eigen_data.values
        eigenvectors = eigen_data.vectors
        positive_directions = [i for (i, λ) in enumerate(eigenvalues) if real(λ) > 0]
        j = first(positive_directions)
        v = eigenvectors[:, j]

        prob = ODEProblem(g, u0 + 0.01 * v, tspan)
        sol = DE.solve(prob, reltol=1e-6, abstol=1e-6)
        flow = Tuple.(sol.u)
        l = length(flow)
        k = div(l, flow_breakpoint_ratio)
        plot!(pl, flow[1:k], linecolor=:steelblue, linewidth=flow_linewidth, label=false, arrow=arrowstyle)
        plot!(pl, flow[k:end], linecolor=:steelblue, linewidth=flow_linewidth, label=false)
        prob = ODEProblem(g, u0 - 0.01 * v, tspan)
        sol = DE.solve(prob, reltol=1e-6, abstol=1e-6)
        flow = Tuple.(sol.u)
        l = length(flow)
        k = div(l, flow_breakpoint_ratio)
        plot!(pl, flow[1:k], linecolor=:steelblue, linewidth=flow_linewidth, label=false, arrow=arrowstyle)
        plot!(pl, flow[k:end], linecolor=:steelblue, linewidth=flow_linewidth, label=false)
    end

    # Plot the routing points
    scatter!(pl, Tuple.(pts[idx0]), markercolor=:green, markersize=markersize, label="Routing point (index 0)")
    scatter!(pl, Tuple.(pts[idx1]), markercolor=:green, markersize=markersize, marker=:diamond, label="Routing point (index > 0)")

    plot!(pl,; xlims=(M_x_min, M_x_max), ylims=(M_y_min, M_y_max), legend=legend, dpi=400, legendfontsize=6)

    pl
end