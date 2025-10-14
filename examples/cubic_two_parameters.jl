using Pkg
Pkg.activate(".")

using ImplicitPlots, Plots, DifferentialEquations, Random

include("../src/functions.jl");

Random.seed!(0x8b868320)

########

# The discriminant of x^3 + a * x^2 + b*x + 1 is
# 4*a^3 - a^2*b^2 - 18*a*b + 4*b^3 + 27
@var a b x
F = System([x^3 + a * x^2 + b*x + 1; 3*x^2 + 2*a*x + b], variables = [a, b, x])

projection_variables = [a; b]
k = length(projection_variables)

B = qr(rand(k, k)).Q |> Matrix # not needed anymore (but kept for reproducibility)
c = 10 .* randn(k)
r = RoutingGradient(F, projection_variables; c = c)

# critical points
res = critical_points(r)
pts = real_solutions(res)

##### Plotting 
M_x = maximum(p -> abs(p[1]), pts) + 6
M_y = maximum(p -> abs(p[2]), pts) + 6


# Discriminant
h(x, y) = 4*x^3 - x^2*y^2 - 18*x*y + 4*y^3 + 27

R(x, y) = log(abs(h(x,y) / (1 + (x-c[1])^2 + (y-c[2])^2)^e))
contour(
    (-M_x):0.1:M_x,
    (-M_y):0.1:M_y,
    R,
    levels = 50,
    color = :plasma,
    clabels = false,
    cbar = false,
    lw = 1,
    fill = true,
)


implicit_plot!(
    h; 
    xlims = (-M_x, M_x),
    ylims = (-M_y, M_y),
    linecolor = :black,
    linewidth = 6,
    label = "discriminant"
)

# Gradient flow for routing points of positive index
g(x, param, t) = real(evaluate(r, x))
tspan = (0.0, 1e4)
for (i, u0) in enumerate(pts)
    println()
    println("Critical point #$i")
    jac = real(evaluate_and_jacobian(r, u0)[2])
    eigen_data = eigen(jac)
    eigenvalues = eigen_data.values
    eigenvectors = eigen_data.vectors
    positive_directions = [i for (i, λ) in enumerate(eigenvalues) if real(λ) > 0]
    println("Index: $(length(positive_directions))")
    if !isempty(positive_directions)
        idx = first(positive_directions)
        v = eigenvectors[:, idx]
        prob = ODEProblem(g, u0 + 0.01*v, tspan)
        sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
        plot!(Tuple.(sol.u), linecolor = :steelblue, linewidth = 4, label="")
        prob = ODEProblem(g, u0 - 0.01*v, tspan)
        sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
        plot!(Tuple.(sol.u), linecolor = :steelblue, linewidth = 4, label="")
        scatter!(Tuple(u0), markercolor = :magenta, markersize = 8, label = "")
    else
        scatter!(Tuple(u0), markercolor = :green, markersize = 8, label = "")
    end

end

# Legend
plot!([], [], color = :steelblue, linewidth = 4, label = "gradient flow")
scatter!(Tuple(NaN), markercolor = :green, markersize = 8, label = "routing pts (index 0)")
scatter!(Tuple(NaN), markercolor = :magenta, markersize = 8, label = "routing pts (index > 0)")

plot!(; legend = :bottomright, dpi=400, legendfontsize=6)

savefig("./figures/example_cubic.svg")
savefig("./figures/example_cubic.png")