using Pkg, Random, Plots, DifferentialEquations, Random, Plots, DifferentialEquations, LightGraphs
Pkg.activate(".")

include("../src/functions.jl");


########
@var a b x
F = System([x^2 + a * x + b; 2x + a], variables = [a, b, x])

B = qr(rand(2, 2)).Q |> Matrix # not needed anymore (but kept for reproducibility)
c = [0.15058793143477558; 11.593556720643692]
r = RoutingGradient(F, [a, b]; c = c)

p1 = zeros(2)
q1 = randn(2)
H = RoutingPointsHomotopy(r, p1, q1)

### Test evaluation
u = randn(ComplexF64, 2)
U = randn(ComplexF64, 2, 2)
x0 = randn(ComplexF64, 2)
t0 = 1.0
@time evaluate_and_jacobian!(u, U, H, x0, t0)
evaluate_and_jacobian!(u, U, H, x0, t0)


### monodromy
res0 = critical_points(r)
pts = real_solutions(res0)

### connecting 
G, idx, failed_info = partition_of_critical_points(r, pts)
G



##### Plotting 
M_x = maximum(p -> abs(p[1]), pts) + 4
M_y = maximum(p -> abs(p[2]), pts) + 3


RR(x, y) = log(abs((x^2 - 4 * y) / (1 + (x-c[1])^2 + (y-c[2])^2)^2)) #This is our routing function
contour(
    (-M_x):0.1:M_x,
    (-M_y):0.1:M_y,
    RR,
    levels = 50,
    color = :plasma,
    clabels = false,
    cbar = false,
    lw = 1,
    fill = true,
)

A = [[b; b^2 / 4] for b = (-M_x):M_x] #discriminant of the quadratic #discriminant of the quadratic
plot!(
    Tuple.(A),
    xlims = (-M_x, M_x),
    ylims = (-M_y, M_y),
    linecolor = :black,
    linewidth = 8,
    label = "discriminant",
)

scatter!(Tuple.(pts[G[1]]), markercolor = :green, markersize = 8, label = "region 1")
scatter!(Tuple.(pts[G[2]]), markercolor = :steelblue, markersize = 8, label = "region 2")





# # Gradient flow for routing points of positive index
# g(x, param, t) = real(evaluate(r, x))
# tspan = (0.0, 1e4)
# for (i, u0) in enumerate(pts)
#     println()
#     println(i)
#     jac = real(evaluate_and_jacobian(r, u0)[2])
#     eigen_data = eigen(jac)
#     eigenvalues = eigen_data.values
#     eigenvectors = eigen_data.vectors
#     positive_directions = [i for (i, λ) in enumerate(eigenvalues) if real(λ) > 0]
#     println("Index: $(length(positive_directions))")
#     if !isempty(positive_directions)
#         idx = first(positive_directions)
#         v = eigenvectors[:, idx]
#         prob = ODEProblem(g, u0 + 0.01*v, tspan)
#         sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
#         plot!(Tuple.(sol.u), linecolor = :steelblue, linewidth = 4, label="")
#         prob = ODEProblem(g, u0 - 0.01*v, tspan)
#         sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
#         plot!(Tuple.(sol.u), linecolor = :steelblue, linewidth = 4, label="")
#         scatter!(Tuple(u0), markercolor = :magenta, markersize = 8, label = "")
#     else
#         scatter!(Tuple(u0), markercolor = :green, markersize = 8, label = "")
#     end

# end

# # Legend
# plot!([], [], color = :steelblue, linewidth = 4, label = "gradient flow")
# scatter!(Tuple(NaN), markercolor = :green, markersize = 8, label = "routing pts (index 0)")
# scatter!(Tuple(NaN), markercolor = :magenta, markersize = 8, label = "routing pts (index > 0)")

# plot!(; legend = :bottomright, dpi=400, legendfontsize=6)

# savefig("./figures/example_quadratic.png")
# savefig("./figures/example_quadratic.svg")