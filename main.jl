using Pkg
Pkg.activate(".")

using Plots, DifferentialEquations

include("src/functions.jl");


# System for the incidence variety of the discriminant
@var a b x
F = System([x^2 + a * x + b; 2x + a], variables = [a, b, x])

# Choice of parameters
B = qr(rand(2, 2)).Q |> Matrix
c = 10 .* randn(2)
e = 2

# Denominator
@var p[1:2]
q = 1 + sum((p - c) .* (p - c))


###### Critical points 
pts = routing_points(F, [a; b]; c = c, e = e)

g = ∇log_r(F, [a; b]; c = c, B = B)
p = randn(2)
@time g(p) # 387 allocations
# old_pts = include("old/discr_pw.jl")|> unique_points
# map(norm, sort(pts)-sort(old_pts))

#### Test RoutingGradient
r = RoutingGradient(F, [a; b]; c = c, B = B)
@time _evaluate(r,p)
u = zeros(Float64, 2)
evaluate!(u, r, p)
_evaluate(r, p)


###### ODE Solver

f(x, param, t) = _evaluate(r, x)




u0 = randn(2)
tspan = (0.0, 1e4)
prob = ODEProblem(f, u0, tspan)
sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)



##### Plotting 
M = maximum(abs, vcat(pts...))
M_x = maximum(p -> abs(p[1]), pts)
M_y = maximum(p -> abs(p[2]), pts)

R(x, y) = log(abs((x^2 - 4 * y) / evaluate(q, p => [x; y])))
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
A = [[b; b^2 / 4] for b = (-M_x):M_x]
plot!(
    Tuple.(A),
    xlims = (-M_x, M_x),
    ylims = (-M_y, M_y),
    linecolor = :black,
    linewidth = 8,
    label = "discriminant",
)

scatter!(Tuple.(pts), markercolor = :green, markersize = 8, label = "critical points")
plot!(Tuple.(sol.u), linecolor = :steelblue, linewidth = 4, label = "gradient flow")
#scatter!([Tuple(u0)], markercolor=:blue, markersize=8, label="gradient flow start")

plot!(; legend = false)


#### Do gradient flow at all critical points
for (i, u0) in enumerate(pts)
    println(i)
    tspan = (0.0, 1e4)
    prob = ODEProblem(f, u0, tspan)
    sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
    plot!(Tuple.(sol.u), linecolor = :steelblue, linewidth = 4, label = "gradient flow")
end

plot!(; legend = false)

#savefig("presentation.png")
