using Pkg
Pkg.activate(".")
using HomotopyContinuation, LinearAlgebra, Plots, DifferentialEquations

const HC = HomotopyContinuation
const DE = DifferentialEquations

include("functions.jl")


# System for the incidence variety of the discriminant
@var a b x
F = System([x^2 + a * x + b; 2x + a], variables=[a, b, x])

# Choice of parameters
B = qr(rand(2, 2)).Q |> Matrix
c = 10 .* randn(2)
e = 2

###### Critical points 
old_pts = include("discr_pw.jl")|> unique_points
pts = routing_points(F, [a; b]; c=c, B=B, e=e)


###### ODE Solver
g = ∇log_r(F, [a; b]; c=c, B=B)
f(x, param, t) = g(x)


u0 = pts[rand(1:length(pts))]
tspan = (0.0, 1e4)
prob = ODEProblem(f, u0, tspan)
sol = DE.solve(prob, reltol=1e-6, abstol=1e-6)

##### Plotting 
M = maximum(abs, vcat(pts...))
M_x = maximum(p->abs(p[1]), pts)
M_y = maximum(p->abs(p[2]), pts)


R(x, y) = log(abs((x^2 - 4 * y) / evaluate(q, p => [x; y])))
contour(
   (-M_x):0.1:M_x,
   (-M_y):0.1:M_y,
    R,
    levels=50,
    color=:plasma,
    clabels=false,
    cbar=false,
    lw=1,
    fill=true,
)
A = [[b; b^2 / 4] for b = (-M_x):M_x]
plot!(
    Tuple.(A),
    xlims=(-M_x, M_x),
    ylims=(-M_y, M_y),
    linecolor=:black,
    linewidth=8,
    label="discriminant",
)

scatter!(Tuple.(pts), markercolor=:green, markersize=8, label="critical points")

plot!(Tuple.(sol.u), linecolor=:steelblue, linewidth=4, label="gradient flow")

scatter!([Tuple(u0)], markercolor=:blue, markersize=8, label="gradient flow start")

plot!(; legend=true)

#savefig("presentation.png")
