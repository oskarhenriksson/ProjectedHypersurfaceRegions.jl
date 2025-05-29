
using Pkg
Pkg.activate(".")
using HomotopyContinuation, LinearAlgebra, Plots, DifferentialEquations

const HC = HomotopyContinuation
const DE = DifferentialEquations

include("grad_r_p.jl")


@var a b x
F = System([x^2 + a*x + b; 2x + a], variables = [a, b, x])
# System for the incidence variety of the discriminant

B = qr(rand(2, 2)).Q |> Matrix
c = 10 .* randn(2)
e = 2

###### Critical points 
pts = include("discr_pw.jl")

###### PWS
u = rand(ComplexF64, 2)
v = rand(ComplexF64, 2)
PWS = PseudoWitnessSet(F, create_line(u, v, 3))

###### ODE Solver
f(x, param, t) = real(∇log_r(c, e, x, PWS))
u0 = [-2, 10]
tspan = (0.0, 1e5)
prob = ODEProblem(f, u0, tspan)
sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)

##### Plotting 
M = 20
R(x, y) = log(abs((x^2 - 4*y)/evaluate(q, p => [x; y])))
contour(
    (-M):0.1:M,
    (-M):0.1:M,
    R,
    levels = 50,
    color = :plasma,
    clabels = false,
    cbar = false,
    lw = 1,
    fill = true,
)
A = [[b; b^2/4] for b = (-M):M]
plot!(
    Tuple.(A),
    xlims = (-M, M),
    ylims = (-M, M),
    linecolor = :black,
    linewidth = 8,
    label = "discriminant",
)

scatter!(Tuple.(pts), markercolor = :green, markersize = 8, label = "critical points")

plot!(Tuple.(sol.u), linecolor = :steelblue, linewidth = 4, label = "gradient flow")

scatter!([Tuple(u0)], markercolor = :blue, markersize = 8, label = "gradient flow start")

plot!(; legend = false)

#savefig("presentation.png")
