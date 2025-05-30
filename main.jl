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


###### ODE Solver
g = ∇log_r(F, [a; b]; c = c, B = B) # 0.041457 seconds (14.08 k allocations: 2.945 MiB)
@time g(randn(2))
f(x, param, t) = g(x)


u0 = [-15, 10]
tspan = (0.0, 1e4)
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
