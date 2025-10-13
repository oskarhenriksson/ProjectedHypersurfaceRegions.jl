using Pkg
Pkg.activate(".")

using ImplicitPlots, Plots, DifferentialEquations, Random

include("../src/functions.jl");

Random.seed!(0x8b868320)

########

# The discriminant of x^3 + a * x^2 + b*x + γ is
# 4*a^3*γ - a^2*b^2 - 18*a*b*γ + 4*b^3 + 27*γ^2
@var a b γ x
f = x^3 + a * x^2 + b*x + γ 
F = System([f; differentiate(f, x)], variables = [a, b, γ, x])

projection_variables = [a; b; γ]
k = length(projection_variables)

B = qr(rand(k, k)).Q |> Matrix # not needed anymore (but kept for reproducibility)
c = 10 .* randn(k)
r = RoutingGradient(F, projection_variables; c = c)
e = denominator_exponent(r)

p1 = zeros(k)
q1 = randn(k)
H = RoutingPointsHomotopy(r, p1, q1)

### Use monodromy to the system ∇r = p0 where we view the right-hand side are the parameters of the system
egtracker = EndgameTracker(H) # we want to add options later 
trackers = [egtracker]
x₀ = zeros(ComplexF64, size(H, k))

unique_points = UniquePoints(
    x₀,
    1;
)

trace = zeros(ComplexF64, length(x₀) + 1, 3)
P = Vector{ComplexF64}
options = MonodromyOptions(
    #parameter_sampler = p -> 10 .* [0; randn(ComplexF64, length(p) - 1)], # bigger lopps
    parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)), # bigger lopps
    # max_loops_no_progress = 10 # change the stopping criterion
) 
MS = HomotopyContinuation.MonodromySolver(
    trackers,
    HomotopyContinuation.MonodromyLoop{P}[],
    unique_points,
    ReentrantLock(),
    options,
    HomotopyContinuation.MonodromyStatistics(),
    trace,
    ReentrantLock(),
)

#### set up start pair
s0 = randn(ComplexF64, k)
p0 = evaluate(r, s0)

# Make sure Jacobians work
evaluate_and_jacobian(r, s0, p0)

### Monodromy 
seed = rand(UInt32)
mon_result = monodromy_solve(
    MS,
    s0,
    p0,
    seed;
)

### Use a parameter homotopy (via the "detour trick") to trace the solutions of ∇r = p0 to solutions of ∇r = 0
start_parameters!(egtracker, p0)
p_intermediate = randn(ComplexF64, length(p0))
target_parameters!(egtracker, p_intermediate)
intermediate_result = HomotopyContinuation.solve(H, solutions(mon_result))

start_parameters!(egtracker, p_intermediate)
target_parameters!(egtracker, zeros(length(p0)))
result = HomotopyContinuation.solve(H, solutions(intermediate_result))


pts = real_solutions(result)

g(x, param, t) = real(evaluate(r, x))
tspan = (0.0, 1e4)
for (i, u0) in enumerate(pts)
    println()
    println("Critical point #$i")
    println("Point: $u0")
    jac = real(evaluate_and_jacobian(r, u0)[2])
    eigen_data = eigen(jac)
    eigenvalues = eigen_data.values
    eigenvectors = eigen_data.vectors
    positive_directions = [i for (i, λ) in enumerate(eigenvalues) if real(λ) > 0]
    println("Index: $(length(positive_directions))")
    if  !isempty(positive_directions)
        idx = first(positive_directions)
        v = 1e-4*eigenvectors[:, idx]
        prob1 = ODEProblem(g, u0 + v, tspan)
        sol1 = DE.solve(prob1, reltol = 1e-6, abstol = 1e-6)
        limit1 = last(sol1.u) 
        closest_distance1, closest_point_idx1 = findmin(i->norm(pts[i] .- limit1), 1:length(pts))
        if closest_distance1 > 1e-6
            @warn "Missing critical point identified by gradient flow! Added to the list of points"
            append!(pts, limit1)
            closest_point_idx1 = length(pts)
        end
        prob2 = ODEProblem(g, u0 - v, tspan)
        sol2 = DE.solve(prob2, reltol = 1e-6, abstol = 1e-6)
        limit2 = last(sol2.u)
        closest_distance2, closest_point_idx2 = findmin(i->norm(pts[i] .- limit2), 1:length(pts))
        if closest_distance2 > 1e-6
            @warn "Missing critical point identified by gradient flow! Added to the list of points"
            append!(pts, limit2)
            closest_point_idx2 = length(pts)
        end
        println("Connects the critical points: $closest_point_idx1 and $closest_point_idx2")
    else
        specialized_polynomial = subs(f, projection_variables=>u0)
        res = HomotopyContinuation.solve([specialized_polynomial])
        println("Number of real roots: $(length(real_solutions(res)))")
    end
end