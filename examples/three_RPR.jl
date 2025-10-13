using Pkg, Random
Pkg.activate(".")
include("../src/functions.jl");

Random.seed!(0x8b868320)

#example 5.2 in the overleaf file

a2 = 14; a3 = 7; b3 = 10; A2 = 16; A3 = 9; B3 = 6;

@var p[1:2] φ[1:2] c[1:3]

f = [ φ[1]^2 + φ[2]^2 - 1,
      p[1]^2 + p[2]^2 - 2*(a3*p[1] +b3*p[2])*φ[1] + 2*(b3*p[1] - a3*p[2])*φ[2] + a3^2 + b3^2 - c[1],
      p[1]^2 + p[2]^2 - 2*A2*p[1] + 2*((a2-a3)*p[1] - b3*p[2] + A2*a3 -A2*a2)*φ[1] + 2*(b3*p[1] + (a2-a3)*p[2] - A2*b3)*φ[2]
        + (a2-a3)^2 + b3^ + A2^2 -c[2],
      p[1]^2 + p[2]^2 -2*(A3*p[1] + B3*p[2]) + A3^2 + B3^2 -c[3]
    ]

J = differentiate(f, vcat(p,φ))
D= det(J)

#Since we want the numerator of the routing function to have D(c)*c, 
#we add D*c to the system before we project down. Not sure about this. 
F = System([f; D])
c
projection_variables = c

k = length(projection_variables)

γ = 10 .* randn(k)
r = RoutingGradient(F, projection_variables; c = γ)
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
    parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)), # bigger loops
    max_loops_no_progress = 20 # change the stopping criterion
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
    println("Routing point #$i")
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
        closest_point1 = argmin(i->norm(pts[i] .- limit1), 1:length(pts))
        prob2 = ODEProblem(g, u0 - v, tspan)
        sol2 = DE.solve(prob2, reltol = 1e-6, abstol = 1e-6)
        limit2 = last(sol2.u)
        closest_point2 = argmin(i->norm(pts[i] .- limit2), 1:length(pts))
        println("Connected to the points: $closest_point1 and $closest_point2")
    end
end