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

B = qr(rand(k, k)).Q |> Matrix
γ = 10 .* randn(k)
r = RoutingGradient(F, projection_variables; c = γ, B = B)
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
options = MonodromyOptions() 
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

evaluate_and_jacobian(r,p0)

### Monodromy 
seed = rand(UInt32)
@profview mon_result = monodromy_solve(
    MS,
    s0,
    p0,
    seed;
)

### Use a parameter homotopy to trace the solutions of ∇r = p0 to solutions of ∇r = 0

# Direct parameter homotopy
# start_parameters!(egtracker, p0)
# target_parameters!(egtracker, zeros(2))
# result = HomotopyContinuation.solve(H, solutions(mon_result))
# #pv = @profview result = HomotopyContinuation.solve(H, solutions(mon_result))
# pts = real_solutions(result)


# Parameter homotopy using the "detour trick"
# NOTE: You might have to repeat this since paths are sometimes lost
start_parameters!(egtracker, p0)
p_intermediate = randn(ComplexF64, 2)
target_parameters!(egtracker, p_intermediate)
intermediate_result = HomotopyContinuation.solve(H, solutions(mon_result))

start_parameters!(egtracker, p_intermediate)
target_parameters!(egtracker, zeros(2))
result = HomotopyContinuation.solve(H, solutions(intermediate_result))

pts = real_solutions(result)