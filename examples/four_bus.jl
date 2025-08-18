# Looking at the lossless four bus system, with zero power injections.
using Pkg
Pkg.activate(".")
include("../src/functions.jl");

# In this lossless four bus system with zero power injections, we have 6 parameters (6 bij where i\neq j) and 6 variables (Vd[1:3], Vq[1:3])
@var b[1:4, 1:4] Vd[1:4] Vq[1:4] 
#Vd is real component of voltage
#Vq is imaginary component of voltage
#b the susceptance of the line connecting buses i and k
eqn2_2a =
    sum.([[b[i, k] * (Vd[k] * Vq[i] - Vd[i] * Vq[k]) for k = 2:4 if i != k] for i = 2:4]) + [Vq[i] * b[1, i] for i = 2:4]
bus_eqns = [
    eqn2_2a
    Vd[2:4] .^ 2 - Vq[2:4] .^ 2 - ones(3)
]
#Note that b[i,k] == b[k,i]. As such, only keep one. 
bus_eqns = subs(bus_eqns, b[2,1]=> b[1,2], b[3,1]=> b[1,3], b[4,1]=> b[1,4], b[3,2]=> b[2,3], b[4,2]=> b[2,4], b[4,3]=> b[3,4])
Jac = differentiate(bus_eqns, [Vd[2:4]; Vq[2:4]])
D = det(Jac)

F = System([bus_eqns; D])
# System for the incidence variety of the discriminant
all_vars = variables(F)
x_vars = [Vd[2:4]; Vq[2:4]]
projection_vars = setdiff(all_vars, x_vars)
F_ordered = System(F.expressions, variables = [projection_vars; x_vars])
k = length(projection_vars)
PWS = PseudoWitnessSet(F_ordered, k; linear_subspace_codim = k - 1)

d = degree(PWS)
e= Int(floor(d/2)+1)
B = qr(rand(k, k)).Q |> Matrix
c = 10 .* randn(k)

r = RoutingGradient(F, projection_vars)

x0 = randn(ComplexF64, k)
evaluate_and_jacobian(r, x0) # works fine




######################################################
#Testing our hessian methods
hess_off_diag = hess_log_r(F_ordered, e, projection_vars; method = :off_diag, c, B)
hess_many_slices = hess_log_r(F_ordered, e, projection_vars; method = :many_slices, c, B)

P = rand(k)
hess_off_diag_eval = hess_off_diag(P)
hess_many_slices_eval = hess_many_slices(P)

# I guess the off_diag method is much slower than many_slices when the degree of the PWS is large.
#@time hess_off_diag(P) # 18 seconds (on my machine)
#@time hess_many_slices(P) # 6 seconds (on my machine)

# Unfortunately, these evaluations are really far apart. 
# You can see something similar when you look at discriminants of deg n univariate polynomials when n gets large

hess_off_diag_eval - hess_many_slices_eval

# I think there is numerical instability.

# Checking for symmetry of Hessians
hess_many_slices_eval' - hess_many_slices_eval
hess_off_diag_eval' - hess_off_diag_eval

########################################################################
#Testing the RoutingGradient and RoutingPointsHomotopy on this example:

r = RoutingGradient(F, projection_vars; c = c, B = B)
e = denominator_exponent(r)

p1 = zeros(k)
q1 = randn(k)
H = RoutingPointsHomotopy(r, p1, q1)

### Test evaluation
u = randn(ComplexF64, k)
x0 = randn(k)
t0 = 1.0
evaluate!(u, H, x0, t0)

u1 = randn(ComplexF64, k)
t1 = 0.0
evaluate!(u1, H, x0, t1)

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
p0 = evaluate(r, s0) #This returns a vector of NaNs. Upon further inspection, 
                     #track! is terminating with the code :terminated_invalid_startvalue_singular_jacobian
                     #Note that this is still happening after making the change of restricting to non-singular solutions in the witness set.

### Monodromy 
seed = rand(UInt32)
mon_result = monodromy_solve(
    MS,
    s0,
    p0,
    seed;
)
