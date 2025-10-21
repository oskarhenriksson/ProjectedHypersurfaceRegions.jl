using Pkg, Random
Pkg.activate(".")

include("../src/functions.jl");

Random.seed!(0x8b868320)

########

# The discriminant of x^3 + a * x^2 + b*x + γ is
# 4*a^3*γ - a^2*b^2 - 18*a*b*γ + 4*b^3 + 27*γ^2
@var a b γ x
f = x^3 + a * x^2 + b * x + γ
F = System([f; differentiate(f, x)], variables=[a, b, γ, x])

projection_variables = [a; b; γ]
k = length(projection_variables)

B = qr(rand(k, k)).Q |> Matrix # not needed anymore (but kept for reproducibility)
c = 10 .* randn(k)
r = RoutingGradient(F, projection_variables; c=c)

# critical points
res = critical_points(r) # should be 22
pts = real_solutions(res)

### connecting 
G, idx, failed_info = partition_of_critical_points(r, pts)
G

# Analyze possible real root counts
for (i, comp) in enumerate(G)
    println("Component #$i")
    u0 = pts[first(comp)]
    println("Sample point: $u0")
    specialized_polynomial = subs(f, projection_variables => u0)
    res = HomotopyContinuation.solve([specialized_polynomial])
    real_pts = real_solutions(res)
    println("Number of real roots: $(length(real_pts))")
end

# g(x, param, t) = real(evaluate(r, x))
# tspan = (0.0, 1e4)
# for (i, u0) in enumerate(pts)
#     println()
#     println("Critical point #$i")
#     println("Point: $u0")
#     jac = real(evaluate_and_jacobian(r, u0)[2])
#     eigen_data = eigen(jac)
#     eigenvalues = eigen_data.values
#     eigenvectors = eigen_data.vectors
#     positive_directions = [i for (i, λ) in enumerate(eigenvalues) if real(λ) > 0]
#     println("Index: $(length(positive_directions))")
#     if  !isempty(positive_directions)
#         idx = first(positive_directions)
#         v = 1e-4*eigenvectors[:, idx]
#         prob1 = ODEProblem(g, u0 + v, tspan)
#         sol1 = DE.solve(prob1, reltol = 1e-6, abstol = 1e-6)
#         limit1 = last(sol1.u) 
#         closest_distance1, closest_point_idx1 = findmin(i->norm(pts[i] .- limit1), 1:length(pts))
#         if closest_distance1 > 1e-6
#             @warn "Missing critical point identified by gradient flow! Added to the list of points"
#             append!(pts, limit1)
#             closest_point_idx1 = length(pts)
#         end
#         prob2 = ODEProblem(g, u0 - v, tspan)
#         sol2 = DE.solve(prob2, reltol = 1e-6, abstol = 1e-6)
#         limit2 = last(sol2.u)
#         closest_distance2, closest_point_idx2 = findmin(i->norm(pts[i] .- limit2), 1:length(pts))
#         if closest_distance2 > 1e-6
#             @warn "Missing critical point identified by gradient flow! Added to the list of points"
#             append!(pts, limit2)
#             closest_point_idx2 = length(pts)
#         end
#         println("Connects the critical points: $closest_point_idx1 and $closest_point_idx2")
#     else
#         specialized_polynomial = subs(f, projection_variables=>u0)
#         res = HomotopyContinuation.solve([specialized_polynomial])
#         println("Number of real roots: $(length(real_solutions(res)))")
#     end
# end