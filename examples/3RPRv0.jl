# This is case 0 of Example 5.2 in the overleaf file
# Reference: https://arxiv.org/pdf/1903.06126

using Pkg, Random
Pkg.activate(".")

using Plots, ImplicitPlots
include("../src/functions.jl");
mkpath("./results/3RPRv0")

Random.seed!(12345)

time_start = time()

# Set up incidence variety of the discriminant
a2 = 1.4; a3 = 0.7; b3 = 1.0; A2 = 1.6; A3 = 0.9; B3 = 0.6; c3 = 1.0;

@var p[1:2] φ[1:2] c[1:2]

f = [ φ[1]^2 + φ[2]^2 - 1,
      p[1]^2 + p[2]^2 - 2*(a3*p[1] +b3*p[2])*φ[1] + 2*(b3*p[1] - a3*p[2])*φ[2] + a3^2 + b3^2 - c[1],
      p[1]^2 + p[2]^2 - 2*A2*p[1] + 2*((a2-a3)*p[1] - b3*p[2] + A2*a3 -A2*a2)*φ[1] + 2*(b3*p[1] + (a2-a3)*p[2] - A2*b3)*φ[2]
        + (a2-a3)^2 + b3^2 + A2^2 -c[2],
      p[1]^2 + p[2]^2 -2*(A3*p[1] + B3*p[2]) + A3^2 + B3^2 - c3
    ]

Jac = differentiate(f, [p; φ])
F = System([f; det(Jac)], variables=[p; φ; c]) # use System(prod(c)*[f;det(Jac)]) to impose c>0
projection_variables = c

# Set up routing function
center = 5*rand(length(projection_variables))

write_parameters("./results/3RPR/center.txt", center)
∇r = RoutingGradient(F, projection_variables; c = center);

# Degree of the discriminant
d = degree(∇r.PWS)
println("Degree of discriminant: $d")

# Routing points
options = MonodromyOptions(
    parameter_sampler = p -> 10 .* randn(ComplexF64, length(p)), # bigger loops
    max_loops_no_progress = 15 # change the stopping criterion
)
res, mon_res = critical_points(∇r, options = options)
pts = real_solutions(res)

write_parameters("./results/3RPR/monodromy_parameters.txt", parameters(mon_res))
write_solutions("./results/3RPR/monodromy_result.txt", solutions(mon_res)) 
write_solutions("./results/3RPR/result.txt", solutions(res))
write_solutions("./results/3RPR/routing_points.txt", pts)

# Connected components 
G, idx, failed_info = partition_of_critical_points(∇r, pts)
write("./results/3RPR/connected_components.txt", string(G))

t_end = time()
println("Computation time: $(t_end - time_start) seconds")

# Analyze root counts
S = System(f, variables = vcat(p,φ), parameters = projection_variables)
for (i, comp) in enumerate(G)
    println("Connected component #$i")
    real_root_counts = Int[]
    for j in comp
        real_steady_states = HC.solve(S, target_parameters=pts[j]) |> real_solutions
        rc = length(real_steady_states)
        push!(real_root_counts, rc)
    end
    println("Real root counts: $(real_root_counts)\n")
end



# Plotting
M_x_max = maximum(p -> p[1], pts)*1.1
M_x_min = minimum(p -> p[1], pts)*1.1
M_y_max = maximum(p -> p[2], pts)*1.1
M_y_min = minimum(p -> p[2], pts)*1.1

h(x, y) = 4907393223753094673156738281250000*x^12 - 19180099848067760467529296875000000*x^11*y + 52227962140007410049438476562500000*x^10*y^2 - 105419814598772792816162109375000000*x^9*y^3 + 180734414210959036064147949218750000*x^8*y^4 - 255164089507854075622558593750000000*x^7*y^5 + 313589204213794072723388671875000000*x^6*y^6 - 323551316987557786560058593750000000*x^5*y^7 + 290540772634933669853210449218750000*x^4*y^8 - 214216724620542226409912109375000000*x^3*y^9 + 131659687860658679580688476562500000*x^2*y^10 - 56617617861305358123779296875000000*x*y^11 + 17616458416794037055969238281250000*y^12 - 63804310917129323959350585937500000*x^11 + 100948227847214796066284179687500000*x^10*y - 54870882657549823379516601562500000*x^9*y^2 - 42993827946998667648315429687500000*x^8*y^3 - 194650071533785475189208984375000000*x^7*y^4 + 650525241674743005340576171875000000*x^6*y^5 - 1064046973156058177764892578125000000*x^5*y^6 + 991588319097624058502197265625000000*x^4*y^7 - 497309537323478430587768554687500000*x^3*y^8 - 238419796577618067764282226562500000*x^2*y^9 + 400462497966007230880737304687500000*x*y^10 - 229878684697759124496459960937500000*y^11 + 400794127545709875255584716796875000*x^10 - 261368244455786910768127441406250000*x^9*y + 284829825638109590214691162109375000*x^8*y^2 - 3299484839005116226849365234375000000*x^7*y^3 + 10377372735896146624700622558593750000*x^6*y^4 - 13879270391529530512128295898437500000*x^5*y^5 + 8650359305735387429847106933593750000*x^4*y^6 - 1073080181350434040872802734375000000*x^3*y^7 + 585792655893876558129730224609375000*x^2*y^8 - 2180028602285135523756408691406250000*x*y^9 + 1513581127313569197477264404296875000*y^10 - 949952490312135636477447509765625000*x^9 - 2988319719594917108336236572265625000*x^8*y + 9589553501421337651869653320312500000*x^7*y^2 - 6082177225822511391773120117187500000*x^6*y^3 - 7216569614691962938207946777343750000*x^5*y^4 + 21257508635081070236676232910156250000*x^4*y^5 - 31590496690689299126005151367187500000*x^3*y^6 + 19106469512000032596833715820312500000*x^2*y^7 + 1789499126137386007183392333984375000*x*y^8 - 4418336507237268032638092041015625000*y^9 + 1269803711430716910284870880126953125*x^8 + 5940971678376923631103842773437500000*x^7*y - 3768165821150253996521217895507812500*x^6*y^2 - 39169525361234826621874666992187500000*x^5*y^3 + 55915288366046798534714090515136718750*x^4*y^4 + 3991955692355840056469493164062500000*x^3*y^5 - 32571041670076771492820385864257812500*x^2*y^6 + 159202135093307157416206054687500000*x*y^7 + 6623896510473469175203642364501953125*y^8 - 795467880036586265553934877929687500*x^7 - 6761109800114637459605675688476562500*x^6*y + 11104605699689882282491155629882812500*x^5*y^2 + 6062489983627144175148372436523437500*x^4*y^3 - 24837922820456817907215162875976562500*x^3*y^4 + 22293772004755734075929013442382812500*x^2*y^5 + 1314151632605675952951939624023437500*x*y^6 - 5262569258765093200799872690429687500*y^7 + 255072918120546488123453265332031250*x^6 + 611645890761538053135463527539062500*x^5*y + 6315304275305556340506253544042968750*x^4*y^2 - 13370027439709660417187130711328125000*x^3*y^3 + 8514006532626860695898524494042968750*x^2*y^4 - 3123140741459884933236736097460937500*x*y^5 + 2086917872429229376097224565332031250*y^6 - 45115379802951792841244445117187500*x^5 + 64079685713219218242329796164062500*x^4*y - 856172925008586187884522586171875000*x^3*y^2 - 368793128845054112097026461671875000*x^2*y^3 + 198036846499900222544023566914062500*x*y^4 - 357001125810523439288288238867187500*y^5 + 4653701410818548623751462842890625*x^4 - 10098394569379416719553869985937500*x^3*y + 30198078740252901201026961562343750*x^2*y^2 + 10029398096993220495436996484062500*x*y^3 + 29676363554048718056005099757890625*y^4 - 291700212229534079704309860525000*x^3 + 438186468835634625289215712225000*x^2*y - 1031204646494982874490355640575000*x*y^2 - 1107599993232982647480477046925000*y^3 + 11859558340541924385576470120500*x^2 - 14063349975844306476647002827000*x*y + 6002095164453061895113993784500*y^2 - 345581942040182647207875885600*x + 374422817085824950639786616800*y + 6141659813187778276739630132
e = ∇r.e
R(x, y) = log(abs(h(x,y) / (1 + (x-center[1])^2 + (y-center[2])^2)^e))

contour(
    (M_x_min):0.01:M_x_max,
    (M_y_min):0.01:M_y_max,
    R,
    levels = 50,
    color = :plasma,
    clabels = false,
    cbar = false,
    lw = 1,
    fill = true,
)

implicit_plot!(
    h; 
    xlims = (M_x_min, M_x_max),
    ylims = (M_y_min, M_y_max),
    linecolor = :black,
    linewidth = 2,
    label = "discriminant",
	legend = false,
    resolution=3000
)

pts1 = pts[idx .!= 0]
g(x, param, t) = real(evaluate(∇r, x))
tspan = (0.0, 1e4)
for u0 in pts1
	jac = real(evaluate_and_jacobian(∇r, u0)[2])
	eigen_data = LinearAlgebra.eigen(jac)
	eigenvalues = eigen_data.values
	eigenvectors = eigen_data.vectors
	positive_directions = [i for (i, λ) in enumerate(eigenvalues) if real(λ) > 0]
	j = first(positive_directions)
	v = eigenvectors[:, j]

	prob = ODEProblem(g, u0 + 0.01*v, tspan)
	sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
	flow = Tuple.(sol.u)
	l = length(flow)
	k = div(l, 3)
	plot!(flow[1:k], linecolor = :steelblue, linewidth = 2, label = false, arrow = :closed)
	plot!(flow[k:end], linecolor = :steelblue, linewidth = 2, label = false)
	prob = ODEProblem(g, u0 - 0.01*v, tspan)
	sol = DE.solve(prob, reltol = 1e-6, abstol = 1e-6)
	flow = Tuple.(sol.u)
	l = length(flow)
	k = div(l, 3)
	plot!(flow[1:k], linecolor = :steelblue, linewidth = 2, label = false, arrow = :closed)
	plot!(flow[k:end], linecolor = :steelblue, linewidth = 2, label = false)
end

palette = collect(range(colorant"darkgreen", stop=colorant"lightgreen", length=length(G)))
for (i, component) in enumerate(G)
    scatter!(Tuple.(pts[component]), markercolor = palette[i], markersize = 3, label = "Critical points in region $i")
end

plot!(; legend = true, dpi=400, legendfontsize=6)

savefig("./figures/3RPR.svg")
savefig("./figures/3RPR.png")

# Try another round of monodromy (only if you think the first attempt missed solutions)
println("Running second round of monodromy...")
old_number_of_monodromy_solutions = length(solutions(mon_res))
options = MonodromyOptions(
    parameter_sampler = p -> 100 .* randn(ComplexF64, length(p)), # bigger loops
    max_loops_no_progress = 15 # change the stopping criterion
)
res, mon_res = critical_points(∇r, solutions(mon_res), parameters(mon_res), options = options)
if length(solutions(mon_res)) > old_number_of_monodromy_solutions
    println("Found new solutions with additional monodromy round!")
else
    println("No new solutions found in the additional monodromy round.")
end

# If new solutions were found, repeat the steps above manually!

