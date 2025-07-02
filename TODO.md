# Computation of the critical points
- The method in `routing_pts.jl` involves introducing auxilary variables in such a way that each critical point gives rise to a whole orbit under a faithful action of $(S_d)^k$, which has very large cardinality, even for moderately sized problems. There are a couple of approaches we can use to deal with this:
- [ ] Use the built-in functionality in HC for doing monodromy modulo a group action. We have implemented a function `relabeling` for computing the orbits. However, the current system that we use for monodromy have extra parameters `V` and `W` (for ensuring that the incidence variety is irreducible), and for random values of these parameters, we lose the symmetry. We need a more clever way to obtain an irreducible incidence variety without breaking the symmetry!
- [ ] Mod out by the group action ahead of time, by computing the ring of invariants.
- [ ] Avoid the auxilary varibles altogether, by turning the gradient into an `AbstractSystem` (see the HC [documentation](https://www.juliahomotopycontinuation.org/HomotopyContinuation.jl/stable/systems/#Interface)), and apply `monodromy_solve` to find the critical points. 

# Find connected components given critical points
- [ ] Get critical points, and identify which have index 1.
- [ ] Implement gradient flow. One option is to use the `DifferentialEquations` package; see `main.jl` for an exanple of this.

# Computation of the Hessian
- [ ] Implement the 'single slice' method (see notes in the overleaf file and email from Jon).

# Find more examples 
- [ ] We are interested in examples where the (maximal) number of real solutions are unknown.
- Problems from intersection theory and enumerative geometry. For instance, we should be able to find a cubic with 27 *real* lines (like the Clebsch surface). Are there other well known intersection theory results worth trying?

# Determinants
- While computing large systems, a bottleneck before even computing the discriminant is computing the determinant of the Jacobian. 
- [ ] One faster way to do this: One can replace `det(M(x))` with a null space condition by adding auxiliary variables `v` and adding the the polynomials `M(x)*v` and `dot(rand(length(v), v) - 1` to the system. See the file `Wnt_network.jl` for an example of this, as well as [this paper](https://www3.nd.edu/~jhauenst/preprints/bhpsRankDecomp.pdf).

# Numerical stability issues
- When systems get large, the different methods of taking the hessian (off_diag, many_slices) start getting different answers. This is likely due to numerical instability.
- [x] One suggested way to improve this is to reparametrize the line from p + t b to tp - b so that our sum goes from -∑1/s_i to (do i add a negative here?)∑s_i. That is, I replace l = p + tb ---> t^{-1} l = t^{-1} p + b.
    - We tried this and it seems like the accuracy is a lot better. 
- [ ] Make this change everywhere!