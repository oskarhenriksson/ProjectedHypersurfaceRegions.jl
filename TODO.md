# Computation of the critical points
- [ ] Debug the code in `routing_pts.jl`.
- [ ] Figure out a way to use `monodromy_solve` in HC for an abstract system (the gradient). Maybe Paul can look at this?
- [ ] Another idea is to mod out by the group action ahead of time, by computing the ring of invariants

# Find connected components given critical points
- [ ] Get critical points, and identify which have index 1
- [ ] Implement gradient flow (w/ DifferentialEquations)

# Computation of the Hessian
- [ ] Implement the 'single slice' method

# Find more examples 
- [ ] We are interested in examples where the (maximal) number of real solutions are unknown.
- Once we have the number of components of complement, can get 21 real lines on a cubic surface and find the Klebsch surface. Other well known intersection theory results?

# Determinants
- While computing large systems, a bottleneck before even computing the discriminant is computing large determinants symbolically.
- [ ] One faster way to do this: One can replace det(M(x)) = 0 with a null space condition by adding auxiliary variables v and adding the the equations M(x)*v = 0 and dot(rand(length(v), v) = 1. See the file `Wnt_network.jl`. See also [this paper](https://www3.nd.edu/~jhauenst/preprints/bhpsRankDecomp.pdf).

# Numerical Stability Issues
- When systems get large, the different methods of taking the hessian (off_diag, many_slices) start getting different answers. This is likely due to numerical instability.
- [x] One suggested way to improve this is to reparametrize the line from p + t b to tp - b so that our sum goes from -∑1/s_i to (do i add a negative here?)∑s_i. That is, I replace l = p + tb ---> t^{-1} l = t^{-1} p + b.
    - We tried this and it seems like the accuracy is a lot better. We should make this change everywhere.