# Test files
- [ ] Create test files and/or doctests. (The code is getting more complex, so we should probably start being a bit more careful with testing!)

# Computation of the critical points
- [ ] Implement caching for the single slice method for computing the Hessian (for instance by adapting the current `GradientCach`).
- [ ] Try implementing the choices of the Hessian for the abstract system and routing homotopies (also consider just numerically differentiating the gradient).

# Find connected components given critical points
- [ ] Get critical points, and identify which have index 1.
- [ ] Implement gradient flow. One option is to use the `DifferentialEquations` package; see `main.jl` for an exanple of this. Use https://github.com/JuliaAlgebra/HypersurfaceRegions.jl as inspiration.

# Find more examples 
- [ ] Examples of parametrized polynomial systems where the (maximal) number of real solutions are unknown, or where the topology of the complement of the discriminant is of interest.
- Problems from intersection theory and enumerative geometry. For instance, we should be able to find a cubic with 27 *real* lines (like the Clebsch surface). Are there other well known intersection theory results worth trying?

# Numerical stability issues
- When systems get large, the different methods of taking the hessian (off_diag, many_slices) start getting different answers. This is likely due to numerical instability.
- [x] One suggested way to improve this is to reparametrize the line from p + t b to tp - b so that our sum goes from -∑1/s_i to (do i add a negative here?)∑s_i. That is, I replace l = p + tb ---> t^{-1} l = t^{-1} p + b.
    - We tried this and it seems like the accuracy is a lot better. 
- [ ] Make this change everywhere!

# Known bugs
- [ ] The abstract sytem gives an error for the four-bus example (singular Jacobians are encountered). 