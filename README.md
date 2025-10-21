
# Computing complements of real hypersurfaces using pseudowitness sets

This repository contains code for the for the project _Computing complements of real hypersurfaces using pseudowitness sets_ by Paul Breiding, John Cobb, Aviva Englander, Nayda Farnsworth, Jon Hauenstein, Oskar Henriksson, David Johnson, Jordy Garcia, and Deepak Mundayur.

## Example

Suppose that we want to study the complement of the discriminant for the quadratic polynomial `x^2 + a * x + b` with parameters `a` and `b`.

The first step is to get access to the functions of the package by running:

```julia
include("src/functions.jl");
```
We then set up the incidence variety of the discriminant:

```julia
@var a b x
F = System([x^2 + a * x + b; 2x + a], variables = [a, b, x])
```

We form the gradient of the routing function by writing:

```julia
r = RoutingGradient(F, [a; b])
```

We find the critical points (the exact output depends on a randomized choice of center for the routing function):

```julia-repl   
julia> pts = critical_points(r, options = options) |> real_solutions
4-element Vector{Vector{Float64}}:
 [-5.175022390506237, -7.4163002433454395]
 [2.7890571286291124, 7.76755857260763]
 [13.795193093096357, -0.1040935401458447]
 [-11.409227831219235, -4.510747011398911]
```

Finally, we connect the critical poitns that belong to the same component of the complement:

```julia
G, idx, failed_info  = partition_of_critical_points(r, pts)
```

From the first output, we see that the the three first points belong to the same connected component, and that the fourth one belongs to its own component (the exact output depends on randomized choices):

```julia-repl
julia> G
2-element Vector{Any}:
 [1, 3, 4]
 [2]
```

## Illustrations

The following pictures are created via the files `quadratic.jl` and `cubic_two_parameters.jl` in the `examples` directory.

<p style="text-align:center;"><img src="figures/example_quadratic.svg" height="200px"/><img src="figures/example_cubic.svg" height="200px"/></p>