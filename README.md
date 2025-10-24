
# Computing complements of real hypersurfaces using pseudowitness sets

This repository contains code for the project _Computing complements of real hypersurfaces using pseudowitness sets_ by Paul Breiding, John Cobb, Aviva Englander, Nayda Farnsworth, Jon Hauenstein, Oskar Henriksson, David Johnson, Jordy Lopez Garcia, and Deepak Mundayur.

## Running the code 

If you're using Git, you can clone the repository by running the following command in your terminal:

```bash
gh repo clone oskarhenriksson/parametric_gradients
```

Alternatively, you can download the repository manually by clicking the green `Code` button at the top of the GitHub page and selecting `Download ZIP`.

Once you have the repository, the easiest way to run the code is to open Julia in the root folder, activate the environment, and (if this is your first time running the code) instantiate the dependencies:

```julia
using Pkg
Pkg.activate(".")  
Pkg.instantiate()
```

Once the environment is ready, load the functions of the package by including the main source file:

```julia
include("src/functions.jl")
```

You're now good to go! 🚀

## Example

Suppose that we want to study the complement of the discriminant for the quadratic polynomial 
```math
f_{a,b}(x)=x^2+ax+b
``` 
with parameters $a$ and $b$.

We start by setting up the incidence variety $`\{(a,b,x)\in ℂ^3\mid f_{a,b}(x)=f′_{a,b}(x)=0\}`$ of the discriminant:

```julia-repl
julia> @var a b x;
julia> F = System([x^2 + a * x + b, 2x + a], variables = [a, b, x]);
```

We form the gradient of the routing function by writing:

```julia-repl
julia> ∇r = RoutingGradient(F, [a, b]);
```

We find the critical points via the `critical_points` function (the exact output depends on the randomized choice of center point `c` that happens when the routing function is created):

```julia-repl   
julia> pts = critical_points(∇r) |> real_solutions
4-element Vector{Vector{Float64}}:
 [-5.175022390506237, -7.4163002433454395]
 [2.7890571286291124, 7.76755857260763]
 [13.795193093096357, -0.1040935401458447]
 [-11.409227831219235, -4.510747011398911]
```

Finally, we connect the critical points that belong to the same component of the complement:

```julia-repl
julia> G, idx, failed_info = partition_of_critical_points(∇r, pts);
```

The first output `G` describes the connected components. We see that the first, third and fourth critical points belong to the same connected component, and that the second one belongs to its own component:

```julia-repl
julia> G
2-element Vector{Any}:
 [1, 3, 4]
 [2]
```

## Illustrations

The following pictures are created via the files `quadratic.jl` and `cubic_two_parameters.jl` in the `examples` directory.

<p align="center"><img src="figures/example_quadratic.svg" height="200px"/><img src="figures/example_cubic.svg" height="200px"/></p>

## Dependencies
The code relies on the following Julia packages:
- `HomotopyContinuation.jl` (for numerical algebraic geometry)
- `DifferentialEquations.jl` (for gradient flow)
- `LightGraphs.jl` (for building the connectivity graph).