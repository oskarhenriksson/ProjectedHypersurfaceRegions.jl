using HomotopyContinuation, LinearAlgebra, DifferentialEquations

const HC = HomotopyContinuation
const DE = DifferentialEquations


include("pseudo_witness_sets.jl")
include("slice_gradient_hessian.jl")
#include("routing_points.jl")
include("slice_system.jl") 
include("homotopy.jl")
