using HomotopyContinuation, LinearAlgebra, Plots, DifferentialEquations

const HC = HomotopyContinuation
const DE = DifferentialEquations

include("pseudo_witness_sets.jl")
include("grad_r_p.jl")
include("routing_pts.jl")