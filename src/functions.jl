using HomotopyContinuation, LinearAlgebra, DifferentialEquations, LightGraphs, ProgressMeter

const HC = HomotopyContinuation
const DE = DifferentialEquations


include("pseudo_witness_sets.jl")
include("gradient_cache.jl")
include("routing_functions.jl")
include("homotopy.jl")
include("ode_solving.jl")
include("graph.jl")
include("hypersurfaces.jl")
