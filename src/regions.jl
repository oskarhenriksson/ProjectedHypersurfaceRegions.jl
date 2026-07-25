export partition_of_critical_points

export return_code,
    regions,
    components,
    morse_indices,
    failed_info,
    nregions

export Region,
    routing_points,
    euler_characteristic,
    number

export membership

### This is adapted from https://github.com/JuliaAlgebra/HypersurfaceRegions.jl/blob/main/src/partition.jl


# return index and unstable eigenvectors of the hessian 
function index_unstable_eigenvector!(u, U, ∇r, a)
    evaluate_and_jacobian!(u, U, ∇r, a)
    if is_almost_singular(U)
        flag = true
        println("The Hessian is almost singular for", a)
        return nothing, nothing, flag
    else
        flag = false
    end
    eigvals, eigvecs = LinearAlgebra.eigen!(real(U))
    eigvals = eigvals
    index = sum(eigvals .> 0)
    unstable_eigenvectors = eigvecs[:, eigvals.>0]
    return index, unstable_eigenvectors, flag
end

function is_almost_singular(matrix::Matrix; threshold = 1e10)
    condition_number = LinearAlgebra.cond(matrix)
    return condition_number > threshold
end

## calculate the index and unstable eigenvectors of the critical points
function _index_list(∇r, crit_pts)
    index_list::Vector{Int} = []
    unstable_vector_list::Vector{Matrix{Float64}} = []
    flag_prime = false
    k = size(∇r, 1)
    u = randn(ComplexF64, k)
    U = zeros(ComplexF64, k, k)

    for a in crit_pts
        index, unstable_eigenvectors, flag = index_unstable_eigenvector!(u, U, ∇r, a)
        if flag == true
            flag_prime = true
            return nothing, nothing, flag_prime
        end

        push!(index_list, index)
        push!(unstable_vector_list, unstable_eigenvectors)
    end

    return index_list, unstable_vector_list, flag_prime
end


# input a list, output a list of indices that have the same value in the list
function partition_indices(lst)
    regions = Dict()
    for (index, element) in enumerate(lst)
        if haskey(regions, element)
            push!(regions[element], index)
        else
            regions[element] = [index]
        end
    end
    return regions
end


@doc raw"""
    partition_of_critical_points(
        r::RoutingFunction,
        crit_pts::AbstractVector{<:AbstractVector{<:Real}},
        epsilon::Float64 = 1e-6,
        reltol::Float64 = 1e-6,
        abstol::Float64 = 1e-9,
    )

Partition a collection `crit_pts` of critical points of a routing function `r` into connected components via gradient flow.
The function returns a [`PartitionResult`](@ref) containing the connected components, Morse indices, and any failed connection attempts.  
"""
function partition_of_critical_points(
    r::RoutingFunction,
    crit_pts::AbstractVector{<:AbstractVector{<:Real}},
    epsilon::Float64 = 1e-6,
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-9,
)

    ∇r = RoutingGradient(r)

    index_list, unstable_eigenvector_list, flag_prime = _index_list(∇r, crit_pts)
    if flag_prime == true
        @warn "The Hessian is almost singular for some critical points"
        return PartitionResult(Vector{Vector{Int}}(), nothing, [], :singular_hessian)
    end


    ode_log! = set_up_ode(∇r)

    graph = LightGraphs.SimpleGraph(length(crit_pts))
    connectivity_status = zeros(Int, length(crit_pts))
    critical_points_indices = collect(1:length(crit_pts))

    # count the number of index 0 critical points
    count_index_0 = sum([index == 0 for index in index_list])

    if count_index_0 == 1
        # we do not need to do any path tracking in this case
        return PartitionResult([critical_points_indices], index_list, [], :success)
    end

    failed_info_list = []
    ProgressMeter.@showprogress for i = 1:length(index_list)
        if connectivity_status[i] == 0 && index_list[i] == 1
            # need to do path tracking in two directions
            critical_point_index = critical_points_indices[i]
            unstable_eigenvector = unstable_eigenvector_list[critical_point_index]
            pair_pos, failed_info_pos = limit_critical_point_from_critical_point(
                ode_log!,
                crit_pts,
                i,
                index_list,
                unstable_eigenvector,
                epsilon,
                reltol,
                abstol,
            )
            if isempty(failed_info_pos)
                LightGraphs.add_edge!(graph, pair_pos[1], pair_pos[2])
            else
                push!(
                    failed_info_list,
                    [critical_point_index, epsilon, unstable_eigenvector, failed_info_pos],
                )
            end

            pair_neg, failed_info_neg = limit_critical_point_from_critical_point(
                ode_log!,
                crit_pts,
                i,
                index_list,
                unstable_eigenvector,
                -epsilon,
                reltol,
                abstol,
            )

            if isempty(failed_info_neg)
                LightGraphs.add_edge!(graph, pair_neg[1], pair_neg[2])
            else
                push!(
                    
                    failed_info_list,
                    [critical_point_index, -epsilon, unstable_eigenvector, failed_info_neg],
                )
            end
            connectivity_status[i] = 1
        end
    end

    ProgressMeter.@showprogress for i = 1:length(index_list)
        if connectivity_status[i] == 0 && index_list[i] > 1
            # track paths and stop whenever one path converges
            critical_point_index = critical_points_indices[i]
            sub_failed_info_list = []
            for v in eachcol(unstable_eigenvector_list[critical_point_index])
                pair, failed_info = limit_critical_point_from_critical_point(
                    ode_log!,
                    crit_pts,
                    i,
                    index_list,
                    v,
                    epsilon,
                    reltol,
                    abstol,
                )
                if isempty(failed_info)
                    LightGraphs.add_edge!(graph, pair[1], pair[2])
                    connectivity_status[i] = 1
                    sub_failed_info_list = []
                    break
                else
                    push!(
                        sub_failed_info_list,
                        [critical_point_index, epsilon, v, failed_info],
                    )
                end

                pair, failed_info = limit_critical_point_from_critical_point(
                    ode_log!,
                    crit_pts,
                    i,
                    index_list,
                    v,
                    -epsilon,
                    reltol,
                    abstol,
                )
                if isempty(failed_info)
                    LightGraphs.add_edge!(graph, pair[1], pair[2])
                    connectivity_status[i] = 1
                    sub_failed_info_list = []
                    break
                else
                    push!(
                        sub_failed_info_list,
                        [critical_point_index, epsilon, v, failed_info],
                    )
                end
            end

            if !isempty(sub_failed_info_list)
                push!(failed_info_list, sub_failed_info_list)
            end
        end
    end


    partition = LightGraphs.connected_components(graph)
    partition_critical_point_indices = Vector{Int}[]
    for par in partition
        push!(partition_critical_point_indices, @view(critical_points_indices[par]))
    end
    code = isempty(failed_info_list) ? :success : :partial_success
    return PartitionResult(partition_critical_point_indices, index_list, failed_info_list, code)
end

partition_of_critical_points(
    r::RoutingFunction,
    result::RoutingPointsResult,
    args...;
    kwargs...,
) = partition_of_critical_points(r, routing_points(result), args...; kwargs...)


@doc raw"""
    PartitionResult

Result returned by [`partition_of_critical_points`](@ref). Use
[`components`](@ref) for connected components, [`morse_indices`](@ref) for the
critical point indices, and [`failed_info`](@ref) for failed connection attempts.
"""
struct PartitionResult{P,I,F}
    components::P
    morse_indices::I # TODO: this is a strange output to expose to users, since it is indexing yet another list you must reference. Considering changing this to a Dict or something.
    failed_info::F
    return_code::Symbol
end

"""
    components(result::PartitionResult)

Return the connected components as vectors of routing point indices.

See also [`regions`](@ref), which wraps each component in a [`Region`](@ref)
carrying the routing point coordinates, Morse indices and Euler characteristic.
"""
components(R::PartitionResult) = R.components

@doc raw"""
    morse_indices(result::PartitionResult)

Return the Morse index computed for each routing point, or `nothing` if the
partition failed before indices were available.
"""
morse_indices(R::PartitionResult) = R.morse_indices

@doc raw"""
    failed_info(result::PartitionResult)

Return information collected from failed connection attempts.
"""
failed_info(R::PartitionResult) = R.failed_info

@doc raw"""
    return_code(result::PartitionResult)

Return a symbolic status code for a partition result.
"""
return_code(R::PartitionResult) = R.return_code

@doc raw"""
    nregions(result::PartitionResult)

Return the number of connected components in the partition result.
"""
nregions(R::PartitionResult) = length(components(R))

function Base.show(io::IO, R::PartitionResult)
    npars = nregions(R)
    nfailures = length(failed_info(R))
    header = "PartitionResult with $npars connected components"
    println(io, header)
    println(io, "="^length(header))
    println(io, "• return_code → :$(return_code(R))")
    println(io, "• $(nfailures) failed path(s)")
    print(io, "• morse_indices → ", isnothing(morse_indices(R)) ? "not computed" : length(morse_indices(R)))
end


### Adapted from the `Region` struct in
### https://github.com/JuliaAlgebra/HypersurfaceRegions.jl/blob/main/src/output.jl

@doc raw"""
    Region

A single region of a [`RoutingFunction`](@ref) `r`, i.e. one connected component
of critical points as computed by [`partition_of_critical_points`](@ref).
Modelled after the `Region` struct in
[HypersurfaceRegions.jl](https://github.com/JuliaAlgebra/HypersurfaceRegions.jl).

A `Region` bundles the routing points lying in the region together with their
Morse indices, the Euler characteristic of the region, and a region number. The
routing points are the **real** critical points of `r` (a subset of its complex
critical points). The routing function `r` is stored so that a point in the
parameter space can be flowed to a routing point in a membership test.

# Fields
- `routing_points::Vector{Vector{Float64}}`: the routing points in the region.
- `morse_indices::Vector{Int}`: the Morse index of each routing point.
- `χ::Int`: the Euler characteristic of the region, `∑ᵢ (-1)^μᵢ`.
- `r::RoutingFunction`: the routing function whose routing points these are.
- `region_number::Int`: the index of this region within the partition.

See also [`routing_points`](@ref), [`morse_indices`](@ref),
[`euler_characteristic`](@ref) and [`number`](@ref).
"""
struct Region
    routing_points::Vector{Vector{Float64}}
    morse_indices::Vector{Int}
    χ::Int
    r::RoutingFunction
    region_number::Int
end

@doc raw"""
    Region(routing_points, morse_indices, r::RoutingFunction, region_number)

Construct a [`Region`](@ref) from its routing points and their Morse indices,
computing the Euler characteristic `χ = ∑ᵢ (-1)^μᵢ` automatically.
"""
function Region(
    routing_points::AbstractVector{<:AbstractVector{<:Real}},
    morse_indices::AbstractVector{<:Integer},
    r::RoutingFunction,
    region_number::Integer,
)
    @assert length(routing_points) == length(morse_indices) "There must be one Morse index per routing point"
    χ = sum(μ -> (-1)^μ, morse_indices; init = 0)
    Region(
        [Float64.(c) for c in routing_points],
        collect(Int, morse_indices),
        χ,
        r,
        Int(region_number),
    )
end

@doc raw"""
    routing_points(C::Region)

Return the routing points (real critical points) of the routing function that
lie in the region `C`.
"""
routing_points(C::Region) = C.routing_points

@doc raw"""
    morse_indices(C::Region)

Return the Morse index of each critical point in the region `C`.
"""
morse_indices(C::Region) = C.morse_indices

@doc raw"""
    euler_characteristic(C::Region)

Return the Euler characteristic `∑ᵢ (-1)^μᵢ` of the region `C`.
"""
euler_characteristic(C::Region) = C.χ

@doc raw"""
    number(C::Region)

Return the number identifying the region `C` within its partition.
"""
number(C::Region) = C.region_number

function Base.show(io::IO, C::Region)
    header = "Region $(number(C))"
    println(io, header)
    println(io, "="^length(header))
    println(io, "• $(length(routing_points(C))) routing point(s)")
    println(io, "• morse_indices → ", morse_indices(C))
    print(io, "• χ → ", euler_characteristic(C))
end

@doc raw"""
    regions(R::PartitionResult, r::RoutingFunction, crit_pts)

Build the vector of [`Region`](@ref) objects described by a [`PartitionResult`](@ref)
`R`, which must have been obtained from `partition_of_critical_points(r, crit_pts)`.

Each connected component of `R` becomes one `Region`, carrying the coordinates of
its routing points (taken from `crit_pts`), their Morse indices, the Euler
characteristic, and the routing function `r`. The routing function is stored on
each region so that the resulting vector is self-contained and can be passed
directly to [`membership`](@ref).

"""
function regions(
    R::PartitionResult,
    r::RoutingFunction,
    crit_pts::AbstractVector{<:AbstractVector{<:Real}},
)
    mi = morse_indices(R)
    components = R.components
    if isnothing(mi)
        @assert isempty(components) "Cannot build regions: Morse indices were not computed"
        return Region[]
    end
    map(enumerate(components)) do (i, component)
        Region([crit_pts[j] for j in component], [mi[j] for j in component], r, i)
    end
end

### Adapted from the membership test in
### https://github.com/JuliaAlgebra/HypersurfaceRegions.jl/blob/main/src/membership.jl

@doc raw"""
    membership(regions::Vector{Region}, p; reltol = 1e-6, abstol = 1e-9)

Determine which [`Region`](@ref) a point `p` in the parameter space belongs to.

The point `p` is flowed by gradient ascent of the routing function until it
reaches a routing point; the region whose routing points contain that limit is
returned. All regions in `regions` are assumed to come from the same partition,
i.e. to share a common routing function (the one stored on the first region).

Returns `nothing` if `regions` is empty or if the gradient flow does not converge
to one of the known routing points.

Options:
* `reltol = 1e-6`, `abstol = 1e-9`: parameters for the accuracy of the ODE solver.
"""
function membership(
    regions::AbstractVector{Region},
    p::AbstractVector{<:Real};
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-9,
)
    isempty(regions) && return nothing

    r = first(regions).r
    ∇r = RoutingGradient(r)
    ode_log! = set_up_ode(∇r)

    # flatten the routing points of all regions into a single list
    all_routing_points = reduce(vcat, routing_points(C) for C in regions)

    routing_point_index, _ =
        limit_critical_point(ode_log!, Float64.(p), all_routing_points, reltol, abstol)

    if routing_point_index == -1
        return nothing
    end

    end_routing_point = all_routing_points[routing_point_index]
    for C in regions
        if end_routing_point in routing_points(C)
            return C
        end
    end
    return nothing
end
