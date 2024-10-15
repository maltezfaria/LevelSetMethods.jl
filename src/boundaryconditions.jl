"""
    abstract type BoundaryCondition

Singleton types used in [`applybc!`](@ref) for dispatch purposes.

Boundary conditions are imposed by a *ghost node* method, meaning that meshes
are composed of *internal nodes*, and *ghost nodes* which sole purpose is to
reinforce a given boundary condition.
"""
abstract type BoundaryCondition end

"""
    num_ghost_points(bc::BoundaryCondition)

The number of ghost-points on which to apply `bc`.
"""
function num_ghost_points end

"""
    applybc!(mf,bc::BoundaryCondition,[dim])

Overwrite the ghost nodes of `mf` do impose the boundary condition `bc` on the
dimension `dim`. If no `dim` argument is passed, the boundary condition is
applied to all dimensions of `mf`.
"""
function applybc! end

function applybc!(mf, bc::BoundaryCondition)
    N = dimension(mf)
    for dim in 1:N
        applybc!(mf, bc, dim)
    end
    return mf
end

"""
    interior_indices(mesh,bc::BoundaryCondition)

Given a `mesh` and a `bc`, return the indices to iterate over the interior nodes
of the `mesh`.
"""
function interior_indices end

function interior_indices(g::CartesianGrid, bc::BoundaryCondition)
    P  = num_ghost_points(bc)
    N  = dimension(g)
    sz = size(g)
    I  = ntuple(N) do dim
        return P+1:sz[dim]-P
    end
    return CartesianIndices(I)
end

"""
    struct PeriodicBC{P}

A periodic boundary condition with `P` periodic nodes.
"""
struct PeriodicBC{N} <: BoundaryCondition end

PeriodicBC(n::Int) = PeriodicBC{n}()

num_ghost_points(bc::PeriodicBC{N}) where {N} = N

function applybc!(mf, bc::PeriodicBC{P}, dir) where {P}
    @assert mesh(mf) isa CartesianGrid "boundary condition $(typeof(bc)) on $(typeof(mesh(mf))) not supported"
    ϕ        = values(mf)
    I⁻r, I⁺r = _index_read(mf, bc, dir)
    I⁻w, I⁺w = _index_write(mf, bc, dir)
    @views copy!(ϕ[I⁻w], ϕ[I⁺r])
    @views copy!(ϕ[I⁺w], ϕ[I⁻r])
    return mf
end

"""
    struct NeumannBC{P}

A homogeneours Neumann boundary condition with `P` nodes.

# TODO: what should this do for `P>1`?
"""
struct NeumannBC{N} <: BoundaryCondition end
NeumannBC(n::Int) = NeumannBC{n}()

num_ghost_points(bc::NeumannBC{N}) where {N} = N

## Helper functions

function _index_write(mf, bc::BoundaryCondition, dir)
    P  = num_ghost_points(bc)
    sz = size(mf)
    N  = length(sz)
    Il = ntuple(N) do dim
        if dim == dir
            1:P
        else
            1:sz[dim]
        end
    end
    Ir = ntuple(N) do dim
        if dim == dir
            (sz[dim]-P+1):sz[dim]
        else
            1:sz[dim]
        end
    end
    return CartesianIndices(Il), CartesianIndices(Ir)
end

function _index_read(mf, bc::BoundaryCondition, dir)
    P  = num_ghost_points(bc)
    sz = size(mf)
    N  = length(sz)
    Il = ntuple(N) do dim
        if dim == dir
            (P+1):2P
        else
            1:sz[dim]
        end
    end
    Ir = ntuple(N) do dim
        if dim == dir
            (sz[dim]-2P+1):(sz[dim]-P)
        else
            1:sz[dim]
        end
    end
    return CartesianIndices(Il), CartesianIndices(Ir)
end
