# SymWoodburyMatrices

A symmetric version of `WoodburyMatrices`
```julia
using SymWoodburyMatrices
A = eye(4); B = ones(4,1); D = eye(1,1); b = ones(4,1)

# M = A + BDBᵀ
M = SymWoodbury(A, B, D)
```

It can be used like a normal matrix

```
julia> M*b
4x1 Array{Float64,2}:
 5.0
 5.0
 5.0
 5.0

julia> M\b
4x1 Array{Float64,2}:
 0.2
 0.2
 0.2
 0.2
```

etc.
