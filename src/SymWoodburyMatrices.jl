module SymWoodburyMatrices

import Base:+,*,-,\,^,sparse

export Diag, SymWoodbury, Id, partialInv, liftFactor, getindex

ViewTypes   = Union{SubArray}
VectorTypes = Union{AbstractMatrix, Vector, ViewTypes}
MatrixTypes = Union{AbstractMatrix, Array{Real,2},
                    SparseMatrixCSC{Real,Integer}}

# ──────────────────────────────────────────────────────────────
#
# Diag
# Wrapper around Diagonal, extending its functionality.
# NOTE: we do not extend Diagonal to avoid contaminating the base
#
# ──────────────────────────────────────────────────────────────

type Diag <: AbstractMatrix{Real}
    diag::Vector
end

function +(A::MatrixTypes,B::Diag)
    O = copy(A);
    for i = 1:size(A,2); O[i,i] = A[i,i] + B.diag[i]; end; O
end

*(α::Real,B::Diag)             = Diag(α*B.diag)
*(B::Diag,α::Real)             = Diag(α*B.diag)
*(A::Diag,B::Diag)             = Diag(A.diag.*B.diag)
*(A::Diag,B::AbstractMatrix)           = A.diag.*B
+(A::Diag,B::Diag)             = Diag(A.diag + B.diag)
\(A::Diag,b::AbstractMatrix)           = size(b,2) == 1 ? A.diag.\b : nothing
+(B::Diag,A::MatrixTypes)      = A + B
^(A::Diag, n::Integer)         = Diag(A.diag.^n)
Base.full(A::Diag)             = full(Diagonal(A.diag))
Base.inv(A::Diag)              = Diag(1.0./A.diag)
*(A::Diag, b::ViewTypes)       = A.diag.*b;
Base.size(A::Diag, k::Integer) = (k == 1 || k == 2) ? size(A.diag,1) : 1
Base.size(A::Diag)             = (size(A.diag,1), size(A.diag,1))
Id(n::Integer)                 = Diag(ones(n))
Base.sparse(A::Diag)           = spdiagm(A.diag)
Base.getindex(A::Diag, i::Integer, j::Integer) = (i == j) ? A.diag[i] : 0

# ──────────────────────────────────────────────────────────────
#
# SymWoodbury
#
# ──────────────────────────────────────────────────────────────

using Base.LinAlg.BLAS:gemm!,gemm
"""
    SymWoodbury(A, B, D)

Represents a matrix of the form A + B*D*B'
"""
type SymWoodbury{TA, TB, TD}

  # Represents the Matrix
  # A + BDBᵀ
  # A is symmetric

  A::TA; B::TB; D::TD;

end

function Base.inv{TA<:Any, TB<:Any, TD<:Matrix}(O::SymWoodbury{TA, TB, TD})

  W = inv(O.A);
  X = W*O.B;
  invD = -1*inv(O.D);

  Z = inv(invD - O.B'*X);
  return SymWoodbury(W,X,Z);

end

function Base.inv{TA<:Any, TB<:Any, TD<:SparseMatrixCSC}(O::SymWoodbury{TA, TB, TD})

  W = inv(O.A);
  X = W*O.B;
  invD = -1*inv(full(O.D));

  Z = inv(invD - O.B'*X);
  return SymWoodbury(W,X,Z);

end

function partialInv(O::SymWoodbury)

  # Get the factors (X,Z) in W + XZXᵀ
  # where W + XZXᵀ = inv( A + BDBᵀ )

  if (size(O.B,2) == 0)
    return (0,0)
  end
  X = (O.A)\O.B;
  invD = -1*inv(O.D);

  Z = inv(invD - O.B'*X);
  return (X,Z);

end

Base.full(O::SymWoodbury) = O.A + O.B*O.D*O.B';
\(O::SymWoodbury, x::Union{VectorTypes, MatrixTypes}) = inv(O)*x;

"""
    liftFactor(A)

More stable version of inv(A). liftFactor(A)(x) is the same as inv(A)*x 

```julia
julia> liftFactor(A)(x)
```
"""
function liftFactor(O::SymWoodbury)

  n  = size(O.A,1)
  k  = size(O.B,2)

  Λ  = sparse(O.A)
  B  = sparse(O.B)
  Di = sparse(inv(O.D))
  M = [Λ    B   ;
       B'  -Di ];
  M = lufact(M)

  return x -> (M\[x; zeros(k,1)])[1:size(O,1),:];

end

function *{TA<:Any, TB<:Matrix, TD<:Any}(O::SymWoodbury{TA,TB,TD}, x::VectorTypes)

  o = O.A*x;
  w = O.D*gemm('T','N',O.B,x);
  gemm!('N','N',1.,O.B,w,1., o)
  return o

end

function *{TA<:Any, TB<:Any, TD<:Any}(O::SymWoodbury{TA,TB,TD}, x::VectorTypes)

  return O.A*x + O.B*(O.D*(O.B'x));

end

PTypes = Union{AbstractMatrix,Diag}

*(α::Real, O::SymWoodbury)        = SymWoodbury(α*O.A, O.B, α*O.D);
*(O::SymWoodbury, α::Real)        = SymWoodbury(α*O.A, O.B, α*O.D);
+(α::Real, O::SymWoodbury)        = α == 0 ?
                                    SymWoodbury(O.A, O.B, O.D) :
                                    SymWoodbury(O.A + α, O.B, O.D);
+(M::PTypes, O::SymWoodbury)      = SymWoodbury(O.A + M, O.B, O.D);
+(O::SymWoodbury, M::PTypes)      = SymWoodbury(O.A + M, O.B, O.D);
+(O::SymWoodbury, M::SymWoodbury) = SymWoodbury(O.A + M.A, [O.B M.B],
                                                cat([1,2],O.D,M.D) );
Base.size(M::SymWoodbury, i)      = size(M.B,1);
Base.size(M::SymWoodbury)         = size(M.A);

dsum(X,Y) = cat([1,2])

function ^(O::SymWoodbury, n::Integer)

  if n == 2
    A = O.A^2
    D = O.D
    B = O.B
    Z = [(O.A*B + B) (O.A*B - B) B]
    #return SymWoodbury(A, Z, full( dsum(dsum(D,-D)/2 , D*B'*B*D) ));
    return SymWoodbury(A, Z, full( cat([1,2],D/2,-D/2, D*B'*B*D) ) )
  else
    warning("Taking the matrix to a power greater than 2 is not supported.")
    return nothing
  end

end

conjm(O::SymWoodbury, M) = SymWoodbury(M*O.A*M', M*O.B, O.D);

Base.getindex(M::Diag, I::UnitRange, I2::UnitRange) =
  Diag(M.diag[I]);

Base.getindex(O::SymWoodbury, I::UnitRange, I2::UnitRange) =
  SymWoodbury(O.A[I,I], O.B[I,:], O.D);

Base.sparse(O::SymWoodbury) = sparse(full(O))

end