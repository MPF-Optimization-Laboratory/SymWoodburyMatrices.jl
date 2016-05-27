using SymWoodburyMatrices
using Base.Test

n = 10; k = 3

A = SymWoodbury(Diag(rand(n)), randn(n,k), randn(k,k))
x = randn(n,1)

@test norm(A*x - full(A)*x) < 1e-10
@test norm(inv(A)*x - inv(full(A))*x) < 1e-10

A = SymWoodbury(Diag(rand(n)), sparse(randn(n,k)), sparse(randn(k,k)))
x = randn(n,1)

@test norm(A*x - full(A)*x) < 1e-10
@test norm(inv(A)*x - inv(full(full(A)))*x) < 1e-10

A = SymWoodbury(Diag(rand(n)), sparse(randn(n,k)), randn(k,k))
x = randn(n,1)

@test norm(A*x - full(A)*x) < 1e-10
@test norm(inv(A)*x - inv(full(full(A)))*x) < 1e-10


A = SymWoodbury(eye(n), sparse(randn(n,k)), randn(k,k))
x = randn(n,1)

@test norm(A*x - full(A)*x) < 1e-10
@test norm(inv(A)*x - inv(full(full(A)))*x) < 1e-10
@test norm(liftFactor(A)(x) - inv(full(full(A)))*x) < 1e-10

x = randn(n)
o = copy(x)
