using Herb.HerbSearch
using Herb.HerbGrammar
using Herb.HerbData
using Herb.HerbLearn

×(a, b) = a * b

g = @cfgrammar begin
    Real = |(0:3)
    Real = x
    Real = Real + Real
    Real = Real × Real
end

problem = Problem([IOExample(Dict(:x => x), 3x) for x ∈ 1:5], "example")

pretrain(g, problem, get_bfs_enumerator, :Real, 1000, 20)
