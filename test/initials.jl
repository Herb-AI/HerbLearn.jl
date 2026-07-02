# Shared fixture: the "initials" running example used across the test suite
# (and in the README). Abbreviate a name: "Ada Lovelace" -> "A.L."

module Initials
first_char(s) = string(first(s))
word(s, i) = String(split(s)[i])
concat(a, b) = a * b
end

const INITIALS_GRAMMAR = @csgrammar begin
    S = x
    S = "."
    S = concat(S, S)
    S = first_char(S)
    S = word(S, N)
    N = 1
    N = 2
end

# Rule indices, for readable tests.
const R_X, R_DOT, R_CONCAT, R_FIRST, R_WORD, R_ONE, R_TWO = 1, 2, 3, 4, 5, 6, 7

const INITIALS_PROBLEM = Problem([
    IOExample(Dict(:x => "Ada Lovelace"), "A.L."),
    IOExample(Dict(:x => "Alan Turing"), "A.T."),
])

# An easier target in the same grammar: just the first initial with a dot.
const FIRST_INITIAL_PROBLEM = Problem([
    IOExample(Dict(:x => "Ada Lovelace"), "A."),
    IOExample(Dict(:x => "Alan Turing"), "A."),
])

# Inputs shared by the learned methods' training in tests.
const INITIALS_INPUTS = [ex.in for ex in INITIALS_PROBLEM.spec]

solves(program, problem) = begin
    ev = Evaluator(INITIALS_GRAMMAR, problem.spec; mod=Initials)
    is_solution(ev, ev(program))
end
