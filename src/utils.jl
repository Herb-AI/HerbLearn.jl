"""

"""
function BCELoss_class_weighted(weights)
    if isnothing(weights)
        weights = torch.ones((2,))/2
    end
    function loss(input, target)
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.sum(bce)
    end

    return loss
end

function top_k_ranking(vec, k=3)
    # Get the indices of the top k elements
    _, indices = torch.topk(vec, k)

    # Create a zero tensor of the same shape as vec
    topk_tensor = torch.zeros_like(vec)

    # Set only the top k elements to 1
    topk_tensor[indices] = vec[indices]
    return topk_tensor
end

function merge_grammars_function_save!(merge_to::AbstractGrammar, merge_from::AbstractGrammar)
    merge_to_rules = [string(rule) for rule in merge_to.rules]   
	for i in eachindex(merge_from.rules)
		expression = :($(merge_from.types[i]) = $(merge_from.rules[i]))
        if string(merge_from.rules[i]) ∉ merge_to_rules
            add_rule!(merge_to, expression)
        end
	end
end
