"""

"""
function BCELoss_class_weighted(weights)
    if isempty(weights)
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
    _, indices = torch.topk(a, k)

    # Create a zero tensor of the same shape as a
    topk_tensor = torch.zeros_like(a)

    # Set only the top k elements to 1
    topk_tensor[indices] = 1
    return topk_tensor
end