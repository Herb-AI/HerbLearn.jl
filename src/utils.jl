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

function PairwiseRankingLoss(margin=0.5)
    function loss(preds, target)
        target = target.squeeze()

        pos_preds = preds * target # mask by target == 1
        neg_preds = preds * (1-target) # mask by target == 0

        pos_preds = pos_preds.unsqueeze(2)  # [batch_size, n, 1]
        neg_preds = neg_preds.unsqueeze(1)  # [batch_size, 1, m]

        # Following contrastive loss of 
        # https://github.com/UKPLab/sentence-transformers/blob/839e58ac826fef6a6875f689633be8a883f5dec6/sentence_transformers/losses/ContrastiveLoss.py#L102
        pw_diff = pos_preds + torch.nn.functional.relu(margin - neg_preds) # pairwise diff
        # pw_diff = pos_preds - neg_preds # pairwise diff
        pw_diff = (pw_diff * target.unsqueeze(2) * (1-target.unsqueeze(1)))

        # println("non-zero indices: ", (pw_diff > 0).nonzero(as_tuple=false) + 1)

        # pw_diff = pw_diff.abs() # get absolute number of errors

        pw_diff = pw_diff.sum(dim=(1,2)) # sum up over matrix
        pw_diff = pw_diff/(target.sum(dim=1) * (1-target).sum(dim=1)) # normalize by number of relevant pairs 

        return -pw_diff
    end

    return loss
end

function misranked_pairs(preds, target)
    target = target.squeeze()

    pos_preds = preds * target # mask by target == 1
    neg_preds = preds * (1-target) # mask by target == 0

    pos_preds = pos_preds.unsqueeze(2)  # [batch_size, n, 1]
    neg_preds = neg_preds.unsqueeze(1)  # [batch_size, 1, m]

    pw_diff = pos_preds < neg_preds # pairwise comparison

    pw_diff = (pw_diff * target.unsqueeze(2) * (1-target.unsqueeze(1)))

    pw_diff = pw_diff.sum(dim=(1,2)) # sum up over matrix

    pw_diff = pw_diff/(target.sum(dim=1) * (1-target).sum(dim=1)) # normalize by number of relevant pairs 

    return pw_diff
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
    
function mask_top_k(predictions; top_k=10)
    println("predictions.shape $(predictions.shape)")
    @assert length(predictions.shape) == 2
    top_k_values, _ = torch.topk(predictions, k=top_k, dim=1)
    # Get k-th largest prediction in each row
    thresholds = top_k_values.min(dim=1)[1]
    # Threshold each row by the top-k-th value
    binary_matrix = (predictions >= thresholds.view(-1, 1)).int()
    return binary_matrix
end


"""
From BigCode library: https://github.com/bigcode-project/bigcode-encoder/blob/master/src/utils.py
"""
function pooling(x, mask)
    @assert length(x.shape) == 3

    B, T, F = x.shape
    eos_idx = mask.sum(dim=1, dtype=torch.long) - 1  # Shape: [B]
    batch_idx = torch.arange(B).view(B, 1)  # Shape: [B, 1]

    # Expand eos_idx to match feature dimension
    eos_idx = eos_idx.view(B, 1, 1).expand(-1, 1, F)  # Shape: [B, 1, F]

    # Use gather to select the last relevant feature vector
    mu = torch.gather(x, dim=1, index=eos_idx).squeeze(1) 

    return mu
end


"""
From BigCode library: https://github.com/bigcode-project/bigcode-encoder/blob/master/src/utils.py
"""
function pool_and_normalize(features_sequence, attention_masks, return_norms=false)
    # Temporal pooling of sequences of vectors and projection onto the unit sphere.
    pooled_embeddings = pooling(features_sequence, attention_masks)
    embedding_norms = pooled_embeddings.norm(dim=1)

    normalizing_factor = torch.where(  # Only normalize embeddings with norm > 1.0.
        embedding_norms > 1.0, embedding_norms, torch.ones_like(embedding_norms)
    )

    pooled_normalized_embeddings = pooled_embeddings / normalizing_factor.view(-1,1)

    if return_norms
        return pooled_normalized_embeddings, embedding_norms
    else
        return pooled_normalized_embeddings
    end
end

"""
Given vec1 of shape B1 x Lat and vec2 of shape B2 x Lat, calculates the pairwise cosine similarity of every pair of vectors in vec1, vec2. 
Returns the similarity matrix /
"""
function pairwise_cosine_sim(vec1, vec2)
    vec1 = torch.nn.functional.normalize(vec1, p=2, dim=1)
    vec2 = torch.nn.functional.normalize(vec2, p=2, dim=1)

    return torch.matmul(vec2, vec1.unsqueeze(dim=2)).squeeze()
end

function pairwise_l2_distance(vec1, vec2)
    # vec1: [B1, D], vec2: [B2, D]
    # Expand dims to allow broadcasting
    diff = vec1.unsqueeze(1) - vec2.unsqueeze(0)  # [B1, B2, D]
    return torch.norm(diff, p=2, dim=2)  # [B1, B2]
end

cos_mat(vec) = torch.nn.functional.cosine_similarity(vec.unsqueeze(1), vec.unsqueeze(0), dim=2)
l_mat(vec, p=1.0) = torch.cdist(vec, vec, p=p)/length(vec[1])