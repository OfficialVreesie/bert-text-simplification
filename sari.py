def get_ngrams(segment, n):
    return set(zip(*[segment[i:] for i in range(n)]))

def compute_precision(predicted_ngrams, reference_ngrams):
    if not predicted_ngrams:
        return 0.0
    return len(predicted_ngrams & reference_ngrams) / len(predicted_ngrams)

def compute_recall(predicted_ngrams, reference_ngrams):
    if not reference_ngrams:
        return 0.0
    return len(predicted_ngrams & reference_ngrams) / len(reference_ngrams)

def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_sari_components(source, prediction, reference):
    # Tokenize
    source_tokens = source.lower().split()
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    # Get n-grams (using unigrams for simplicity)
    source_ngrams = set(source_tokens)
    pred_ngrams = set(pred_tokens)
    ref_ngrams = set(ref_tokens)
    
    # Keep score
    keep_source = source_ngrams & ref_ngrams
    keep_pred = pred_ngrams & ref_ngrams
    keep_score = compute_f1(
        compute_precision(keep_pred, keep_source),
        compute_recall(keep_pred, keep_source)
    )
    
    # Delete score
    del_source = source_ngrams - ref_ngrams
    del_pred = source_ngrams - pred_ngrams
    del_score = compute_f1(
        compute_precision(del_pred, del_source),
        compute_recall(del_pred, del_source)
    )
    
    # Add score
    add_ref = ref_ngrams - source_ngrams
    add_pred = pred_ngrams - source_ngrams
    add_score = compute_f1(
        compute_precision(add_pred, add_ref),
        compute_recall(add_pred, add_ref)
    )
    
    return {
        'keep_score': keep_score * 100,
        'deletion_score': del_score * 100,
        'add_score': add_score * 100,
        'sari': (keep_score + del_score + add_score) * 100 / 3
    }