def run_deduplication(graph_store, threshold=95):
    """
    Finds candidates using rapidfuzz and merges them directly if they exceed a high threshold.
    Returns the number of merges performed.
    """
    candidates = graph_store.find_duplicate_candidates(threshold=threshold)
    merge_count = 0
    
    for id1, id2, score in candidates:
        if not graph_store.graph.has_node(id1) or not graph_store.graph.has_node(id2):
            continue
            
        print(f"[*] Merging '{id2}' into '{id1}' (fuzzy score: {score})")
        graph_store.merge_nodes(keep_id=id1, merge_id=id2)
        merge_count += 1
            
    if merge_count > 0:
        graph_store.save_to_disk()
        
    return merge_count
