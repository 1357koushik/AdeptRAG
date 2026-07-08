import os
import networkx as nx

class GraphStore:
    def __init__(self, storage_path="workspace/knowledge_graph.graphml"):
        self.storage_path = storage_path
        self.graph = nx.DiGraph()
        
        # Load existing graph if it exists
        if os.path.exists(self.storage_path):
            try:
                self.graph = nx.read_graphml(self.storage_path)
            except Exception as e:
                print(f"Failed to load existing graph: {e}")

    def upsert_entity(self, name: str, entity_type: str, description: str):
        # Normalize name for node ID
        node_id = name.strip().lower()
        
        if self.graph.has_node(node_id):
            # Entity exists, append the new description if not duplicate
            existing_desc = self.graph.nodes[node_id].get("description", "")
            if description not in existing_desc:
                new_desc = existing_desc + "\n---\n" + description
                self.graph.nodes[node_id]["description"] = new_desc
        else:
            self.graph.add_node(
                node_id, 
                name=name.strip(), 
                entity_type=entity_type.strip(), 
                description=description.strip()
            )

    def upsert_relation(self, source: str, target: str, keywords: str, description: str):
        source_id = source.strip().lower()
        target_id = target.strip().lower()
        
        # Ensure nodes exist
        if not self.graph.has_node(source_id):
            self.upsert_entity(source, "Unknown", "Implicitly created from relation.")
        if not self.graph.has_node(target_id):
            self.upsert_entity(target, "Unknown", "Implicitly created from relation.")
            
        if self.graph.has_edge(source_id, target_id):
            existing_desc = self.graph.edges[source_id, target_id].get("description", "")
            if description not in existing_desc:
                new_desc = existing_desc + "\n---\n" + description
                self.graph.edges[source_id, target_id]["description"] = new_desc
                
            # Merge keywords
            existing_kw = self.graph.edges[source_id, target_id].get("keywords", "")
            merged_kw = list(set([k.strip() for k in existing_kw.split(",")] + [k.strip() for k in keywords.split(",")]))
            self.graph.edges[source_id, target_id]["keywords"] = ", ".join(filter(bool, merged_kw))
        else:
            self.graph.add_edge(
                source_id, 
                target_id, 
                keywords=keywords.strip(), 
                description=description.strip()
            )

    def find_duplicate_candidates(self, threshold=95):
        from rapidfuzz import fuzz, utils
        nodes = list(self.graph.nodes(data=True))
        candidates = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                id1, data1 = nodes[i]
                id2, data2 = nodes[j]
                
                name1 = data1.get("name", id1)
                name2 = data2.get("name", id2)
                
                score = fuzz.token_sort_ratio(name1, name2, processor=utils.default_process)
                if score >= threshold:
                    candidates.append((id1, id2, score))
                    
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    def merge_nodes(self, keep_id: str, merge_id: str):
        if not self.graph.has_node(keep_id) or not self.graph.has_node(merge_id):
            return
            
        keep_data = self.graph.nodes[keep_id]
        merge_data = self.graph.nodes[merge_id]
        
        # Merge descriptions
        desc1 = keep_data.get("description", "")
        desc2 = merge_data.get("description", "")
        if desc2 and desc2 not in desc1:
            keep_data["description"] = (desc1 + "\n---\n" + desc2).strip()
            
        # Redirect incoming edges
        for u, v, data in list(self.graph.in_edges(merge_id, data=True)):
            if u != keep_id:
                self.upsert_relation(u, keep_id, data.get("keywords", ""), data.get("description", ""))
                
        # Redirect outgoing edges
        for u, v, data in list(self.graph.out_edges(merge_id, data=True)):
            if v != keep_id:
                self.upsert_relation(keep_id, v, data.get("keywords", ""), data.get("description", ""))
                
        self.graph.remove_node(merge_id)

    def save_to_disk(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
        nx.write_graphml(self.graph, self.storage_path)
        print(f"\n[GraphStore] Graph saved to {self.storage_path} with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.\n")
