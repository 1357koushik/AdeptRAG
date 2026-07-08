import json
from db.vector_store import VectorStore
from db.graph_store import GraphStore
from model.llm import generate_text

KEYWORD_EXTRACTION_SYSTEM_PROMPT = """You are an expert AI assistant tasked with extracting key information from user queries for a dual-search Retrieval-Augmented Generation (RAG) system.
Your job is to identify and separate keywords into two distinct categories:

1. "low_level": Specific entities, names, locations, acronyms, or concrete objects (e.g., "Elon Musk", "San Francisco", "API", "Q3 Revenue").
2. "high_level": Broad concepts, themes, actions, or topical domains (e.g., "artificial intelligence", "climate change impact", "financial performance", "software integration").

You must return ONLY a valid JSON object with these two keys, containing lists of strings. Do not include any markdown formatting like ```json or any conversational text.

Example Output:
{
  "low_level": ["Frodo", "Ring", "Mount Doom"],
  "high_level": ["journey", "destruction", "temptation"]
}"""

KEYWORD_EXTRACTION_USER_PROMPT = """Extract keywords from the following query:
"{user_query}"
"""

FINAL_GENERATION_SYSTEM_PROMPT = """You are a highly intelligent knowledge synthesizer and AI assistant. 
Your task is to answer the user's query comprehensively and accurately based STRICTLY on the provided context.

The context is aggregated from a Knowledge Graph (Nodes and Relationships) and a Vector Database (Text Chunks).
- Synthesize the information logically.
- If the context contains conflicting information, mention it.
- If the context does not contain sufficient information to fully answer the query, state what is missing instead of hallucinating.
- Keep your answer clear, well-structured, and directly address the user's intent."""

FINAL_GENERATION_USER_PROMPT = """---Context---
{context_string}

---User Query---
{user_query}

---Answer---"""

class QueryEngine:
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore, model_name: str = "gpt-4o-mini"):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.model_name = model_name

    def get_keywords_from_query(self, query: str) -> dict:
        """Extracts low-level and high-level keywords from the query using the LLM."""
        print(f"\n[QueryEngine] Extracting keywords using {self.model_name}...")
        
        response = generate_text(
            system_prompt=KEYWORD_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=KEYWORD_EXTRACTION_USER_PROMPT.format(user_query=query),
            model_name=self.model_name
        )
        
        try:
            # Clean up potential markdown formatting if the model disobeys instructions
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
                
            keywords = json.loads(cleaned_response.strip())
            return keywords
        except json.JSONDecodeError:
            print(f"  [!] Failed to parse keywords JSON: {response}")
            # Fallback
            return {"low_level": [query], "high_level": []}

    def _build_query_context(self, keywords: dict, query: str) -> str:
        """Retrieves data from both Vector and Graph stores and formats it into a context string."""
        print("[QueryEngine] Building context from VectorDB and GraphDB...")
        
        low_level = keywords.get("low_level", [])
        high_level = keywords.get("high_level", [])
        
        # 1. Vector Store Search
        # We search using the original query + keywords to cast a wide semantic net
        search_query = query + " " + " ".join(low_level + high_level)
        vector_results = self.vector_store.search_chunks(query=search_query, n_results=5)
        
        text_chunks = []
        if vector_results and "documents" in vector_results and vector_results["documents"]:
            text_chunks = vector_results["documents"][0]
            
        # 2. Graph Store Search
        graph_nodes = []
        graph_edges = []
        
        # Look for nodes that match low-level keywords (exact or substring)
        matched_nodes = set()
        for kw in low_level:
            kw_lower = kw.lower()
            for node_id, data in self.graph_store.graph.nodes(data=True):
                # Match by node ID or name
                if kw_lower in str(node_id).lower() or kw_lower in str(data.get("name", "")).lower():
                    matched_nodes.add(node_id)
                    
        # Extract node descriptions and connected edges
        for node in matched_nodes:
            data = self.graph_store.graph.nodes[node]
            graph_nodes.append(f"Entity: {data.get('name', node)} ({data.get('entity_type', 'Unknown')})\nDescription: {data.get('description', '')}")
            
            # Get edges (both outgoing and incoming)
            # Outgoing
            for target in self.graph_store.graph.successors(node):
                edge_data = self.graph_store.graph.edges[node, target]
                target_data = self.graph_store.graph.nodes[target]
                source_name = data.get('name', node)
                target_name = target_data.get('name', target)
                graph_edges.append(f"Relation: {source_name} -> {target_name} [{edge_data.get('keywords', '')}]\nDescription: {edge_data.get('description', '')}")
            # Incoming
            for source in self.graph_store.graph.predecessors(node):
                edge_data = self.graph_store.graph.edges[source, node]
                source_data = self.graph_store.graph.nodes[source]
                source_name = source_data.get('name', source)
                target_name = data.get('name', node)
                graph_edges.append(f"Relation: {source_name} -> {target_name} [{edge_data.get('keywords', '')}]\nDescription: {edge_data.get('description', '')}")

        # Remove duplicates from edges
        graph_edges = list(set(graph_edges))
        
        # 3. Compile Context String
        context_parts = []
        
        if text_chunks:
            context_parts.append("===== SEMANTIC TEXT CHUNKS =====")
            for i, chunk in enumerate(text_chunks):
                context_parts.append(f"--- Chunk {i+1} ---\n{chunk}")
                
        if graph_nodes:
            context_parts.append("\n===== KNOWLEDGE GRAPH ENTITIES =====")
            context_parts.append("\n\n".join(graph_nodes))
            
        if graph_edges:
            context_parts.append("\n===== KNOWLEDGE GRAPH RELATIONSHIPS =====")
            context_parts.append("\n\n".join(graph_edges))
            
        return "\n".join(context_parts)

    def query(self, user_query: str) -> tuple[str, str]:
        """Executes the full dual-search RAG pipeline."""
        # 1. Keyword Extraction
        keywords = self.get_keywords_from_query(user_query)
        print(f"  -> Keywords extracted: {keywords}")
        
        # 2. Context Building
        context_string = self._build_query_context(keywords, user_query)
        
        if not context_string.strip():
            print("  [!] Warning: No relevant context found in VectorDB or GraphDB.")
            
        # 3. Final Generation
        print(f"[QueryEngine] Generating final answer using {self.model_name}...")
        final_prompt = FINAL_GENERATION_USER_PROMPT.format(
            context_string=context_string,
            user_query=user_query
        )
        
        final_answer = generate_text(
            system_prompt=FINAL_GENERATION_SYSTEM_PROMPT,
            user_prompt=final_prompt,
            model_name=self.model_name
        )
        
        # Return both the answer and the exact prompt sent to the LLM
        return final_answer, final_prompt
