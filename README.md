<div align="center">
  <h1>🚀 AdeptRAG</h1>
  <p><b>Intelligent Hybrid Graph & Vector Retrieval-Augmented Generation</b></p>
</div>

---

**AdeptRAG** is a powerful, Retrieval-Augmented Generation (RAG) framework that seamlessly integrates Vector Search with Knowledge Graphs. Designed for extreme scalability and accuracy, AdeptRAG introduces agentic routing, asynchronous community summarization, and cross-document coreference resolution to deliver deep, contextual answers over massive datasets without the performance bottlenecks typical of traditional GraphRAG systems.


---

## ✨ Key Features

- **Intelligent Chunk Filtering:** Maximizes indexing efficiency and prevents graph bloat by automatically routing list-heavy or non-narrative data directly to the vector database, bypassing expensive graph extraction.
- **Async Community Summarization:** Pre-computes and summarizes graph communities in the background using hierarchical clustering. This drastically reduces query latency for global, thematic questions from minutes to seconds.
- **Agentic Query Routing:** An LLM-driven router autonomously analyzes user intents to direct queries to the Vector DB, Local Graph, or Global Summaries, ensuring the most efficient retrieval path is always used.
- **Self-Reflective Retrieval:** Implements a critique loop where the system evaluates its own drafted answers. If confidence is low, it triggers fallbacks (like expanding graph hops or querying the web) to prevent hallucinations.

---

## 🏗️ Architecture Workflow

AdeptRAG relies on a dual-pipeline architecture:

1. **Ingestion Pipeline:** 
   Raw documents are chunked and classified. High-value narrative chunks undergo LLM-based entity and relationship extraction. Entities are disambiguated and merged into the Graph DB, while all texts and vectors are simultaneously stored in the Vector DB and KV Store. Finally, a background process clusters the graph and generates high-level community summaries.
   
2. **Agentic Query Pipeline:** 
   User queries are intercepted by the Agentic Router. Depending on the intent (specific fact vs. broad theme), the router queries the appropriate databases. Context is aggregated and passed to the LLM for generation, followed by a self-reflection check before delivering the final cited response to the user.

---

## 💻 Interfaces

AdeptRAG ships with two native interfaces to suit any workflow:

### 1. AdeptRAG CLI (For Developers)
A robust command-line tool built for seamless CI/CD integration, local testing, and pipeline management.

- *(coming soon)*

### 2. AdeptRAG Web Interface (For End Users)
A sleek, modern web dashboard connecting directly to the API backend.
- **Knowledge Base Manager**: Drag-and-drop file uploads for PDFs, TXTs, and Office documents.
- **Graph Visualizer**: Interactive 2D/3D visualization of your data, allowing for visual exploration of entities and relationships.
- **Chat Workspace**: A chat interface where users can ask questions, view source citations, and inspect the retrieval path chosen by the Agentic Router.

---

## 🚀 Getting Started

*(Installation and Setup instructions coming soon)*
