import os
import networkx as nx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AdeptRAG Graph Visualizer")

# Ensure template directory exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

GRAPH_FILE = os.path.join(os.path.dirname(BASE_DIR), "workspace", "knowledge_graph.graphml")

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serves the 3D Graph Visualization UI."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/favicon.ico")
async def favicon():
    """Silences the browser's 404 favicon request."""
    from fastapi import Response
    return Response(status_code=204)

@app.get("/api/graph")
async def get_graph_data():
    """Reads the GraphML file and returns node-link JSON for 3d-force-graph."""
    if not os.path.exists(GRAPH_FILE):
        return JSONResponse({"nodes": [], "links": []})
        
    try:
        graph = nx.read_graphml(GRAPH_FILE)
        
        nodes = []
        for node, data in graph.nodes(data=True):
            # Fallback for name if missing
            nodes.append({
                "id": node,
                "name": data.get("name", node),
                "group": data.get("entity_type", "Unknown"),
                "val": 1  # Base node size
            })
            
        links = []
        for source, target, data in graph.edges(data=True):
            links.append({
                "source": source,
                "target": target,
                "label": data.get("keywords", ""),
                "description": data.get("description", "")
            })
            
        return JSONResponse({"nodes": nodes, "links": links})
        
    except Exception as e:
        print(f"Error reading graph: {e}")
        return JSONResponse({"nodes": [], "links": []}, status_code=500)
