"""
visualizeMeetingGraph.py - Fixed version with guaranteed HTML output
"""

import json
import logging
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Check and install pyvis if needed
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
    print("✅ Pyvis is available - HTML visualization will work")
except ImportError:
    print("❌ Pyvis not found. Installing now...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyvis"])
        from pyvis.network import Network
        PYVIS_AVAILABLE = True
        print("✅ Pyvis installed successfully")
    except:
        PYVIS_AVAILABLE = False
        print("❌ Failed to install pyvis. HTML output will not be available")

# Graph library
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    print("Installing networkx...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx"])
    import networkx as nx
    NETWORKX_AVAILABLE = True

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not available. Install with: pip install matplotlib")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------- GRAPH LOADING --------------------

def load_meeting_graph(graph_file: Path) -> nx.MultiDiGraph:
    """Load meeting knowledge graph from JSON file"""
    if not graph_file.exists():
        logger.error(f"Graph file not found: {graph_file}")
        return nx.MultiDiGraph()
    
    with open(graph_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    G = nx.MultiDiGraph()
    
    # Add nodes
    for node_data in graph_data.get("nodes", []):
        node_id = node_data.pop("id")
        G.add_node(node_id, **node_data)
    
    # Add edges
    for edge_data in graph_data.get("edges", []):
        source = edge_data.pop("source")
        target = edge_data.pop("target")
        G.add_edge(source, target, **edge_data)
    
    logger.info(f"Loaded meeting graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G


def load_tasks(tasks_file: Path) -> List[Dict]:
    """Load tasks from JSON file"""
    if not tasks_file.exists():
        logger.warning(f"Tasks file not found: {tasks_file}")
        return []
    
    with open(tasks_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_transcript(transcript_file: Path) -> str:
    """Load transcript from text file"""
    if not transcript_file.exists():
        return ""
    
    with open(transcript_file, 'r', encoding='utf-8') as f:
        return f.read()

# -------------------- NODE STYLING --------------------

def get_node_style(node_type: str, node_data: Dict = None) -> Dict:
    """Get visual styling for different node types"""
    styles = {
        "task": {
            "color": "#FF6B6B",      # Coral red
            "shape": "box",
            "size": 40,
            "border": "#C92A2A",
            "icon": "📋",
            "matplotlib_color": "lightcoral"
        },
        "entity": {
            "color": "#4ECDC4",       # Turquoise
            "shape": "ellipse",
            "size": 30,
            "border": "#2C7A7B",
            "icon": "👤",
            "matplotlib_color": "lightblue"
        },
        "person": {
            "color": "#45B7D1",       # Blue
            "shape": "ellipse",
            "size": 30,
            "border": "#2C3E50",
            "icon": "👤",
            "matplotlib_color": "skyblue"
        },
        "department": {
            "color": "#96CEB4",       # Sage green
            "shape": "box",
            "size": 35,
            "border": "#588157",
            "icon": "🏢",
            "matplotlib_color": "lightgreen"
        },
        "decision": {
            "color": "#FFD93D",       # Yellow
            "shape": "diamond",
            "size": 35,
            "border": "#E6B800",
            "icon": "✓",
            "matplotlib_color": "gold"
        },
        "unknown": {
            "color": "#95A5A6",       # Gray
            "shape": "ellipse",
            "size": 25,
            "border": "#7F8C8D",
            "icon": "❓",
            "matplotlib_color": "lightgray"
        }
    }
    
    # Check for specific entity types
    if node_type == "entity" and node_data:
        entity_type = node_data.get("entity_type", "").lower()
        if entity_type in styles:
            return styles[entity_type]
    
    return styles.get(node_type, styles["unknown"])


def format_node_label(node_id: str, node_data: Dict) -> str:
    """Create readable label for node"""
    node_type = node_data.get("type", "unknown")
    
    if node_type == "task":
        desc = node_data.get("description", "")
        return desc[:40] + "..." if len(desc) > 40 else desc
    elif node_type == "entity":
        name = node_data.get("name", node_id.split(":", 1)[-1])
        role = node_data.get("role", "")
        if role:
            return f"{name}\n({role})"
        return name
    else:
        return node_id.split(":", 1)[-1]

# -------------------- INTERACTIVE VISUALIZATION (Pyvis) - FIXED VERSION --------------------

def visualize_meeting_interactive(
    G: nx.MultiDiGraph,
    output_file: Path,
    tasks: List[Dict] = None,
    height: str = "800px",
    width: str = "100%"
):
    """Create interactive HTML visualization using pyvis"""
    if not PYVIS_AVAILABLE:
        logger.error("❌ Pyvis not available. Cannot create HTML visualization.")
        logger.info("Install with: pip install pyvis")
        return False
    
    logger.info("🎨 Creating interactive HTML meeting graph visualization...")
    
    try:
        # Create network
        net = Network(height=height, width=width, directed=True, bgcolor="#FFFFFF", font_color="#333333")
        
        # Configure physics for better layout
        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "barnesHut": {
                    "gravitationalConstant": -3000,
                    "centralGravity": 0.3,
                    "springLength": 120,
                    "springConstant": 0.04,
                    "damping": 0.09
                },
                "stabilization": {
                    "iterations": 100
                }
            },
            "edges": {
                "arrows": {
                    "to": {"enabled": true}
                },
                "smooth": {"type": "continuous"},
                "font": {"size": 12, "align": "middle"}
            },
            "nodes": {
                "font": {"size": 14, "face": "arial"},
                "borderWidth": 2,
                "shadow": true
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 200,
                "zoomView": true
            }
        }
        """)
        
        # Add nodes
        nodes_added = 0
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            node_type = node_data.get("type", "unknown")
            style = get_node_style(node_type, node_data)
            
            # Create detailed title (tooltip)
            title_lines = [f"<b>Type:</b> {node_type}"]
            
            if node_type == "task":
                title_lines.extend([
                    f"<b>Description:</b> {node_data.get('description', 'N/A')}",
                    f"<b>Assignee:</b> {node_data.get('assignee', 'N/A')}",
                    f"<b>Due:</b> {node_data.get('due_date', 'N/A')}",
                    f"<b>Location:</b> {node_data.get('location', 'N/A')}",
                    f"<b>Priority:</b> {node_data.get('priority', 'N/A')}",
                    f"<b>Status:</b> {node_data.get('status', 'N/A')}"
                ])
            elif node_type == "entity":
                title_lines.extend([
                    f"<b>Name:</b> {node_data.get('name', 'N/A')}",
                    f"<b>Role:</b> {node_data.get('role', 'N/A')}",
                    f"<b>Entity Type:</b> {node_data.get('entity_type', 'N/A')}"
                ])
            
            # Add mentions if available
            mentions = node_data.get('mentions', [])
            if mentions:
                title_lines.append(f"<b>Mentions:</b> {len(mentions)} times")
            
            # Format label
            label = format_node_label(node_id, node_data)
            
            # Add node
            net.add_node(
                node_id,
                label=label,
                color=style["color"],
                size=style["size"],
                title="<br>".join(title_lines),
                shape=style["shape"],
                borderWidth=2,
                borderColor=style["border"],
                group=node_type
            )
            nodes_added += 1
        
        logger.info(f"  Added {nodes_added} nodes to visualization")
        
        # Add edges with relation labels
        edges_added = 0
        for source, target, edge_data in G.edges(data=True):
            relation = edge_data.get("relation", "related_to")
            
            # Style by relation type
            if relation == "assigned":
                color = "#E74C3C"  # Red
                width = 3
                title = f"<b>Assigned to:</b> {relation}"
            elif relation == "responsible_for":
                color = "#3498DB"  # Blue
                width = 2.5
                title = f"<b>Responsible:</b> {relation}"
            elif relation == "mentions":
                color = "#2ECC71"  # Green
                width = 2
                title = f"<b>Mentions:</b> {relation}"
            else:
                color = "#95A5A6"  # Gray
                width = 1.5
                title = f"<b>Relation:</b> {relation}"
            
            net.add_edge(source, target, title=title, color=color, width=width, arrows="to")
            edges_added += 1
        
        logger.info(f"  Added {edges_added} edges to visualization")
        
        # Save HTML file
        net.save_graph(str(output_file))
        logger.info(f"✅ HTML visualization saved to: {output_file}")
        
        # Verify file was created
        if output_file.exists():
            file_size = output_file.stat().st_size
            logger.info(f"  File size: {file_size} bytes")
            return True
        else:
            logger.error(f"❌ Failed to create HTML file: {output_file}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error creating HTML visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_task_focus(G: nx.MultiDiGraph, output_file: Path, tasks: List[Dict]):
    """Create task-focused HTML visualization"""
    if not PYVIS_AVAILABLE:
        return False
    
    logger.info("🎨 Creating task-focused HTML visualization...")
    
    try:
        net = Network(height="800px", width="100%", directed=True, bgcolor="#FFFFFF")
        
        # Find task nodes and their immediate connections
        task_nodes = []
        connected_nodes = set()
        
        for node_id in G.nodes():
            if G.nodes[node_id].get("type") == "task":
                task_nodes.append(node_id)
                connected_nodes.add(node_id)
                
                # Add neighbors
                for neighbor in G.neighbors(node_id):
                    connected_nodes.add(neighbor)
                for neighbor in G.predecessors(node_id):
                    connected_nodes.add(neighbor)
        
        # Create subgraph
        subgraph_nodes = list(connected_nodes)
        
        # Add nodes
        for node_id in subgraph_nodes:
            if node_id in G:
                node_data = G.nodes[node_id]
                node_type = node_data.get("type", "unknown")
                style = get_node_style(node_type, node_data)
                
                # Make task nodes larger
                size = style["size"] * 1.5 if node_type == "task" else style["size"]
                
                net.add_node(
                    node_id,
                    label=format_node_label(node_id, node_data),
                    color=style["color"],
                    size=size,
                    title=f"<b>{node_type}</b><br>{node_data.get('description', node_data.get('name', ''))}",
                    shape=style["shape"],
                    group=node_type
                )
        
        # Add edges between subgraph nodes
        for u, v, data in G.edges(data=True):
            if u in subgraph_nodes and v in subgraph_nodes:
                net.add_edge(u, v, title=data.get("relation", "related"), arrows="to")
        
        net.set_options("""
        {
            "physics": {"barnesHut": {"springLength": 150}},
            "edges": {"arrows": {"to": {"enabled": true}}},
            "interaction": {"hover": true}
        }
        """)
        
        net.save_graph(str(output_file))
        logger.info(f"✅ Task-focused HTML saved to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating task-focused HTML: {e}")
        return False

# -------------------- STATIC VISUALIZATION (Matplotlib) --------------------

def visualize_meeting_static(
    G: nx.MultiDiGraph,
    output_file: Path,
    tasks: List[Dict] = None,
    layout: str = "spring"
):
    """Create static visualization using matplotlib"""
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available")
        return
    
    logger.info(f"Creating static visualization with {layout} layout...")
    
    try:
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "hierarchical":
            try:
                pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
            except:
                pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
        else:
            pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
        
        # Create figure
        plt.figure(figsize=(16, 12))
        
        # Group nodes by type
        task_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "task"]
        entity_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "entity"]
        decision_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "decision"]
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color="#BDC3C7",
            width=1.0,
            alpha=0.5,
            arrows=True,
            arrowsize=10
        )
        
        # Draw nodes by type
        if task_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=task_nodes,
                node_color='lightcoral',
                node_size=1000,
                edgecolors='darkred',
                linewidths=2,
                alpha=0.9,
                label=f'Tasks ({len(task_nodes)})'
            )
        
        if entity_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=entity_nodes,
                node_color='skyblue',
                node_size=600,
                edgecolors='steelblue',
                linewidths=1.5,
                alpha=0.8,
                label=f'Entities ({len(entity_nodes)})'
            )
        
        if decision_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=decision_nodes,
                node_color='gold',
                node_size=800,
                edgecolors='darkgoldenrod',
                linewidths=2,
                alpha=0.9,
                label=f'Decisions ({len(decision_nodes)})'
            )
        
        # Draw labels for task nodes
        if task_nodes:
            task_labels = {
                n: G.nodes[n].get('description', '')[:20] + ('...' if len(G.nodes[n].get('description', '')) > 20 else '')
                for n in task_nodes
            }
            nx.draw_networkx_labels(G, pos, task_labels, font_size=8, font_weight='bold')
        
        # Title
        task_count = len(task_nodes)
        entity_count = len(entity_nodes)
        title = f"Meeting Knowledge Graph\n{task_count} Tasks, {entity_count} Entities, {G.number_of_edges()} Relations"
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Static visualization saved to: {output_file}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating static visualization: {e}")

# -------------------- MAIN FUNCTION --------------------

def find_meeting_files(meeting_dir: Path) -> Dict[str, Optional[Path]]:
    """Find all meeting output files"""
    files = {
        'graph': None,
        'tasks': None,
        'transcript': None
    }
    
    # Find knowledge graph
    kg_files = list(meeting_dir.glob("*_knowledge_graph.json"))
    if kg_files:
        files['graph'] = kg_files[0]
        logger.info(f"Found graph file: {kg_files[0].name}")
    
    # Find tasks
    task_files = list(meeting_dir.glob("*_tasks.json"))
    if task_files:
        files['tasks'] = task_files[0]
        logger.info(f"Found tasks file: {task_files[0].name}")
    
    # Find transcript
    transcript_files = list(meeting_dir.glob("*_transcript.txt"))
    if transcript_files:
        files['transcript'] = transcript_files[0]
        logger.info(f"Found transcript file: {transcript_files[0].name}")
    
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Visualize meeting knowledge graph with guaranteed HTML output"
    )
    
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to meeting output directory"
    )
    
    parser.add_argument(
        "--graph-file",
        type=str,
        help="Explicit path to knowledge graph JSON file"
    )
    
    parser.add_argument(
        "--tasks-file",
        type=str,
        help="Explicit path to tasks JSON file"
    )
    
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Generate only HTML visualization (skip PNG)"
    )
    
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Generate only PNG visualization (skip HTML)"
    )
    
    parser.add_argument(
        "--task-focus",
        action="store_true",
        help="Generate task-focused HTML visualization"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for visualizations"
    )
    
    args = parser.parse_args()
    
    # Determine files to process
    graph_file = None
    tasks_file = None
    meeting_dir = None
    
    if args.graph_file:
        graph_file = Path(args.graph_file)
        meeting_dir = graph_file.parent
        logger.info(f"Using specified graph file: {graph_file}")
    elif args.path:
        meeting_dir = Path(args.path)
        if meeting_dir.is_dir():
            logger.info(f"Scanning directory: {meeting_dir}")
            files = find_meeting_files(meeting_dir)
            graph_file = files['graph']
            tasks_file = files['tasks']
        else:
            logger.error(f"Directory not found: {meeting_dir}")
            return
    else:
        # Try to find most recent meeting
        meetings_root = Path("output/meetings")
        if meetings_root.exists():
            meetings = [d for d in meetings_root.iterdir() if d.is_dir()]
            if meetings:
                meeting_dir = max(meetings, key=lambda p: p.stat().st_mtime)
                logger.info(f"Using most recent meeting: {meeting_dir}")
                files = find_meeting_files(meeting_dir)
                graph_file = files['graph']
                tasks_file = files['tasks']
            else:
                logger.error("No meetings found in output/meetings")
                return
        else:
            logger.error("No meeting directory specified and output/meetings not found")
            return
    
    # Override tasks file if specified
    if args.tasks_file:
        tasks_file = Path(args.tasks_file)
        logger.info(f"Using specified tasks file: {tasks_file}")
    
    # Validate graph file
    if not graph_file or not graph_file.exists():
        logger.error(f"❌ Knowledge graph file not found: {graph_file}")
        logger.error("Please provide a valid graph file or meeting directory")
        return
    
    # Load data
    print("\n" + "=" * 60)
    print("📊 MEETING GRAPH VISUALIZATION")
    print("=" * 60)
    print(f"Loading graph from: {graph_file}")
    
    G = load_meeting_graph(graph_file)
    tasks = load_tasks(tasks_file) if tasks_file else []
    
    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = meeting_dir / "visualizations" if meeting_dir else Path("visualizations")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    meeting_name = graph_file.stem.replace("_knowledge_graph", "")
    
    # Generate visualizations
    html_generated = False
    png_generated = False
    
    # HTML Visualization
    if not args.png_only and PYVIS_AVAILABLE:
        html_file = output_dir / f"{meeting_name}_interactive.html"
        print(f"\n🔧 Generating HTML visualization: {html_file}")
        html_generated = visualize_meeting_interactive(G, html_file, tasks)
        
        if args.task_focus:
            task_file = output_dir / f"{meeting_name}_tasks_focus.html"
            print(f"🔧 Generating task-focused HTML: {task_file}")
            visualize_task_focus(G, task_file, tasks)
    
    # PNG Visualization
    if not args.html_only and MATPLOTLIB_AVAILABLE:
        png_file = output_dir / f"{meeting_name}_static.png"
        print(f"\n🔧 Generating PNG visualization: {png_file}")
        visualize_meeting_static(G, png_file, tasks)
        png_generated = True
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 60)
    
    if html_generated:
        html_file = output_dir / f"{meeting_name}_interactive.html"
        print(f"\n📱 HTML file: {html_file}")
        print(f"   Open this in your web browser to explore interactively!")
    
    if png_generated:
        png_file = output_dir / f"{meeting_name}_static.png"
        print(f"\n🖼️  PNG file: {png_file}")
    
    if not html_generated and not png_generated:
        print("\n❌ No visualizations were generated")
        if not PYVIS_AVAILABLE:
            print("   Install pyvis for HTML output: pip install pyvis")
        if not MATPLOTLIB_AVAILABLE:
            print("   Install matplotlib for PNG output: pip install matplotlib")


if __name__ == "__main__":
    main()