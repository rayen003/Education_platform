from typing import Dict, List, Any
from langgraph.graph import StateGraph, END
import matplotlib.pyplot as plt
import networkx as nx

# Mock agent classes for demonstration
class LogicAgent:
    def analyze(self, state):
        return state

class MathAgent:
    def analyze(self, state):
        return state

class LanguageAgent:
    def analyze(self, state):
        return state

class AggregatorAgent:
    def aggregate(self, state):
        return state

def build_graph():
    """
    Builds the LangGraph for the feedback system.
    """
    # Define the graph
    builder = StateGraph(Any)
    
    # Add nodes
    builder.add_node("logic_analysis", LogicAgent().analyze)
    builder.add_node("math_analysis", MathAgent().analyze)
    builder.add_node("language_analysis", LanguageAgent().analyze)
    builder.add_node("aggregate_feedback", AggregatorAgent().aggregate)
    
    # Define edges
    # Start with parallel execution of analysis agents
    builder.add_edge("START", "logic_analysis")
    builder.add_edge("START", "math_analysis")
    builder.add_edge("START", "language_analysis")
    
    # All analysis agents feed into the aggregator
    builder.add_edge("logic_analysis", "aggregate_feedback")
    builder.add_edge("math_analysis", "aggregate_feedback")
    builder.add_edge("language_analysis", "aggregate_feedback")
    
    # End after aggregation
    builder.add_edge("aggregate_feedback", END)
    
    return builder

if __name__ == "__main__":
    print("Building agent graph...")
    graph_builder = build_graph()
    
    print("Visualizing agent graph...")
    plt.figure(figsize=(10, 6))
    
    # Create a NetworkX graph from our state graph
    G = nx.DiGraph()
    
    # Add nodes
    G.add_node("START")
    G.add_node("logic_analysis")
    G.add_node("math_analysis")
    G.add_node("language_analysis")
    G.add_node("aggregate_feedback")
    G.add_node("END")
    
    # Add edges
    G.add_edge("START", "logic_analysis")
    G.add_edge("START", "math_analysis")
    G.add_edge("START", "language_analysis")
    G.add_edge("logic_analysis", "aggregate_feedback")
    G.add_edge("math_analysis", "aggregate_feedback")
    G.add_edge("language_analysis", "aggregate_feedback")
    G.add_edge("aggregate_feedback", "END")
    
    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=2000, font_size=10, font_weight='bold',
            arrows=True, arrowsize=15)
    
    plt.title("Feedback Agent System Architecture")
    plt.tight_layout()
    plt.savefig('agent_graph.png')  # Save to file in case display doesn't work
    print("Graph saved to agent_graph.png")
    plt.show()
    print("Done!")
