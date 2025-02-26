from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('knowledge_graph.html')

@app.route('/api/knowledge_graph')
def get_knowledge_graph():
    data = {
        "nodes": [
            {
                "id": "1",
                "name": "Supply",
                "description": "The quantity of a good that producers are willing to sell at various prices",
                "difficulty": "Beginner",
                "resources": [
                    {"type": "video", "url": "https://example.com/supply-video", "title": "Understanding Supply"},
                    {"type": "article", "url": "https://example.com/supply-article", "title": "Supply in Economics"}
                ]
            },
            {
                "id": "2",
                "name": "Demand",
                "description": "The quantity of a good that consumers are willing to purchase at various prices",
                "difficulty": "Beginner",
                "resources": [
                    {"type": "video", "url": "https://example.com/demand-video", "title": "Demand Explained"},
                    {"type": "article", "url": "https://example.com/demand-article", "title": "Understanding Demand"}
                ]
            },
            {
                "id": "3",
                "name": "Price",
                "description": "The amount of money expected in exchange for a good or service",
                "difficulty": "Beginner",
                "resources": [
                    {"type": "video", "url": "https://example.com/price-video", "title": "Price Theory"},
                    {"type": "article", "url": "https://example.com/price-article", "title": "Price Mechanisms"}
                ]
            },
            {
                "id": "4",
                "name": "Elasticity",
                "description": "Measure of how demand or supply responds to changes in price",
                "difficulty": "Intermediate",
                "resources": [
                    {"type": "video", "url": "https://example.com/elasticity-video", "title": "Understanding Elasticity"},
                    {"type": "article", "url": "https://example.com/elasticity-article", "title": "Types of Elasticity"}
                ]
            },
            {
                "id": "5",
                "name": "Market Equilibrium",
                "description": "The point where supply meets demand",
                "difficulty": "Intermediate",
                "resources": [
                    {"type": "video", "url": "https://example.com/equilibrium-video", "title": "Market Equilibrium"},
                    {"type": "article", "url": "https://example.com/equilibrium-article", "title": "Understanding Market Balance"}
                ]
            }
        ],
        "links": [
            {"source": "1", "target": "3", "relationship": "determines"},
            {"source": "2", "target": "3", "relationship": "affects"},
            {"source": "1", "target": "5", "relationship": "contributes to"},
            {"source": "2", "target": "5", "relationship": "contributes to"},
            {"source": "3", "target": "5", "relationship": "establishes"},
            {"source": "1", "target": "4", "relationship": "measured by"},
            {"source": "2", "target": "4", "relationship": "measured by"}
        ]
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
