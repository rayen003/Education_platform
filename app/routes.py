from flask import Flask, render_template, jsonify
from app import app

@app.route('/')
def index():
    return render_template('knowledge_graph.html')

@app.route('/api/knowledge_graph')
def get_knowledge_graph():
    data = {
        "nodes": [
            {"id": "1", "name": "Supply"},
            {"id": "2", "name": "Demand"},
            {"id": "3", "name": "Price"}
        ],
        "links": [
            {"source": "1", "target": "3"},
            {"source": "2", "target": "3"}
        ]
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
