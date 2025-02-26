// Color schemes
const domainColors = {
    'Introduction': '#4CAF50',
    'Supply and Demand': '#2196F3',
    'Consumer Behavior': '#9C27B0',
    'Production and Costs': '#FF9800',
    'Market Structures': '#F44336',
    'Factor Markets': '#795548',
    'Market Failures': '#607D8B'
};

const difficultyColors = d3.scaleLinear()
    .domain([1, 5])
    .range(['#a8e6cf', '#3d405b']);

// Knowledge Graph Visualization
class KnowledgeGraph {
    constructor(containerId) {
        this.svg = d3.select(containerId)
            .append("svg")
            .attr("width", "100%")
            .attr("height", "100%")
            .attr("viewBox", [0, 0, window.innerWidth, window.innerHeight]);

        this.simulation = null;
        this.nodes = [];
        this.links = [];
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {
                this.svg.select("g").attr("transform", event.transform);
            });
        
        this.svg.call(zoom);
        
        // Create container for graph
        this.container = this.svg.append("g");
        
        // Create tooltip
        this.tooltip = d3.select("body")
            .append("div")
            .attr("class", "tooltip")
            .style("opacity", 0);
    }

    async loadData() {
        try {
            const response = await fetch("/api/concepts");
            const data = await response.json();
            this.nodes = data.nodes;
            this.links = data.links;
            this.render();
        } catch (error) {
            console.error("Error loading graph data:", error);
        }
    }

    render() {
        // Create links
        const links = this.container.selectAll(".link")
            .data(this.links)
            .join("line")
            .attr("class", "link");

        // Create nodes
        const nodes = this.container.selectAll(".node")
            .data(this.nodes)
            .join("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", (event, d) => this.dragStarted(event, d))
                .on("drag", (event, d) => this.dragged(event, d))
                .on("end", (event, d) => this.dragEnded(event, d)));

        // Add circles to nodes
        nodes.append("circle")
            .attr("r", 15);

        // Add labels to nodes
        nodes.append("text")
            .attr("dy", 25)
            .attr("text-anchor", "middle")
            .text(d => d.name)
            .call(this.wrap, 120);

        // Add tooltips
        nodes.on("mouseover", (event, d) => {
            this.tooltip.transition()
                .duration(200)
                .style("opacity", 1);
            this.tooltip.html(`
                <strong>${d.name}</strong><br/>
                ${d.type ? `Type: ${d.type}` : ""}
                ${d.difficulty ? `<br/>Difficulty: ${d.difficulty}` : ""}
            `)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 10) + "px");
        })
        .on("mouseout", () => {
            this.tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        });

        // Set up force simulation
        this.simulation = d3.forceSimulation(this.nodes)
            .force("link", d3.forceLink(this.links)
                .id(d => d.id)
                .distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(window.innerWidth / 2, window.innerHeight / 2))
            .force("collision", d3.forceCollide().radius(30));

        // Update positions on tick
        this.simulation.on("tick", () => {
            links
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            nodes.attr("transform", d => `translate(${d.x},${d.y})`);
        });
    }

    wrap(text, width) {
        text.each(function() {
            let text = d3.select(this);
            let words = text.text().split(/\s+/).reverse();
            let word;
            let line = [];
            let lineNumber = 0;
            let lineHeight = 1.1;
            let y = text.attr("y");
            let dy = parseFloat(text.attr("dy"));
            let tspan = text.text(null).append("tspan")
                .attr("x", 0)
                .attr("y", y)
                .attr("dy", dy + "px");
            
            while (word = words.pop()) {
                line.push(word);
                tspan.text(line.join(" "));
                if (tspan.node().getComputedTextLength() > width) {
                    line.pop();
                    tspan.text(line.join(" "));
                    line = [word];
                    tspan = text.append("tspan")
                        .attr("x", 0)
                        .attr("y", y)
                        .attr("dy", ++lineNumber * lineHeight + dy + "px")
                        .text(word);
                }
            }
        });
    }

    dragStarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragEnded(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

// Initialize graph when document is ready
document.addEventListener("DOMContentLoaded", () => {
    const graph = new KnowledgeGraph("#graph-container");
    graph.loadData();
});

// Global functions for button clicks
window.startModuleQuiz = (moduleId) => {
    console.log(`Starting quiz for module: ${moduleId}`);
    // Implement quiz start logic
};

window.startPractice = (conceptId) => {
    console.log(`Starting practice for concept: ${conceptId}`);
    // Implement practice start logic
};

window.findResources = (conceptId) => {
    console.log(`Finding resources for concept: ${conceptId}`);
    // Implement resource finding logic
};

window.resetView = () => {
    // Implement view reset logic
};

window.togglePhysics = () => {
    // Implement physics toggle logic
};

// Add this function for quiz generation
function generateQuiz(conceptId) {
    fetch(`/api/quiz/${conceptId}`)
        .then(response => response.json())
        .then(quiz => {
            // Create and show quiz modal
            const modal = document.createElement('div');
            modal.className = 'quiz-modal';
            
            let questionsHtml = '';
            quiz.questions.forEach((q, index) => {
                let questionContent = '';
                if (q.type === 'multiple_choice') {
                    questionContent = `
                        <div class="options">
                            ${q.options.map((opt, i) => `
                                <label class="option">
                                    <input type="radio" name="q${q.id}" value="${i}">
                                    ${opt}
                                </label>
                            `).join('')}
                        </div>
                    `;
                } else if (q.type === 'true_false') {
                    questionContent = `
                        <div class="options">
                            <label class="option">
                                <input type="radio" name="q${q.id}" value="true"> True
                            </label>
                            <label class="option">
                                <input type="radio" name="q${q.id}" value="false"> False
                            </label>
                        </div>
                    `;
                } else if (q.type === 'short_answer') {
                    questionContent = `
                        <textarea class="short-answer" placeholder="Type your answer here..."></textarea>
                    `;
                }
                
                questionsHtml += `
                    <div class="question">
                        <h4>Question ${index + 1}</h4>
                        <p>${q.question}</p>
                        ${questionContent}
                    </div>
                `;
            });
            
            modal.innerHTML = `
                <div class="quiz-content">
                    <div class="quiz-header">
                        <h3>Quiz: ${quiz.concept}</h3>
                        <button onclick="closeQuiz()" class="close-quiz">×</button>
                    </div>
                    <div class="questions">
                        ${questionsHtml}
                    </div>
                    <div class="quiz-footer">
                        <button onclick="submitQuiz()" class="submit-quiz">Submit Answers</button>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
        })
        .catch(error => {
            console.error('Error generating quiz:', error);
            alert('Failed to generate quiz. Please try again.');
        });
}

function closeQuiz() {
    const modal = document.querySelector('.quiz-modal');
    if (modal) {
        modal.remove();
    }
}

function submitQuiz() {
    // Implement quiz submission and scoring
    alert('Quiz submitted! This is a mock implementation.');
    closeQuiz();
}

// Add these functions for sidebar control
function toggleSidebar(type, show, conceptId = null) {
    const sidebar = document.getElementById(`${type}Sidebar`);
    if (show) {
        sidebar.classList.add('open');
        if (conceptId) {
            loadSidebarContent(type, conceptId);
        }
    } else {
        sidebar.classList.remove('open');
    }
}

function loadSidebarContent(type, conceptId) {
    if (type === 'resources') {
        // Fetch and load resources
        fetch(`/api/resources/${conceptId}`)
            .then(response => response.json())
            .then(data => {
                // Resources will be loaded by the API
                console.log('Resources loaded for:', conceptId);
            });
    } else if (type === 'quiz') {
        // Fetch and load quiz
        fetch(`/api/quiz/${conceptId}`)
            .then(response => response.json())
            .then(data => {
                // Quiz will be loaded by the API
                console.log('Quiz loaded for:', conceptId);
            });
    }
}
