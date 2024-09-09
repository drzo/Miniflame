document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('input-form');
    const output = document.getElementById('output');
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadStatus = document.getElementById('upload-status');
    const modelStats = document.getElementById('model-stats');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        const inputText = document.getElementById('input-text').value;
        
        const response = await fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ input_text: inputText }),
        });

        const data = await response.json();
        output.textContent = data.response;
    });

    uploadBtn.addEventListener('click', async function() {
        const file = fileInput.files[0];
        if (!file) {
            uploadStatus.textContent = 'Please select a file first.';
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        uploadStatus.textContent = 'Uploading and processing file...';

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            uploadStatus.textContent = data.message;
        } catch (error) {
            uploadStatus.textContent = 'Error uploading and processing file.';
            console.error('Error:', error);
        }
    });

    // Fetch and display model stats
    fetch('/model_stats')
        .then(response => response.json())
        .then(data => {
            modelStats.innerHTML = `
                <p>Total Parameters: ${data.total_params.toLocaleString()}</p>
                <p>Trainable Parameters: ${data.trainable_params.toLocaleString()}</p>
                <p>Model Size: ${data.model_size_mb.toFixed(2)} MB</p>
                <p>Device: ${data.device}</p>
                <p>Data Type: ${data.dtype}</p>
            `;
        });

    // Fetch and visualize model structure
    fetch('/model_structure')
        .then(response => response.json())
        .then(data => visualizeModelStructure(data));
});

function visualizeModelStructure(data) {
    const width = 800;
    const height = 600;

    const svg = d3.select("#model-structure")
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    const simulation = d3.forceSimulation(data.layers)
        .force("link", d3.forceLink(data.connections).id(d => d.name).distance(100))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg.append("g")
        .selectAll("line")
        .data(data.connections)
        .enter().append("line")
        .attr("class", "link");

    const node = svg.append("g")
        .selectAll("g")
        .data(data.layers)
        .enter().append("g")
        .attr("class", "node")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    node.append("circle")
        .attr("r", d => d.type === "transformer" ? 15 : 10)
        .attr("fill", d => {
            switch (d.type) {
                case "input": return "#66c2a5";
                case "embedding": return "#fc8d62";
                case "transformer": return "#8da0cb";
                case "output": return "#e78ac3";
                default: return "#a6d854";
            }
        });

    node.append("text")
        .attr("dy", ".35em")
        .attr("text-anchor", "middle")
        .text(d => d.name)
        .attr("font-size", "10px");

    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("transform", d => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}
