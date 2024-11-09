// Global DOM Object
const inputQuery = document.getElementById('query');
const formSubmit = document.getElementById('queryForm');

inputQuery.addEventListener('keypress', (e) => {
    if (e.key === "Enter") {
        e.preventDefault(); // 기본 엔터 키 동작 방지
        handleSubmit();
    }
});

// Handle query form submission
formSubmit.addEventListener('submit', function (e) {
    e.preventDefault();
    handleSubmit();
});

function handleSubmit() {
    const query = inputQuery.value;

        fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        })
            .then(response => response.json())
            .then(data => {
                document.getElementById('response-box').innerText = data.response;
                // Update chart with filtered data
                myChart.data.labels = data.labels;
                myChart.data.datasets = [{
                    label: '인수',
                    data: data.quantities,
                    backgroundColor: 'green'
                }];
                myChart.update();
            });
}

// Chart.js setup for monthly data with bar chart type
const ctx = document.getElementById('myChart').getContext('2d');
const myChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: [],
        datasets: []
    },
    options: {
        responsive: true,
        scales: {
            y: { beginAtZero: true }
        }
    }
});
