<script>
    document.addEventListener("DOMContentLoaded", () => {
        fetchPracticeData();
        fetchPracticeHistory();

        // Fetch the main statistics data
        function fetchPracticeData() {
            fetch(`/practice/main/`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('total-solved').textContent = data.data.total_solved;
                        document.getElementById('success-rate').textContent = `${data.data.success_rate}%`;
                        renderCorrectRateChart(data.data.success_rate);
                        renderAnalyticsChart(data.data.analytics_data);
                    } else {
                        console.error('Failed to load practice data:', data.message);
                    }
                })
                .catch(error => console.error('Error loading practice data:', error));
        }

        // Render the correct answer rate chart
        function renderCorrectRateChart(successRate) {
            const ctx = document.getElementById('correctRateChart').getContext('2d');
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Correct', 'Incorrect'],
                    datasets: [{
                        data: [successRate, 100 - successRate],
                        backgroundColor: ['#057159', '#ddd']
                    }]
                },
                options: {
                    responsive: true,
                    cutout: '70%',
                    plugins: {
                        legend: { position: 'bottom' }
                    }
                }
            });
        }

        // Render the concept mastery analytics chart
        function renderAnalyticsChart(analyticsData) {
            const ctx = document.getElementById('analyticsChart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: analyticsData.labels,
                    datasets: [{
                        label: 'Correct Answers %',
                        data: analyticsData.correct_rates,
                        backgroundColor: 'rgba(75, 192, 192, 0.6)'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { beginAtZero: true, max: 100 }
                    }
                }
            });
        }

        // Fetch and render practice history
        function fetchPracticeHistory() {
            fetch(`/api/user/practice_history/`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const tableBody = document.getElementById('practiceHistoryTable');
                        tableBody.innerHTML = '';
                        data.history.forEach(item => {
                            const row = document.createElement('tr');
                            row.innerHTML = `
                                <td><a href="/practice/${item.set_id}/">${item.concept_area}</a></td>
                                <td>${item.practice_date}</td>
                                <td>${item.correct_rate}%</td>
                            `;
                            tableBody.appendChild(row);
                        });
                    } else {
                        console.error('Failed to fetch practice history:', data.message);
                    }
                })
                .catch(error => console.error('Error fetching practice history:', error));
        }
    });
</script>