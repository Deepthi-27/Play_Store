document.addEventListener('DOMContentLoaded', () => {
    // State
    const state = {
        categories: [],
        minRating: 3.5,
        minInstalls: 10000
    };

    // Elements
    const checkboxes = document.querySelectorAll('input[name="category"]');
    const ratingSlider = document.getElementById('rating-slider');
    const installsSlider = document.getElementById('installs-slider');
    const ratingVal = document.getElementById('rating-val');
    const installsVal = document.getElementById('installs-val');
    const lastUpdatedEl = document.getElementById('last-updated');

    // Metrics Elements
    const metricEls = {
        total_apps: document.getElementById('metric-total-apps'),
        total_installs: document.getElementById('metric-total-installs'),
        avg_rating: document.getElementById('metric-avg-rating'),
        total_reviews: document.getElementById('metric-total-reviews')
    };

    // Chart Containers
    const chartIds = ['chart1', 'chart2', 'chart3', 'chart4', 'chart5', 'chart6'];

    // Debounce function
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Update Data
    async function updateData() {
        // Collect current state
        const params = new URLSearchParams();
        if (state.categories.length > 0) {
            params.append('categories', state.categories.join(','));
        }
        params.append('min_rating', state.minRating);
        params.append('min_installs', state.minInstalls);

        try {
            const response = await fetch(`/api/data?${params.toString()}`);
            const data = await response.json();

            // Update Metrics
            metricEls.total_apps.textContent = data.metrics.total_apps;
            metricEls.total_installs.textContent = data.metrics.total_installs;
            metricEls.avg_rating.textContent = data.metrics.avg_rating;
            metricEls.total_reviews.textContent = data.metrics.total_reviews;

            lastUpdatedEl.textContent = `Last updated: ${data.last_updated}`;

            // Update Charts
            chartIds.forEach(id => {
                const chartData = data.charts[id];
                const container = document.getElementById(id);

                if (chartData) {
                    // Update layout for responsiveness
                    if (!chartData.layout) chartData.layout = {};
                    chartData.layout.autosize = true;
                    chartData.layout.font = { color: '#B0B3C5' };
                    chartData.layout.paper_bgcolor = 'rgba(0,0,0,0)';
                    chartData.layout.plot_bgcolor = 'rgba(0,0,0,0)';

                    Plotly.react(id, chartData.data, chartData.layout, { responsive: true, displayModeBar: false });
                } else {
                    container.innerHTML = '<div style="color: #666; display: flex; align-items: center; justify-content: center; height: 100%;">No data available</div>';
                }
            });

        } catch (error) {
            console.error('Error fetching data:', error);
        }
    }

    // Event Listeners
    checkboxes.forEach(cb => {
        cb.addEventListener('change', () => {
            if (cb.checked) {
                state.categories.push(cb.value);
            } else {
                state.categories = state.categories.filter(c => c !== cb.value);
            }
            updateData();
        });
    });

    const debouncedUpdate = debounce(updateData, 300);

    ratingSlider.addEventListener('input', (e) => {
        state.minRating = e.target.value;
        ratingVal.textContent = e.target.value;
        debouncedUpdate();
    });

    installsSlider.addEventListener('input', (e) => {
        state.minInstalls = e.target.value;
        installsVal.textContent = e.target.value;
        debouncedUpdate();
    });

    // Initial Load
    updateData();
});
