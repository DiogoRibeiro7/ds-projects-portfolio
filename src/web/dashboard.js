/**
 * Interactive utilities for A/B testing dashboard and visualizations.
 * 
 * This module provides client-side functionality for experiment dashboards.
 */

class ExperimentDashboard {
    constructor(containerId, config = {}) {
        this.container = document.getElementById(containerId);
        this.config = {
            refreshInterval: 30000, // 30 seconds
            maxDataPoints: 1000,
            ...config
        };
        
        // TODO: Add comprehensive error handling for missing DOM elements
        // Current implementation assumes container exists
        // ASSIGNEE: @diogoribeiro7
        // LABELS: error-handling, dom-validation
        // PRIORITY: medium
        
        this.data = null;
        this.charts = {};
        this.isRealTimeEnabled = false;
        
        this.init();
    }
    
    init() {
        /**
         * Initialize the dashboard components.
         * 
         * TODO: Add loading state and progress indicators
         * Users should see feedback during data loading
         * LABELS: ux, loading-states
         * PRIORITY: medium
         */
        
        this.createLayout();
        this.setupEventListeners();
        
        // BUG: Dashboard initializes without checking if data is available
        // Should validate data source before proceeding
        
        // TODO: Add keyboard shortcuts for common actions
        // Power users would benefit from keyboard navigation
        // ASSIGNEE: @diogoribeiro7
        // LABELS: accessibility, keyboard-shortcuts
        // PRIORITY: low
    }
    
    createLayout() {
        /**
         * Create the basic dashboard layout.
         */
        const layout = `
            <div class="dashboard-header">
                <h1>A/B Test Results</h1>
                <div class="controls">
                    <button id="refresh-btn" class="btn btn-primary">Refresh</button>
                    <button id="export-btn" class="btn btn-secondary">Export</button>
                    <label>
                        <input type="checkbox" id="realtime-toggle"> Real-time updates
                    </label>
                </div>
            </div>
            <div class="dashboard-content">
                <div class="metrics-overview" id="metrics-overview">
                    <!-- Metrics cards will be inserted here -->
                </div>
                <div class="charts-container">
                    <div class="chart-panel" id="conversion-chart">
                        <h3>Conversion Rate Over Time</h3>
                        <canvas id="conversion-canvas"></canvas>
                    </div>
                    <div class="chart-panel" id="distribution-chart">
                        <h3>Metric Distribution</h3>
                        <canvas id="distribution-canvas"></canvas>
                    </div>
                </div>
            </div>
        `;
        
        this.container.innerHTML = layout;
        
        // TODO: Make layout responsive for mobile devices
        // Current CSS is desktop-only
        // LABELS: responsive, mobile
        // PRIORITY: medium
    }
    
    setupEventListeners() {
        /**
         * Setup event handlers for dashboard interactions.
         */
        
        // Refresh button
        document.getElementById('refresh-btn')?.addEventListener('click', () => {
            this.refreshData();
        });
        
        // Export button
        document.getElementById('export-btn')?.addEventListener('click', () => {
            this.exportResults();
        });
        
        // Real-time toggle
        document.getElementById('realtime-toggle')?.addEventListener('change', (e) => {
            this.toggleRealTime(e.target.checked);
        });
        
        // TODO: Add support for custom date range selection
        // Users should be able to filter data by time period
        // ASSIGNEE: @diogoribeiro7
        // LABELS: filtering, date-range
        // PRIORITY: high
        
        // FIXME: Event listeners are not properly cleaned up
        // Could cause memory leaks in single-page applications
    }
    
    async loadData(endpoint) {
        /**
         * Load experiment data from API endpoint.
         * 
         * TODO: Add retry logic for failed requests
         * Network issues shouldn't break the dashboard
         * LABELS: networking, retry-logic
         * PRIORITY: high
         */
        
        try {
            const response = await fetch(endpoint);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            this.data = await response.json();
            this.updateDashboard();
            
        } catch (error) {
            console.error('Failed to load data:', error);
            
            // TODO: Show user-friendly error messages in the UI
            // Console errors are not visible to end users
            // LABELS: error-handling, user-feedback
            // PRIORITY: high
            
            this.showError('Failed to load experiment data. Please try again.');
        }
    }
    
    updateDashboard() {
        /**
         * Update all dashboard components with new data.
         */
        
        if (!this.data) {
            console.warn('No data available for dashboard update');
            return;
        }
        
        this.updateMetricsOverview();
        this.updateCharts();
        
        // TODO: Add data freshness indicator
        // Users should know when data was last updated
        // LABELS: data-freshness, timestamps
        // PRIORITY: medium
        
        // TODO: Highlight significant changes since last update
        // Important changes should be visually emphasized
        // ASSIGNEE: @diogoribeiro7
        // LABELS: change-detection, visual-emphasis
        // PRIORITY: low
    }
    
    updateMetricsOverview() {
        /**
         * Update the metrics overview cards.
         */
        const overview = document.getElementById('metrics-overview');
        if (!overview) return;
        
        // BUG: Assumes specific data structure without validation
        // Should check if required properties exist
        
        const metrics = this.data.metrics || {};
        
        let html = '';
        for (const [key, value] of Object.entries(metrics)) {
            const card = this.createMetricCard(key, value);
            html += card;
        }
        
        overview.innerHTML = html;
        
        // TODO: Add trend indicators (up/down arrows) for metrics
        // LABELS: trends, visual-indicators
        // PRIORITY: medium
    }
    
    createMetricCard(name, data) {
        /**
         * Create a metric card HTML element.
         * 
         * TODO: Add statistical significance indicators
         * Cards should show if differences are statistically significant
         * ASSIGNEE: @diogoribeiro7
         * LABELS: statistics, significance
         * PRIORITY: high
         */
        
        const {
            control = 0,
            treatment = 0,
            lift = 0,
            pValue = null,
            confidenceInterval = null
        } = data;
        
        const liftClass = lift > 0 ? 'positive' : lift < 0 ? 'negative' : 'neutral';
        const significanceIndicator = pValue && pValue < 0.05 ? 
            '<span class="significant">*</span>' : '';
        
        return `
            <div class="metric-card">
                <h4>${name}${significanceIndicator}</h4>
                <div class="metric-values">
                    <div class="control-value">
                        <label>Control:</label>
                        <span>${this.formatMetricValue(control)}</span>
                    </div>
                    <div class="treatment-value">
                        <label>Treatment:</label>
                        <span>${this.formatMetricValue(treatment)}</span>
                    </div>
                    <div class="lift-value ${liftClass}">
                        <label>Lift:</label>
                        <span>${this.formatPercentage(lift)}</span>
                    </div>
                </div>
                ${pValue ? `<div class="p-value">p = ${pValue.toFixed(4)}</div>` : ''}
                ${confidenceInterval ? 
                    `<div class="confidence-interval">
                        95% CI: [${this.formatPercentage(confidenceInterval[0])}, 
                                ${this.formatPercentage(confidenceInterval[1])}]
                    </div>` : ''
                }
            </div>
        `;
    }
    
    updateCharts() {
        /**
         * Update all chart visualizations.
         * 
         * TODO: Add chart type selection (line, bar, scatter)
         * Different metrics might benefit from different visualizations
         * LABELS: chart-types, flexibility
         * PRIORITY: medium
         */
        
        this.updateConversionChart();
        this.updateDistributionChart();
        
        // TODO: Add statistical power visualization
        // Show if experiment has sufficient power to detect effects
        // ASSIGNEE: @diogoribeiro7
        // LABELS: power-analysis, visualization
        // PRIORITY: medium
    }
    
    updateConversionChart() {
        /**
         * Update the conversion rate time series chart.
         */
        const canvas = document.getElementById('conversion-canvas');
        if (!canvas) return;
        
        // TODO: Implement actual chart rendering
        // Placeholder implementation using Chart.js or D3.js
        // LABELS: implementation, charting-library
        // PRIORITY: high
        
        // HACK: Using placeholder chart until proper implementation
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#333';
        ctx.font = '16px Arial';
        ctx.fillText('Conversion Chart (To be implemented)', 10, 30);
    }
    
    updateDistributionChart() {
        /**
         * Update the metric distribution comparison chart.
         */
        const canvas = document.getElementById('distribution-canvas');
        if (!canvas) return;
        
        // TODO: Implement distribution visualization
        // Should show overlapping histograms or violin plots
        // LABELS: implementation, distribution-plots
        // PRIORITY: high
        
        // Placeholder implementation
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#333';
        ctx.font = '16px Arial';
        ctx.fillText('Distribution Chart (To be implemented)', 10, 30);
    }
    
    formatMetricValue(value) {
        /**
         * Format metric values for display.
         * 
         * TODO: Add smart formatting based on metric type
         * Currency, percentages, counts should be formatted differently
         * LABELS: formatting, internationalization
         * PRIORITY: low
         */
        if (typeof value === 'number') {
            return value.toLocaleString();
        }
        return value;
    }
    
    formatPercentage(value) {
        /**
         * Format percentage values for display.
         */
        return `${(value * 100).toFixed(2)}%`;
    }
    
    refreshData() {
        /**
         * Manually refresh dashboard data.
         */
        // TODO: Show loading spinner during refresh
        // LABELS: loading-indicators, ux
        // PRIORITY: medium
        
        if (this.config.dataEndpoint) {
            this.loadData(this.config.dataEndpoint);
        }
    }
    
    toggleRealTime(enabled) {
        /**
         * Enable/disable real-time data updates.
         * 
         * TODO: Add websocket support for true real-time updates
         * Current polling approach is inefficient for real-time data
         * ASSIGNEE: @diogoribeiro7
         * LABELS: websockets, real-time
         * PRIORITY: medium
         */
        
        this.isRealTimeEnabled = enabled;
        
        if (enabled) {
            this.startRealTimeUpdates();
        } else {
            this.stopRealTimeUpdates();
        }
    }
    
    startRealTimeUpdates() {
        /**
         * Start polling for real-time updates.
         */
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        this.updateInterval = setInterval(() => {
            this.refreshData();
        }, this.config.refreshInterval);
        
        // NOTE: Polling can be resource-intensive
        // Consider implementing exponential backoff for failed requests
    }
    
    stopRealTimeUpdates() {
        /**
         * Stop real-time updates.
         */
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
    
    exportResults() {
        /**
         * Export dashboard results to file.
         * 
         * TODO: Add multiple export formats (CSV, PDF, PNG)
         * Users might need different formats for different purposes
         * ASSIGNEE: @diogoribeiro7
         * LABELS: export, file-formats
         * PRIORITY: medium
         */
        
        if (!this.data) {
            this.showError('No data available to export');
            return;
        }
        
        // Simple JSON export for now
        const dataStr = JSON.stringify(this.data, null, 2);
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
        
        const exportFileDefaultName = `experiment_results_${new Date().toISOString().split('T')[0]}.json`;
        
        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.click();
        
        // TODO: Add export progress indicator for large datasets
        // LABELS: progress-indicators, large-data
        // PRIORITY: low
    }
    
    showError(message) {
        /**
         * Display error message to user.
         * 
         * TODO: Implement proper toast notification system
         * Current implementation is too basic
         * LABELS: notifications, error-display
         * PRIORITY: medium
         */
        
        // Simple alert for now - should be replaced with proper UI
        alert(`Error: ${message}`);
        
        // TODO: Log errors to analytics service for debugging
        // LABELS: analytics, error-tracking
        // PRIORITY: low
    }
    
    destroy() {
        /**
         * Clean up dashboard resources.
         * 
         * FIXME: Incomplete cleanup implementation
         * Should remove all event listeners and clear intervals
         */
        
        this.stopRealTimeUpdates();
        
        // TODO: Remove all event listeners
        // TODO: Clear chart instances
        // TODO: Remove DOM elements if dynamically created
        // LABELS: cleanup, memory-management
        // PRIORITY: high
    }
}

// TODO: Add A/B test calculator widget
// Standalone widget for sample size and power calculations
// ASSIGNEE: @diogoribeiro7
// LABELS: calculator, widget
// PRIORITY: medium

// TODO: Create experiment design wizard
// Step-by-step interface for designing new experiments
// LABELS: wizard, experiment-design
// PRIORITY: low

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ExperimentDashboard };
}

// HACK: Global export for browser environments
// Should use proper module bundling instead
window.ExperimentDashboard = ExperimentDashboard;

/**
 * Utility functions for statistical calculations in the browser.
 * 
 * NOTE: These are simplified implementations for demonstration
 * Production code should use proper statistical libraries
 */

function calculateTwoProportionTest(x1, n1, x2, n2) {
    /**
     * Simple two-proportion z-test implementation.
     * 
     * TODO: Add proper statistical libraries (jStat, simple-statistics)
     * Hand-rolled statistics implementations are error-prone
     * LABELS: statistics, libraries
     * PRIORITY: high
     */
    
    const p1 = x1 / n1;
    const p2 = x2 / n2;
    const pPool = (x1 + x2) / (n1 + n2);
    
    const se = Math.sqrt(pPool * (1 - pPool) * (1/n1 + 1/n2));
    const z = (p2 - p1) / se;
    
    // TODO: Implement proper p-value calculation
    // Current approximation is not accurate for all z-values
    // LABELS: p-value, accuracy
    // PRIORITY: high
    
    // Rough p-value approximation
    const pValue = 2 * (1 - Math.abs(z) / 4); // FIXME: Very crude approximation
    
    return {
        zStatistic: z,
        pValue: Math.max(0, Math.min(1, pValue)),
        lift: p2 - p1,
        relativeUplift: p1 > 0 ? (p2 - p1) / p1 : 0
    };
}

// TODO: Add confidence interval calculations
// TODO: Add effect size calculations (Cohen's d, etc.)
// TODO: Add sample size calculation functions
// ASSIGNEE: @diogoribeiro7
// LABELS: statistical-functions, browser-stats
// PRIORITY: medium
