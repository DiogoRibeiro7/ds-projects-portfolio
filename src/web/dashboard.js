/**
 * Interactive utilities for A/B testing dashboard and visualizations.
 *
 * This module provides client-side functionality for experiment dashboards.
 * It is framework-agnostic and assumes a plain HTML container exists.
 */

(function () {
    'use strict';

    /**
     * @typedef {Object} ExperimentDashboardConfig
     * @property {string} [dataEndpoint] - Base URL for fetching experiment data.
     * @property {number} [refreshInterval] - Polling interval in ms for real-time updates.
     * @property {number} [maxDataPoints] - Maximum number of time-series points to render.
     * @property {string} [chartLibrary] - "auto" | "chartjs" | "canvas".
     * @property {function(string, string): string} [buildEndpointForRange] - Custom builder for date-range endpoints.
     * @property {string} [websocketUrl] - Optional WebSocket URL for true real-time updates.
     * @property {Object.<string, function(number): string>} [metricFormatters] - Optional per-metric formatters.
     * @property {string} [currencySymbol] - Optional currency symbol for formatting.
     */

    class ExperimentDashboard {
        /**
         * @param {string} containerId - DOM id of the dashboard container.
         * @param {ExperimentDashboardConfig} config - Configuration options.
         */
        constructor(containerId, config = {}) {
            /** @type {HTMLElement | null} */
            this.container = document.getElementById(containerId);

            if (!this.container) {
                console.error(`ExperimentDashboard: container with id "${containerId}" was not found.`);
                throw new Error(`ExperimentDashboard initialization failed: container "${containerId}" is missing.`);
            }

            /** @type {ExperimentDashboardConfig} */
            this.config = {
                refreshInterval: 30000, // 30 seconds
                maxDataPoints: 1000,
                chartLibrary: 'auto',
                currencySymbol: '€',
                ...config
            };

            /** @type {any} */
            this.data = null;
            /** @type {any} */
            this.previousData = null;
            /** @type {Record<string, any>} */
            this.charts = {};
            /** @type {boolean} */
            this.isRealTimeEnabled = false;
            /** @type {number | null} */
            this.updateInterval = null;
            /** @type {WebSocket | null} */
            this.websocket = null;
            /** @type {Array<{target: EventTarget, type: string, handler: EventListenerOrEventListenerObject}>} */
            this._eventHandlers = [];

            this._chartLib = this._detectChartLibrary();

            this._injectBaseStyles();
            this._init();
        }

        /**
         * Detect the chart library to use.
         * @returns {"chartjs" | "canvas"}
         * @private
         */
        _detectChartLibrary() {
            if (this.config.chartLibrary && this.config.chartLibrary !== 'auto') {
                return this.config.chartLibrary;
            }
            // Prefer Chart.js if available.
            if (typeof window !== 'undefined' && window.Chart) {
                return 'chartjs';
            }
            return 'canvas';
        }

        /**
         * Register event listener and keep reference for cleanup.
         * @param {EventTarget | null} target
         * @param {string} type
         * @param {EventListenerOrEventListenerObject} handler
         * @private
         */
        _registerEventListener(target, type, handler) {
            if (!target || typeof target.addEventListener !== 'function') {
                return;
            }
            target.addEventListener(type, handler);
            this._eventHandlers.push({ target, type, handler });
        }

        /**
         * Inject minimal responsive styles and UI helpers.
         * @private
         */
        _injectBaseStyles() {
            if (document.getElementById('experiment-dashboard-base-styles')) {
                return;
            }

            const style = document.createElement('style');
            style.id = 'experiment-dashboard-base-styles';
            style.innerHTML = `
                .dashboard-header {
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                    margin-bottom: 1rem;
                }

                .dashboard-header h1 {
                    margin: 0;
                    font-size: 1.4rem;
                }

                .dashboard-header .controls {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.5rem;
                    align-items: center;
                }

                .dashboard-header .meta {
                    font-size: 0.85rem;
                    color: #666;
                }

                .dashboard-content {
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                }

                .metrics-overview {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                    gap: 0.75rem;
                }

                .metric-card {
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 0.75rem;
                    background: #fff;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
                    transition: box-shadow 0.2s ease, transform 0.2s ease;
                }

                .metric-card.changed-up {
                    box-shadow: 0 0 0 2px rgba(46, 204, 113, 0.4);
                }

                .metric-card.changed-down {
                    box-shadow: 0 0 0 2px rgba(231, 76, 60, 0.4);
                }

                .metric-card h4 {
                    margin: 0 0 0.5rem 0;
                    font-size: 1rem;
                    display: flex;
                    align-items: center;
                    gap: 0.25rem;
                }

                .metric-card .significant {
                    color: #e74c3c;
                    font-weight: bold;
                }

                .metric-values {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 0.25rem 0.75rem;
                    font-size: 0.9rem;
                }

                .metric-values label {
                    display: block;
                    font-size: 0.75rem;
                    color: #666;
                }

                .metric-values span {
                    font-weight: 500;
                }

                .lift-value.positive span {
                    color: #27ae60;
                }

                .lift-value.negative span {
                    color: #c0392b;
                }

                .lift-value.neutral span {
                    color: #7f8c8d;
                }

                .metric-card .p-value,
                .metric-card .confidence-interval {
                    margin-top: 0.4rem;
                    font-size: 0.8rem;
                    color: #555;
                }

                .charts-container {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 1rem;
                }

                .chart-panel {
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 0.75rem;
                    background: #fff;
                }

                .chart-panel h3 {
                    margin: 0 0 0.5rem 0;
                    font-size: 1rem;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 0.5rem;
                }

                .chart-panel .chart-type-select {
                    font-size: 0.8rem;
                }

                .btn {
                    padding: 0.35rem 0.8rem;
                    border-radius: 4px;
                    border: 1px solid transparent;
                    cursor: pointer;
                    font-size: 0.85rem;
                }

                .btn-primary {
                    background: #3498db;
                    border-color: #2980b9;
                    color: #fff;
                }

                .btn-secondary {
                    background: #ecf0f1;
                    border-color: #bdc3c7;
                    color: #333;
                }

                .dashboard-error-container {
                    position: fixed;
                    top: 1rem;
                    right: 1rem;
                    z-index: 9999;
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                }

                .dashboard-toast {
                    background: #c0392b;
                    color: #fff;
                    padding: 0.6rem 0.9rem;
                    border-radius: 4px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    font-size: 0.85rem;
                }

                .dashboard-toast--info {
                    background: #2980b9;
                }

                .dashboard-loading-overlay {
                    position: absolute;
                    inset: 0;
                    background: rgba(255,255,255,0.85);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 999;
                    font-size: 0.95rem;
                    color: #333;
                }

                .dashboard-loading-overlay.hidden {
                    display: none;
                }

                .dashboard-loading-overlay .spinner {
                    border: 3px solid #ecf0f1;
                    border-top: 3px solid #3498db;
                    border-radius: 50%;
                    width: 18px;
                    height: 18px;
                    animation: dash-spinner 0.8s linear infinite;
                    margin-right: 0.5rem;
                }

                @keyframes dash-spinner {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }

                .date-range-controls {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.25rem;
                    align-items: center;
                    font-size: 0.8rem;
                }

                .date-range-controls label {
                    display: flex;
                    align-items: center;
                    gap: 0.25rem;
                }

                .date-range-controls input[type="date"] {
                    font-size: 0.8rem;
                    padding: 0.15rem;
                }

                .dashboard-widgets {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 1rem;
                    margin-top: 1rem;
                }

                .widget-panel {
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 0.75rem;
                    background: #fff;
                }

                .widget-panel h3 {
                    margin: 0 0 0.5rem 0;
                    font-size: 1rem;
                }

                .widget-panel form {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 0.4rem 0.6rem;
                    font-size: 0.85rem;
                }

                .widget-panel form label {
                    display: flex;
                    flex-direction: column;
                    gap: 0.1rem;
                }

                .widget-panel form input,
                .widget-panel form select {
                    font-size: 0.85rem;
                    padding: 0.2rem;
                }

                .widget-panel .widget-results {
                    margin-top: 0.6rem;
                    font-size: 0.85rem;
                }

                @media (min-width: 768px) {
                    .dashboard-header {
                        flex-direction: row;
                        justify-content: space-between;
                        align-items: flex-end;
                    }
                }

                @media (max-width: 600px) {
                    .charts-container {
                        grid-template-columns: 1fr;
                    }

                    .dashboard-header .controls {
                        flex-direction: column;
                        align-items: flex-start;
                    }
                }
            `;
            document.head.appendChild(style);
        }

        /**
         * Initialize the dashboard components.
         * @private
         */
        _init() {
            this._createLayout();
            this._setupEventListeners();

            // Initial data check and load.
            if (this.config.dataEndpoint) {
                this.refreshData();
            } else {
                this._hideLoading();
            }

            this._setupKeyboardShortcuts();
        }

        /**
         * Create the basic dashboard layout.
         * @private
         */
        _createLayout() {
            this.container.style.position = 'relative';

            const layout = `
                <div class="dashboard-header">
                    <div>
                        <h1>A/B Test Results</h1>
                        <div class="meta">
                            <span id="data-timestamp">Last updated: not yet loaded</span>
                        </div>
                    </div>
                    <div class="controls">
                        <button id="refresh-btn" class="btn btn-primary" type="button">Refresh</button>
                        <button id="export-btn" class="btn btn-secondary" type="button">Export</button>
                        <label>
                            <input type="checkbox" id="realtime-toggle">
                            Real-time updates
                        </label>
                        <div class="date-range-controls">
                            <label>
                                From
                                <input type="date" id="date-from">
                            </label>
                            <label>
                                To
                                <input type="date" id="date-to">
                            </label>
                            <button id="apply-range-btn" class="btn btn-secondary" type="button">Apply range</button>
                        </div>
                    </div>
                </div>
                <div class="dashboard-content">
                    <div class="metrics-overview" id="metrics-overview">
                        <!-- Metrics cards will be inserted here -->
                    </div>
                    <div class="charts-container">
                        <div class="chart-panel" id="conversion-chart">
                            <h3>
                                Conversion Rate Over Time
                                <select id="conversion-chart-type" class="chart-type-select">
                                    <option value="line" selected>Line</option>
                                    <option value="bar">Bar</option>
                                    <option value="scatter">Scatter</option>
                                </select>
                            </h3>
                            <canvas id="conversion-canvas"></canvas>
                        </div>
                        <div class="chart-panel" id="distribution-chart">
                            <h3>
                                Metric Distribution
                                <select id="distribution-chart-type" class="chart-type-select">
                                    <option value="bar" selected>Histogram (bar)</option>
                                    <option value="boxplot">Box-approx</option>
                                </select>
                            </h3>
                            <canvas id="distribution-canvas"></canvas>
                        </div>
                        <div class="chart-panel" id="power-chart">
                            <h3>Statistical Power</h3>
                            <canvas id="power-canvas"></canvas>
                        </div>
                    </div>
                    <div class="dashboard-widgets">
                        <div class="widget-panel" id="ab-calculator-widget">
                            <!-- A/B calculator widget rendered here -->
                        </div>
                        <div class="widget-panel" id="experiment-design-wizard">
                            <!-- Experiment design wizard rendered here -->
                        </div>
                    </div>
                </div>
                <div id="dashboard-error-container" class="dashboard-error-container"></div>
                <div id="dashboard-loading" class="dashboard-loading-overlay hidden">
                    <div class="spinner"></div>
                    <span id="dashboard-loading-text">Loading...</span>
                </div>
            `;

            this.container.innerHTML = layout;

            // Initialize standalone widgets.
            const calcContainer = this.container.querySelector('#ab-calculator-widget');
            if (calcContainer) {
                // eslint-disable-next-line no-new
                new ABTestCalculatorWidget(calcContainer.id);
            }

            const wizardContainer = this.container.querySelector('#experiment-design-wizard');
            if (wizardContainer) {
                // eslint-disable-next-line no-new
                new ExperimentDesignWizard(wizardContainer.id);
            }
        }

        /**
         * Setup event handlers for dashboard interactions.
         * @private
         */
        _setupEventListeners() {
            const refreshBtn = document.getElementById('refresh-btn');
            const exportBtn = document.getElementById('export-btn');
            const realtimeToggle = document.getElementById('realtime-toggle');
            const applyRangeBtn = document.getElementById('apply-range-btn');
            const conversionTypeSelect = document.getElementById('conversion-chart-type');
            const distributionTypeSelect = document.getElementById('distribution-chart-type');

            this._registerEventListener(refreshBtn, 'click', () => {
                this.refreshData();
            });

            this._registerEventListener(exportBtn, 'click', () => {
                this._handleExportClick();
            });

            this._registerEventListener(realtimeToggle, 'change', (e) => {
                const target = /** @type {HTMLInputElement} */ (e.target);
                this.toggleRealTime(target.checked);
            });

            // Custom date range selection.
            this._registerEventListener(applyRangeBtn, 'click', () => {
                const fromInput = /** @type {HTMLInputElement | null} */ (document.getElementById('date-from'));
                const toInput = /** @type {HTMLInputElement | null} */ (document.getElementById('date-to'));

                const from = fromInput ? fromInput.value : '';
                const to = toInput ? toInput.value : '';

                if (!this.config.dataEndpoint) {
                    this.showError('No data endpoint configured for date range filtering.');
                    return;
                }

                const endpoint = this._buildEndpointWithRange(
                    this.config.dataEndpoint,
                    from || '',
                    to || ''
                );

                this.refreshData(endpoint);
            });

            // Chart type selection.
            this._registerEventListener(conversionTypeSelect, 'change', () => {
                this._updateConversionChart();
            });

            this._registerEventListener(distributionTypeSelect, 'change', () => {
                this._updateDistributionChart();
            });
        }

        /**
         * Keyboard shortcuts for common actions.
         * r: refresh, e: export, t: toggle real-time
         * @private
         */
        _setupKeyboardShortcuts() {
            const handler = (e) => {
                const activeTag = (document.activeElement && document.activeElement.tagName) || '';
                if (activeTag === 'INPUT' || activeTag === 'TEXTAREA' || activeTag === 'SELECT') {
                    return;
                }

                if (e.key === 'r' || e.key === 'R') {
                    e.preventDefault();
                    this.refreshData();
                } else if (e.key === 'e' || e.key === 'E') {
                    e.preventDefault();
                    this._handleExportClick();
                } else if (e.key === 't' || e.key === 'T') {
                    e.preventDefault();
                    this.toggleRealTime(!this.isRealTimeEnabled);
                    const toggleEl = /** @type {HTMLInputElement | null} */ (document.getElementById('realtime-toggle'));
                    if (toggleEl) toggleEl.checked = this.isRealTimeEnabled;
                }
            };

            this._registerEventListener(document, 'keydown', handler);
        }

        /**
         * Build endpoint with date range parameters.
         * @param {string} baseEndpoint
         * @param {string} fromDate - YYYY-MM-DD or empty.
         * @param {string} toDate - YYYY-MM-DD or empty.
         * @returns {string}
         * @private
         */
        _buildEndpointWithRange(baseEndpoint, fromDate, toDate) {
            if (typeof this.config.buildEndpointForRange === 'function') {
                return this.config.buildEndpointForRange(fromDate, toDate);
            }

            const url = new URL(baseEndpoint, window.location.origin);
            if (fromDate) url.searchParams.set('from', fromDate);
            if (toDate) url.searchParams.set('to', toDate);
            return url.toString();
        }

        /**
         * Show loading overlay with optional text.
         * @param {string} [text]
         * @private
         */
        _showLoading(text = 'Loading data...') {
            const overlay = /** @type {HTMLElement | null} */ (document.getElementById('dashboard-loading'));
            const textEl = /** @type {HTMLElement | null} */ (document.getElementById('dashboard-loading-text'));
            if (!overlay) return;
            overlay.classList.remove('hidden');
            if (textEl) textEl.textContent = text;
        }

        /**
         * Hide loading overlay.
         * @private
         */
        _hideLoading() {
            const overlay = /** @type {HTMLElement | null} */ (document.getElementById('dashboard-loading'));
            if (!overlay) return;
            overlay.classList.add('hidden');
        }

        /**
         * Fetch helper with retry logic and basic exponential backoff.
         * @param {string} endpoint
         * @param {RequestInit} [options]
         * @param {number} [maxRetries]
         * @returns {Promise<Response>}
         * @private
         */
        async _fetchWithRetry(endpoint, options = {}, maxRetries = 3) {
            let attempt = 0;
            let delay = 500;

            while (true) {
                try {
                    const response = await fetch(endpoint, options);
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    return response;
                } catch (error) {
                    attempt += 1;
                    if (attempt > maxRetries) {
                        console.error(`ExperimentDashboard: fetch failed after ${attempt} attempts`, error);
                        throw error;
                    }
                    await new Promise((resolve) => setTimeout(resolve, delay));
                    delay *= 2; // simple exponential backoff
                }
            }
        }

        /**
         * Load experiment data from API endpoint.
         * @param {string} [overrideEndpoint] - Optional override of the configured endpoint.
         * @returns {Promise<void>}
         */
        async loadData(overrideEndpoint) {
            if (!this.config.dataEndpoint && !overrideEndpoint) {
                this.showError('No data endpoint configured.');
                return;
            }

            const endpoint = overrideEndpoint || this.config.dataEndpoint;
            if (!endpoint) return;

            this._showLoading('Loading experiment data...');

            try {
                const response = await this._fetchWithRetry(endpoint);
                const jsonData = await response.json();

                // Basic shape validation.
                if (typeof jsonData !== 'object' || jsonData === null) {
                    throw new Error('Invalid data format: expected JSON object.');
                }

                this.previousData = this.data;
                this.data = jsonData;
                this._updateDashboard();
            } catch (error) {
                console.error('ExperimentDashboard: failed to load data:', error);
                this.showError('Failed to load experiment data. Please try again.');
            } finally {
                this._hideLoading();
            }
        }

        /**
         * Update timestamp indicator.
         * @param {Date} date
         * @private
         */
        _updateDataTimestamp(date) {
            const el = /** @type {HTMLElement | null} */ (document.getElementById('data-timestamp'));
            if (!el) return;
            const iso = date.toISOString();
            el.textContent = `Last updated: ${iso}`;
        }

        /**
         * Update all dashboard components with new data.
         * @private
         */
        _updateDashboard() {
            if (!this.data) {
                console.warn('ExperimentDashboard: no data available for dashboard update');
                return;
            }

            this._updateDataTimestamp(new Date());
            this._updateMetricsOverview();
            this._updateCharts();
        }

        /**
         * Update the metrics overview cards.
         * @private
         */
        _updateMetricsOverview() {
            const overview = document.getElementById('metrics-overview');
            if (!overview) return;

            if (!this.data || typeof this.data.metrics !== 'object') {
                overview.innerHTML = '<p>No metrics available.</p>';
                return;
            }

            /** @type {Record<string, any>} */
            const metrics = this.data.metrics || {};

            let html = '';
            for (const [key, value] of Object.entries(metrics)) {
                const previous = this.previousData && this.previousData.metrics
                    ? this.previousData.metrics[key]
                    : null;

                /** @type {{direction: "up" | "down" | "flat", magnitude: number}} */
                let changeInfo = { direction: 'flat', magnitude: 0 };

                if (previous && typeof previous.lift === 'number' && typeof value.lift === 'number') {
                    const diff = value.lift - previous.lift;
                    if (diff > 0) {
                        changeInfo = { direction: 'up', magnitude: diff };
                    } else if (diff < 0) {
                        changeInfo = { direction: 'down', magnitude: Math.abs(diff) };
                    }
                }

                html += this._createMetricCard(key, value, changeInfo);
            }

            overview.innerHTML = html;
        }

        /**
         * Create a metric card HTML element string.
         * @param {string} name
         * @param {any} data
         * @param {{direction: "up" | "down" | "flat", magnitude: number}} changeInfo
         * @returns {string}
         * @private
         */
        _createMetricCard(name, data, changeInfo) {
            const {
                control = 0,
                treatment = 0,
                lift = 0,
                pValue = null,
                confidenceInterval = null,
                sampleSizeControl = null,
                sampleSizeTreatment = null
            } = data || {};

            const liftClass = lift > 0 ? 'positive' : lift < 0 ? 'negative' : 'neutral';
            const cardChangeClass =
                changeInfo.direction === 'up'
                    ? 'changed-up'
                    : changeInfo.direction === 'down'
                    ? 'changed-down'
                    : '';

            const isSignificant =
                typeof pValue === 'number' && !Number.isNaN(pValue) && pValue < 0.05;

            const significanceIndicator = isSignificant
                ? '<span class="significant" title="Statistically significant at 5%">*</span>'
                : '';

            let trendIcon = '';
            if (changeInfo.direction === 'up') {
                trendIcon = ` <span title="Lift increased vs last update">▲</span>`;
            } else if (changeInfo.direction === 'down') {
                trendIcon = ` <span title="Lift decreased vs last update">▼</span>`;
            }

            const ciHtml =
                confidenceInterval && Array.isArray(confidenceInterval) && confidenceInterval.length === 2
                    ? `<div class="confidence-interval">
                            95% CI: [${this._formatPercentage(confidenceInterval[0])},
                                     ${this._formatPercentage(confidenceInterval[1])}]
                       </div>`
                    : '';

            const pValHtml =
                typeof pValue === 'number'
                    ? `<div class="p-value">p = ${pValue.toFixed(4)}</div>`
                    : '';

            const sampleInfo =
                sampleSizeControl != null && sampleSizeTreatment != null
                    ? `<div class="sample-sizes">
                           <small>n<sub>C</sub>=${sampleSizeControl}, n<sub>T</sub>=${sampleSizeTreatment}</small>
                       </div>`
                    : '';

            return `
                <div class="metric-card ${cardChangeClass}">
                    <h4>${name}${significanceIndicator}${trendIcon}</h4>
                    <div class="metric-values">
                        <div class="control-value">
                            <label>Control:</label>
                            <span>${this._formatMetricValue(name, control)}</span>
                        </div>
                        <div class="treatment-value">
                            <label>Treatment:</label>
                            <span>${this._formatMetricValue(name, treatment)}</span>
                        </div>
                        <div class="lift-value ${liftClass}">
                            <label>Lift:</label>
                            <span>${this._formatPercentage(lift)}</span>
                        </div>
                    </div>
                    ${sampleInfo}
                    ${pValHtml}
                    ${ciHtml}
                </div>
            `;
        }

        /**
         * Update all chart visualizations.
         * @private
         */
        _updateCharts() {
            this._updateConversionChart();
            this._updateDistributionChart();
            this._updatePowerChart();
        }

        /**
         * Build chart data for conversion chart from this.data.
         * Expects this.data.timeSeries to be an array of:
         * { timestamp: string | number | Date, control: number, treatment: number }
         * @returns {{labels: string[], datasets: {label: string, data: number[]}[]}}
         * @private
         */
        _buildConversionChartData() {
            const ts = Array.isArray(this.data && this.data.timeSeries)
                ? this.data.timeSeries
                : [];

            if (!ts.length) {
                return {
                    labels: [],
                    datasets: []
                };
            }

            const maxPoints = this.config.maxDataPoints || 1000;
            const sliced = ts.length > maxPoints ? ts.slice(ts.length - maxPoints) : ts;

            const labels = sliced.map((item) => {
                const d = new Date(item.timestamp);
                if (Number.isNaN(d.getTime())) {
                    return String(item.timestamp);
                }
                return d.toISOString();
            });

            const controlData = sliced.map((item) => Number(item.control) || 0);
            const treatmentData = sliced.map((item) => Number(item.treatment) || 0);

            return {
                labels,
                datasets: [
                    {
                        label: 'Control',
                        data: controlData
                    },
                    {
                        label: 'Treatment',
                        data: treatmentData
                    }
                ]
            };
        }

        /**
         * Update the conversion rate time series chart.
         * @private
         */
        _updateConversionChart() {
            const canvas = /** @type {HTMLCanvasElement | null} */ (document.getElementById('conversion-canvas'));
            if (!canvas) return;

            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            const chartData = this._buildConversionChartData();
            const typeSelect = /** @type {HTMLSelectElement | null} */ (document.getElementById('conversion-chart-type'));
            const chartType = typeSelect ? typeSelect.value : 'line';

            if (!chartData.labels.length || !chartData.datasets.length) {
                // fallback placeholder
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#f0f0f0';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#333';
                ctx.font = '14px Arial';
                ctx.fillText('No time-series data available', 10, 30);
                if (this.charts.conversion) {
                    this.charts.conversion.destroy?.();
                    this.charts.conversion = null;
                }
                return;
            }

            if (this._chartLib === 'chartjs' && typeof window.Chart !== 'undefined') {
                if (this.charts.conversion) {
                    this.charts.conversion.destroy();
                }

                this.charts.conversion = new window.Chart(ctx, {
                    type: chartType === 'scatter' ? 'scatter' : chartType,
                    data: {
                        labels: chartData.labels,
                        datasets: chartData.datasets.map((ds, idx) => ({
                            label: ds.label,
                            data:
                                chartType === 'scatter'
                                    ? ds.data.map((y, i) => ({ x: i, y }))
                                    : ds.data,
                            borderWidth: 1,
                            fill: false
                        }))
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: {
                                display: true,
                                title: {
                                    display: true,
                                    text: 'Time'
                                }
                            },
                            y: {
                                display: true,
                                title: {
                                    display: true,
                                    text: 'Conversion rate'
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                display: true
                            },
                            tooltip: {
                                enabled: true
                            }
                        }
                    }
                });
            } else {
                // Simple canvas rendering fallback.
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#f0f0f0';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#333';
                ctx.font = '14px Arial';
                ctx.fillText('Conversion chart (Chart.js not loaded)', 10, 30);
            }
        }

        /**
         * Build chart data for distribution chart from this.data.
         * Expects this.data.distributions to be:
         * { metric: { control: number[], treatment: number[] }, ... }
         * Uses first metric if multiple present.
         * @returns {{labels: string[], control: number[], treatment: number[], metricName: string | null}}
         * @private
         */
        _buildDistributionData() {
            const distributions = this.data && this.data.distributions;
            if (!distributions || typeof distributions !== 'object') {
                return { labels: [], control: [], treatment: [], metricName: null };
            }

            const entries = Object.entries(distributions);
            if (!entries.length) {
                return { labels: [], control: [], treatment: [], metricName: null };
            }

            const [metricName, values] = entries[0];
            const controlArr = Array.isArray(values.control) ? values.control : [];
            const treatmentArr = Array.isArray(values.treatment) ? values.treatment : [];

            const maxLen = Math.max(controlArr.length, treatmentArr.length);
            const labels = Array.from({ length: maxLen }, (_, i) => `Obs ${i + 1}`);

            return {
                labels,
                control: controlArr,
                treatment: treatmentArr,
                metricName
            };
        }

        /**
         * Update the metric distribution comparison chart.
         * @private
         */
        _updateDistributionChart() {
            const canvas = /** @type {HTMLCanvasElement | null} */ (document.getElementById('distribution-canvas'));
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            const distData = this._buildDistributionData();
            const typeSelect = /** @type {HTMLSelectElement | null} */ (document.getElementById('distribution-chart-type'));
            const chartType = typeSelect ? typeSelect.value : 'bar';

            if (!distData.labels.length) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#f0f0f0';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#333';
                ctx.font = '14px Arial';
                ctx.fillText('No distribution data available', 10, 30);
                if (this.charts.distribution) {
                    this.charts.distribution.destroy?.();
                    this.charts.distribution = null;
                }
                return;
            }

            if (this._chartLib === 'chartjs' && typeof window.Chart !== 'undefined') {
                if (this.charts.distribution) {
                    this.charts.distribution.destroy();
                }

                // For simplicity: "box-approx" uses bar chart with quartiles approximated by simple bars.
                if (chartType === 'boxplot') {
                    const controlStats = approximateBox(distData.control);
                    const treatmentStats = approximateBox(distData.treatment);

                    this.charts.distribution = new window.Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: ['Control', 'Treatment'],
                            datasets: [
                                {
                                    label: 'Median',
                                    data: [controlStats.median, treatmentStats.median],
                                    borderWidth: 1
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: { display: true },
                                tooltip: { enabled: true }
                            },
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: distData.metricName || 'Metric'
                                    }
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Value'
                                    }
                                }
                            }
                        }
                    });
                } else {
                    // Histogram-like bar chart.
                    this.charts.distribution = new window.Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: distData.labels,
                            datasets: [
                                {
                                    label: 'Control',
                                    data: distData.control
                                },
                                {
                                    label: 'Treatment',
                                    data: distData.treatment
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: { display: true },
                                tooltip: { enabled: true }
                            },
                            scales: {
                                x: {
                                    display: false
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: distData.metricName || 'Metric'
                                    }
                                }
                            }
                        }
                    });
                }
            } else {
                // Canvas fallback.
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#f0f0f0';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#333';
                ctx.font = '14px Arial';
                ctx.fillText('Distribution chart (Chart.js not loaded)', 10, 30);
            }
        }

        /**
         * Update statistical power visualization.
         * Uses first binary metric with sample size info if available.
         * @private
         */
        _updatePowerChart() {
            const canvas = /** @type {HTMLCanvasElement | null} */ (document.getElementById('power-canvas'));
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            if (!this.data || !this.data.metrics || typeof this.data.metrics !== 'object') {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#f0f0f0';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#333';
                ctx.font = '14px Arial';
                ctx.fillText('No metrics with sample sizes available', 10, 30);
                if (this.charts.power) {
                    this.charts.power.destroy?.();
                    this.charts.power = null;
                }
                return;
            }

            const metricsEntries = Object.entries(this.data.metrics);
            let chosen = null;
            for (const [, val] of metricsEntries) {
                if (
                    typeof val.control === 'number' &&
                    typeof val.treatment === 'number' &&
                    typeof val.sampleSizeControl === 'number' &&
                    typeof val.sampleSizeTreatment === 'number'
                ) {
                    chosen = val;
                    break;
                }
            }

            if (!chosen) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#f0f0f0';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#333';
                ctx.font = '14px Arial';
                ctx.fillText('No metric with sample size info for power calculation', 10, 30);
                if (this.charts.power) {
                    this.charts.power.destroy?.();
                    this.charts.power = null;
                }
                return;
            }

            const alpha = 0.05;
            const effectSizes = [0.01, 0.02, 0.03, 0.05, 0.1];
            const powers = effectSizes.map((delta) =>
                approximatePowerTwoProportions(
                    chosen.control,
                    chosen.sampleSizeControl,
                    chosen.control + delta,
                    chosen.sampleSizeTreatment,
                    alpha
                )
            );

            if (this._chartLib === 'chartjs' && typeof window.Chart !== 'undefined') {
                if (this.charts.power) {
                    this.charts.power.destroy();
                }

                this.charts.power = new window.Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: effectSizes.map((d) => `${(d * 100).toFixed(1)}pp`),
                        datasets: [
                            {
                                label: 'Power vs uplift (percentage points)',
                                data: powers,
                                borderWidth: 1,
                                fill: false
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                min: 0,
                                max: 1,
                                title: {
                                    display: true,
                                    text: 'Power'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Absolute uplift (percentage points)'
                                }
                            }
                        },
                        plugins: {
                            legend: { display: false }
                        }
                    }
                });
            } else {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#f0f0f0';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#333';
                ctx.font = '14px Arial';
                ctx.fillText('Power chart (Chart.js not loaded)', 10, 30);
            }
        }

        /**
         * Smart formatting for metric values based on name and configuration.
         * @param {string} name
         * @param {any} value
         * @returns {string}
         * @private
         */
        _formatMetricValue(name, value) {
            if (this.config.metricFormatters && this.config.metricFormatters[name]) {
                return this.config.metricFormatters[name](Number(value) || 0);
            }

            if (typeof value !== 'number' || Number.isNaN(value)) {
                return String(value);
            }

            const lowerName = name.toLowerCase();

            if (lowerName.includes('revenue') || lowerName.includes('gmv') || lowerName.includes('amount')) {
                return `${this.config.currencySymbol}${value.toLocaleString(undefined, {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                })}`;
            }

            if (lowerName.includes('rate') || lowerName.includes('ratio') || lowerName.includes('prob')) {
                return this._formatPercentage(value);
            }

            if (Math.abs(value) < 1 && !Number.isInteger(value)) {
                return value.toFixed(4);
            }

            return value.toLocaleString();
        }

        /**
         * Format percentage values for display.
         * @param {number} value
         * @returns {string}
         * @private
         */
        _formatPercentage(value) {
            if (typeof value !== 'number' || Number.isNaN(value)) {
                return '0.00%';
            }
            return `${(value * 100).toFixed(2)}%`;
        }

        /**
         * Manually refresh dashboard data.
         * @param {string} [endpointOverride]
         */
        refreshData(endpointOverride) {
            const text = endpointOverride ? 'Loading filtered data...' : 'Refreshing data...';
            this._showLoading(text);
            this.loadData(endpointOverride);
        }

        /**
         * Enable/disable real-time data updates.
         * Uses WebSocket if configured, otherwise falls back to polling.
         * @param {boolean} enabled
         */
        toggleRealTime(enabled) {
            this.isRealTimeEnabled = enabled;

            if (enabled) {
                if (this.config.websocketUrl) {
                    this._startWebSocketUpdates();
                } else {
                    this._startRealTimePolling();
                }
            } else {
                this._stopRealTimePolling();
                this._stopWebSocketUpdates();
            }
        }

        /**
         * Start polling for real-time updates.
         * @private
         */
        _startRealTimePolling() {
            if (this.updateInterval) {
                clearInterval(this.updateInterval);
            }

            this.updateInterval = window.setInterval(() => {
                this.refreshData();
            }, this.config.refreshInterval);
        }

        /**
         * Stop real-time polling.
         * @private
         */
        _stopRealTimePolling() {
            if (this.updateInterval) {
                clearInterval(this.updateInterval);
                this.updateInterval = null;
            }
        }

        /**
         * Start WebSocket-based updates if websocketUrl is configured.
         * Expects server to send JSON payloads compatible with this.data structure.
         * @private
         */
        _startWebSocketUpdates() {
            if (!this.config.websocketUrl) {
                this.showError('WebSocket URL not configured for real-time updates.');
                return;
            }

            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                return;
            }

            this._stopRealTimePolling();

            try {
                this.websocket = new WebSocket(this.config.websocketUrl);

                this.websocket.onopen = () => {
                    this._showToast('Connected to real-time stream.', true);
                };

                this.websocket.onmessage = (event) => {
                    try {
                        const payload = JSON.parse(event.data);
                        if (payload && typeof payload === 'object') {
                            this.previousData = this.data;
                            this.data = payload;
                            this._updateDashboard();
                        }
                    } catch (e) {
                        console.error('ExperimentDashboard: failed to parse WebSocket message', e);
                    }
                };

                this.websocket.onerror = (event) => {
                    console.error('ExperimentDashboard: WebSocket error', event);
                    this.showError('Real-time connection error. Falling back to polling.');
                    this._startRealTimePolling();
                };

                this.websocket.onclose = () => {
                    if (this.isRealTimeEnabled) {
                        this._showToast('Real-time connection closed; resuming polling.', true);
                        this._startRealTimePolling();
                    }
                };
            } catch (error) {
                console.error('ExperimentDashboard: failed to open WebSocket', error);
                this.showError('Failed to open real-time connection. Using polling instead.');
                this._startRealTimePolling();
            }
        }

        /**
         * Stop WebSocket-based updates.
         * @private
         */
        _stopWebSocketUpdates() {
            if (this.websocket) {
                try {
                    this.websocket.close();
                } catch (e) {
                    // ignore
                }
                this.websocket = null;
            }
        }

        /**
         * Handle export button click, asking user for format.
         * @private
         */
        _handleExportClick() {
            if (!this.data) {
                this.showError('No data available to export.');
                return;
            }

            const format = (window.prompt('Choose export format: json, csv, png', 'json') || '').toLowerCase();
            if (!format || !['json', 'csv', 'png'].includes(format)) {
                this.showError('Unsupported export format. Use json, csv, or png.');
                return;
            }

            this._showLoading('Preparing export...');
            try {
                if (format === 'json') {
                    this._exportAsJson();
                } else if (format === 'csv') {
                    this._exportAsCsv();
                } else if (format === 'png') {
                    this._exportAsPng();
                }
            } finally {
                this._hideLoading();
            }
        }

        /**
         * Export as JSON file.
         * @private
         */
        _exportAsJson() {
            const dataStr = JSON.stringify(this.data, null, 2);
            const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
            const exportFileDefaultName = `experiment_results_${new Date().toISOString().split('T')[0]}.json`;

            const linkElement = document.createElement('a');
            linkElement.setAttribute('href', dataUri);
            linkElement.setAttribute('download', exportFileDefaultName);
            document.body.appendChild(linkElement);
            linkElement.click();
            linkElement.remove();
        }

        /**
         * Export metrics table as simple CSV.
         * @private
         */
        _exportAsCsv() {
            if (!this.data || !this.data.metrics) {
                this.showError('No metrics available to export as CSV.');
                return;
            }

            const lines = [];
            lines.push([
                'metric',
                'control',
                'treatment',
                'lift',
                'pValue',
                'ciLower',
                'ciUpper',
                'sampleSizeControl',
                'sampleSizeTreatment'
            ].join(','));

            for (const [name, m] of Object.entries(this.data.metrics)) {
                const ci = Array.isArray(m.confidenceInterval) ? m.confidenceInterval : [null, null];
                const row = [
                    name,
                    m.control ?? '',
                    m.treatment ?? '',
                    m.lift ?? '',
                    m.pValue ?? '',
                    ci[0] ?? '',
                    ci[1] ?? '',
                    m.sampleSizeControl ?? '',
                    m.sampleSizeTreatment ?? ''
                ];
                lines.push(row.join(','));
            }

            const csvStr = lines.join('\n');
            const dataUri = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csvStr);
            const exportFileDefaultName = `experiment_metrics_${new Date().toISOString().split('T')[0]}.csv`;

            const linkElement = document.createElement('a');
            linkElement.setAttribute('href', dataUri);
            linkElement.setAttribute('download', exportFileDefaultName);
            document.body.appendChild(linkElement);
            linkElement.click();
            linkElement.remove();
        }

        /**
         * Export main charts as PNG (conversion + distribution in one image if possible).
         * @private
         */
        _exportAsPng() {
            const conversionCanvas = /** @type {HTMLCanvasElement | null} */ (document.getElementById('conversion-canvas'));
            const distributionCanvas = /** @type {HTMLCanvasElement | null} */ (document.getElementById('distribution-canvas'));

            if (!conversionCanvas && !distributionCanvas) {
                this.showError('No charts available to export as PNG.');
                return;
            }

            // If only one chart exists, export it.
            if (conversionCanvas && !distributionCanvas) {
                this._downloadCanvasAsPng(conversionCanvas, 'conversion_chart');
                return;
            }
            if (!conversionCanvas && distributionCanvas) {
                this._downloadCanvasAsPng(distributionCanvas, 'distribution_chart');
                return;
            }

            // Combine two canvases side by side into a temporary canvas.
            const width = conversionCanvas.width + distributionCanvas.width;
            const height = Math.max(conversionCanvas.height, distributionCanvas.height);
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = width;
            tempCanvas.height = height;
            const ctx = tempCanvas.getContext('2d');
            if (!ctx) {
                this.showError('Failed to prepare combined PNG export.');
                return;
            }

            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, width, height);
            ctx.drawImage(conversionCanvas, 0, 0);
            ctx.drawImage(distributionCanvas, conversionCanvas.width, 0);

            this._downloadCanvasAsPng(tempCanvas, 'experiment_charts');
        }

        /**
         * Download a canvas as PNG.
         * @param {HTMLCanvasElement} canvas
         * @param {string} baseName
         * @private
         */
        _downloadCanvasAsPng(canvas, baseName) {
            const dataUrl = canvas.toDataURL('image/png');
            const link = document.createElement('a');
            link.href = dataUrl;
            link.download = `${baseName}_${new Date().toISOString().split('T')[0]}.png`;
            document.body.appendChild(link);
            link.click();
            link.remove();
        }

        /**
         * Display error message to user as toast.
         * @param {string} message
         */
        showError(message) {
            console.error('ExperimentDashboard error:', message);
            this._showToast(message, false);
        }

        /**
         * Show toast notification.
         * @param {string} message
         * @param {boolean} [isInfo]
         * @private
         */
        _showToast(message, isInfo = false) {
            const container = /** @type {HTMLElement | null} */ (document.getElementById('dashboard-error-container'));
            if (!container) {
                // Fallback for environments where toast container is missing.
                // eslint-disable-next-line no-alert
                alert(`Error: ${message}`);
                return;
            }

            const toast = document.createElement('div');
            toast.className = `dashboard-toast ${isInfo ? 'dashboard-toast--info' : ''}`;
            toast.textContent = message;
            container.appendChild(toast);

            setTimeout(() => {
                toast.remove();
            }, 4000);
        }

        /**
         * Clean up dashboard resources: event listeners, intervals, charts, sockets.
         */
        destroy() {
            this._stopRealTimePolling();
            this._stopWebSocketUpdates();

            // Remove event listeners.
            for (const { target, type, handler } of this._eventHandlers) {
                try {
                    target.removeEventListener(type, handler);
                } catch (e) {
                    // ignore
                }
            }
            this._eventHandlers = [];

            // Destroy charts.
            Object.keys(this.charts).forEach((key) => {
                if (this.charts[key] && typeof this.charts[key].destroy === 'function') {
                    this.charts[key].destroy();
                }
                this.charts[key] = null;
            });

            // Optionally clear container.
            // this.container.innerHTML = '';
        }
    }

    /**
     * Standalone A/B test calculator widget.
     * Allows sample size and power calculations.
     */
    class ABTestCalculatorWidget {
        /**
         * @param {string} containerId
         */
        constructor(containerId) {
            /** @type {HTMLElement | null} */
            this.container = document.getElementById(containerId);
            if (!this.container) {
                console.error(`ABTestCalculatorWidget: container "${containerId}" not found.`);
                return;
            }

            this._render();
            this._setupEvents();
        }

        /**
         * Render widget UI.
         * @private
         */
        _render() {
            this.container.innerHTML = `
                <h3>A/B Test Calculator</h3>
                <form id="ab-calculator-form">
                    <label>
                        Baseline conversion (p<sub>0</sub>)
                        <input type="number" id="baseline-rate" min="0" max="1" step="0.0001" value="0.1">
                    </label>
                    <label>
                        Minimum detectable uplift (absolute, p<sub>1</sub>-p<sub>0</sub>)
                        <input type="number" id="uplift" min="0" max="1" step="0.0001" value="0.02">
                    </label>
                    <label>
                        Alpha (significance)
                        <input type="number" id="alpha" min="0.0001" max="0.5" step="0.0001" value="0.05">
                    </label>
                    <label>
                        Desired power
                        <input type="number" id="power" min="0.1" max="0.9999" step="0.0001" value="0.8">
                    </label>
                    <label>
                        Allocation ratio (T : C)
                        <input type="number" id="allocation-ratio" min="0.1" step="0.1" value="1">
                    </label>
                    <div style="grid-column: 1 / -1; margin-top: 0.4rem;">
                        <button type="submit" class="btn btn-primary">Compute sample size</button>
                    </div>
                </form>
                <div class="widget-results" id="ab-calculator-results"></div>
            `;
        }

        /**
         * Setup events.
         * @private
         */
        _setupEvents() {
            const form = /** @type {HTMLFormElement | null} */ (this.container.querySelector('#ab-calculator-form'));
            const results = /** @type {HTMLElement | null} */ (this.container.querySelector('#ab-calculator-results'));

            if (!form || !results) return;

            form.addEventListener('submit', (e) => {
                e.preventDefault();

                const baseline = parseFloat((/** @type {HTMLInputElement} */ (form.querySelector('#baseline-rate'))).value);
                const uplift = parseFloat((/** @type {HTMLInputElement} */ (form.querySelector('#uplift'))).value);
                const alpha = parseFloat((/** @type {HTMLInputElement} */ (form.querySelector('#alpha'))).value);
                const power = parseFloat((/** @type {HTMLInputElement} */ (form.querySelector('#power'))).value);
                const ratio = parseFloat((/** @type {HTMLInputElement} */ (form.querySelector('#allocation-ratio'))).value);

                if ([baseline, uplift, alpha, power, ratio].some((v) => Number.isNaN(v))) {
                    results.textContent = 'Please fill all fields with valid numbers.';
                    return;
                }

                try {
                    const sample = calculateSampleSizeTwoProportions(baseline, baseline + uplift, alpha, power, ratio);
                    results.innerHTML = `
                        <div>Required sample size (Control): <strong>${Math.ceil(sample.nControl)}</strong></div>
                        <div>Required sample size (Treatment): <strong>${Math.ceil(sample.nTreatment)}</strong></div>
                        <div>Total sample size: <strong>${Math.ceil(sample.nControl + sample.nTreatment)}</strong></div>
                    `;
                } catch (error) {
                    console.error('ABTestCalculatorWidget error:', error);
                    results.textContent = 'Error computing sample size. Check input values.';
                }
            });
        }
    }

    /**
     * Simple experiment design wizard.
     */
    class ExperimentDesignWizard {
        /**
         * @param {string} containerId
         */
        constructor(containerId) {
            /** @type {HTMLElement | null} */
            this.container = document.getElementById(containerId);
            if (!this.container) {
                console.error(`ExperimentDesignWizard: container "${containerId}" not found.`);
                return;
            }

            /** @type {number} */
            this.currentStep = 0;
            this._render();
            this._setupEvents();
        }

        /**
         * Render wizard layout.
         * @private
         */
        _render() {
            this.container.innerHTML = `
                <h3>Experiment Design Wizard</h3>
                <div id="wizard-steps">
                    <div class="wizard-step" data-step="0">
                        <p><strong>Step 1:</strong> Define hypothesis and primary metric.</p>
                        <label>
                            Hypothesis
                            <input type="text" id="wizard-hypothesis" placeholder="e.g. Treatment increases signup rate">
                        </label>
                        <label>
                            Primary metric
                            <input type="text" id="wizard-metric" placeholder="e.g. Signup conversion rate">
                        </label>
                    </div>
                    <div class="wizard-step" data-step="1" style="display:none;">
                        <p><strong>Step 2:</strong> Define guardrails and constraints.</p>
                        <label>
                            Guardrail metrics
                            <input type="text" id="wizard-guardrails" placeholder="e.g. Revenue, churn rate">
                        </label>
                        <label>
                            Max acceptable risk
                            <input type="text" id="wizard-risk" placeholder="e.g. No more than 2% drop in guardrails">
                        </label>
                    </div>
                    <div class="wizard-step" data-step="2" style="display:none;">
                        <p><strong>Step 3:</strong> Define duration and rollout plan.</p>
                        <label>
                            Target duration (days)
                            <input type="number" id="wizard-duration" min="1" value="14">
                        </label>
                        <label>
                            Rollout strategy
                            <input type="text" id="wizard-rollout" placeholder="e.g. 50/50 split, staggered rollout">
                        </label>
                    </div>
                </div>
                <div style="margin-top:0.5rem; display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <button type="button" id="wizard-prev" class="btn btn-secondary" disabled>Previous</button>
                        <button type="button" id="wizard-next" class="btn btn-primary">Next</button>
                    </div>
                    <button type="button" id="wizard-summary" class="btn btn-secondary">Generate summary</button>
                </div>
                <div class="widget-results" id="wizard-summary-output" style="margin-top:0.6rem;"></div>
            `;
        }

        /**
         * Setup wizard events.
         * @private
         */
        _setupEvents() {
            const prevBtn = /** @type {HTMLButtonElement | null} */ (this.container.querySelector('#wizard-prev'));
            const nextBtn = /** @type {HTMLButtonElement | null} */ (this.container.querySelector('#wizard-next'));
            const summaryBtn = /** @type {HTMLButtonElement | null} */ (this.container.querySelector('#wizard-summary'));
            const steps = /** @type {NodeListOf<HTMLElement>} */ (this.container.querySelectorAll('.wizard-step'));
            const summaryOutput = /** @type {HTMLElement | null} */ (this.container.querySelector('#wizard-summary-output'));

            if (!prevBtn || !nextBtn || !summaryBtn || !steps.length || !summaryOutput) {
                return;
            }

            const updateStepVisibility = () => {
                steps.forEach((stepEl, index) => {
                    stepEl.style.display = index === this.currentStep ? 'block' : 'none';
                });
                prevBtn.disabled = this.currentStep === 0;
                nextBtn.disabled = this.currentStep === steps.length - 1;
            };

            prevBtn.addEventListener('click', () => {
                if (this.currentStep > 0) {
                    this.currentStep -= 1;
                    updateStepVisibility();
                }
            });

            nextBtn.addEventListener('click', () => {
                if (this.currentStep < steps.length - 1) {
                    this.currentStep += 1;
                    updateStepVisibility();
                }
            });

            summaryBtn.addEventListener('click', () => {
                const hypothesis = /** @type {HTMLInputElement | null} */ (this.container.querySelector('#wizard-hypothesis'));
                const metric = /** @type {HTMLInputElement | null} */ (this.container.querySelector('#wizard-metric'));
                const guardrails = /** @type {HTMLInputElement | null} */ (this.container.querySelector('#wizard-guardrails'));
                const risk = /** @type {HTMLInputElement | null} */ (this.container.querySelector('#wizard-risk'));
                const duration = /** @type {HTMLInputElement | null} */ (this.container.querySelector('#wizard-duration'));
                const rollout = /** @type {HTMLInputElement | null} */ (this.container.querySelector('#wizard-rollout'));

                const hypothesisText = hypothesis?.value || '(not specified)';
                const metricText = metric?.value || '(not specified)';
                const guardrailsText = guardrails?.value || '(not specified)';
                const riskText = risk?.value || '(not specified)';
                const durationText = duration?.value || '(not specified)';
                const rolloutText = rollout?.value || '(not specified)';

                summaryOutput.innerHTML = `
                    <strong>Experiment design summary</strong><br>
                    Hypothesis: ${hypothesisText}<br>
                    Primary metric: ${metricText}<br>
                    Guardrail metrics: ${guardrailsText}<br>
                    Risk tolerance: ${riskText}<br>
                    Target duration: ${durationText} days<br>
                    Rollout plan: ${rolloutText}
                `;
            });

            updateStepVisibility();
        }
    }

    /**
     * Utility: approximate boxplot statistics (median only for now).
     * @param {number[]} arr
     * @returns {{median: number}}
     */
    function approximateBox(arr) {
        if (!arr || !arr.length) return { median: 0 };
        const sorted = [...arr].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        const median =
            sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
        return { median };
    }

    /**
     * Utility functions for statistical calculations in the browser.
     *
     * Note: These are simplified implementations for demonstration.
     * For production use, consider a dedicated statistical library.
     */

    /**
     * Cumulative distribution function for standard normal distribution.
     * @param {number} x
     * @returns {number}
     */
    function normalCdf(x) {
        // Abramowitz-Stegun approximation for erf.
        const sign = x < 0 ? -1 : 1;
        const absX = Math.abs(x) / Math.sqrt(2);
        const t = 1 / (1 + 0.3275911 * absX);
        const a1 = 0.254829592;
        const a2 = -0.284496736;
        const a3 = 1.421413741;
        const a4 = -1.453152027;
        const a5 = 1.061405429;
        const erf =
            1 -
            (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) *
                t *
                Math.exp(-absX * absX);
        return 0.5 * (1 + sign * erf);
    }

    /**
     * Inverse CDF (quantile) for standard normal.
     * Approximation by Peter John Acklam.
     * @param {number} p
     * @returns {number}
     */
    function normalInv(p) {
        if (p <= 0 || p >= 1) {
            throw new Error('normalInv: p must be in (0,1)');
        }
        const a = [
            -3.969683028665376e1,
            2.209460984245205e2,
            -2.759285104469687e2,
            1.38357751867269e2,
            -3.066479806614716e1,
            2.506628277459239e0
        ];
        const b = [
            -5.447609879822406e1,
            1.615858368580409e2,
            -1.556989798598866e2,
            6.680131188771972e1,
            -1.328068155288572e1
        ];
        const c = [
            -7.784894002430293e-3,
            -3.223964580411365e-1,
            -2.400758277161838e0,
            -2.549732539343734e0,
            4.374664141464968e0,
            2.938163982698783e0
        ];
        const d = [
            7.784695709041462e-3,
            3.224671290700398e-1,
            2.445134137142996e0,
            3.754408661907416e0
        ];
        const plow = 0.02425;
        const phigh = 1 - plow;
        let q, r;
        let ret;

        if (p < plow) {
            q = Math.sqrt(-2 * Math.log(p));
            ret =
                (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        } else if (p > phigh) {
            q = Math.sqrt(-2 * Math.log(1 - p));
            ret =
                -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        } else {
            q = p - 0.5;
            r = q * q;
            ret =
                (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
                q /
                (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
        }
        return ret;
    }

    /**
     * Simple two-proportion z-test implementation.
     *
     * @param {number} x1 - Successes in control.
     * @param {number} n1 - Trials in control.
     * @param {number} x2 - Successes in treatment.
     * @param {number} n2 - Trials in treatment.
     * @returns {{
     *   zStatistic: number,
     *   pValue: number,
     *   lift: number,
     *   relativeUplift: number
     * }}
     */
    function calculateTwoProportionTest(x1, n1, x2, n2) {
        if (n1 <= 0 || n2 <= 0) {
            throw new Error('calculateTwoProportionTest: sample sizes must be positive.');
        }
        if (x1 < 0 || x2 < 0 || x1 > n1 || x2 > n2) {
            throw new Error('calculateTwoProportionTest: invalid counts.');
        }

        const p1 = x1 / n1;
        const p2 = x2 / n2;
        const pPool = (x1 + x2) / (n1 + n2);

        const se = Math.sqrt(pPool * (1 - pPool) * (1 / n1 + 1 / n2));
        if (se === 0) {
            return {
                zStatistic: 0,
                pValue: 1,
                lift: p2 - p1,
                relativeUplift: p1 > 0 ? (p2 - p1) / p1 : 0
            };
        }
        const z = (p2 - p1) / se;

        // Two-sided p-value using normal CDF.
        const pValue = 2 * (1 - normalCdf(Math.abs(z)));

        return {
            zStatistic: z,
            pValue: Math.max(0, Math.min(1, pValue)),
            lift: p2 - p1,
            relativeUplift: p1 > 0 ? (p2 - p1) / p1 : 0
        };
    }

    /**
     * Confidence interval for a single proportion using normal approximation.
     * @param {number} x - successes
     * @param {number} n - trials
     * @param {number} [alpha] - significance level
     * @returns {{lower: number, upper: number}}
     */
    function calculateProportionConfidenceInterval(x, n, alpha = 0.05) {
        if (n <= 0) {
            throw new Error('calculateProportionConfidenceInterval: n must be > 0');
        }
        const p = x / n;
        const z = normalInv(1 - alpha / 2);
        const se = Math.sqrt(p * (1 - p) / n);
        return {
            lower: Math.max(0, p - z * se),
            upper: Math.min(1, p + z * se)
        };
    }

    /**
     * Cohen's d for two independent samples.
     * @param {number[]} control
     * @param {number[]} treatment
     * @returns {number}
     */
    function calculateCohensD(control, treatment) {
        if (!control.length || !treatment.length) {
            throw new Error('calculateCohensD: both samples must be non-empty.');
        }

        const mean = (arr) => arr.reduce((s, v) => s + v, 0) / arr.length;
        const variance = (arr, m) =>
            arr.reduce((s, v) => s + (v - m) * (v - m), 0) / (arr.length - 1);

        const m1 = mean(control);
        const m2 = mean(treatment);
        const v1 = variance(control, m1);
        const v2 = variance(treatment, m2);

        const pooledStd = Math.sqrt(
            ((control.length - 1) * v1 + (treatment.length - 1) * v2) /
                (control.length + treatment.length - 2)
        );
        if (pooledStd === 0) return 0;
        return (m2 - m1) / pooledStd;
    }

    /**
     * Sample size for two-proportion z-test (approximate).
     * @param {number} p1 - baseline
     * @param {number} p2 - variant
     * @param {number} alpha
     * @param {number} power
     * @param {number} allocationRatio - nTreatment / nControl
     * @returns {{nControl: number, nTreatment: number}}
     */
    function calculateSampleSizeTwoProportions(p1, p2, alpha, power, allocationRatio = 1) {
        if (p1 <= 0 || p1 >= 1 || p2 <= 0 || p2 >= 1) {
            throw new Error('calculateSampleSizeTwoProportions: p1 and p2 must be in (0,1).');
        }
        if (alpha <= 0 || alpha >= 1 || power <= 0 || power >= 1) {
            throw new Error('calculateSampleSizeTwoProportions: alpha and power must be in (0,1).');
        }
        if (allocationRatio <= 0) {
            throw new Error('calculateSampleSizeTwoProportions: allocationRatio must be > 0.');
        }

        const zAlpha = normalInv(1 - alpha / 2);
        const zBeta = normalInv(power);
        const diff = Math.abs(p2 - p1);
        const pBar = (p1 + p2) / 2;

        const numerator =
            (zAlpha * Math.sqrt(2 * pBar * (1 - pBar)) + zBeta * Math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) **
            2;
        const nPerArm = numerator / (diff * diff);

        const nControl = nPerArm * (1 / (1 + allocationRatio));
        const nTreatment = nPerArm * (allocationRatio / (1 + allocationRatio));
        return { nControl, nTreatment };
    }

    /**
     * Approximate power for two-proportion z-test with given effect size.
     * @param {number} p1
     * @param {number} n1
     * @param {number} p2
     * @param {number} n2
     * @param {number} alpha
     * @returns {number}
     */
    function approximatePowerTwoProportions(p1, n1, p2, n2, alpha) {
        if (n1 <= 0 || n2 <= 0) return 0;
        const seAlt = Math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2);
        if (seAlt === 0) return 0;
        const zAlpha = normalInv(1 - alpha / 2);
        const diff = Math.abs(p2 - p1);
        const z = diff / seAlt;
        const power = normalCdf(z - zAlpha) + (1 - normalCdf(z + zAlpha));
        return Math.max(0, Math.min(1, power));
    }

    // Export for module systems
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = {
            ExperimentDashboard,
            ABTestCalculatorWidget,
            ExperimentDesignWizard,
            calculateTwoProportionTest,
            calculateProportionConfidenceInterval,
            calculateCohensD,
            calculateSampleSizeTwoProportions
        };
    }

    // Global export for browser environments (non-module)
    if (typeof window !== 'undefined') {
        window.ExperimentDashboard = ExperimentDashboard;
        window.ABTestCalculatorWidget = ABTestCalculatorWidget;
        window.ExperimentDesignWizard = ExperimentDesignWizard;
        window.calculateTwoProportionTest = calculateTwoProportionTest;
    }
})();
