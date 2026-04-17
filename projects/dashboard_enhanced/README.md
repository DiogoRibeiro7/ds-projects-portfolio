# 📊 Enhanced Dashboard System

A comprehensive, production-ready dashboard system with real-time data streaming, interactive visualizations, API infrastructure, and extensive testing capabilities.

## 🌟 Features

### Dashboard Framework
- ✅ **Real-time Data Streaming**: WebSocket and Redis integration for live updates
- ✅ **Interactive Filtering**: Dynamic data filtering with date ranges, categories, and custom parameters
- ✅ **Export Functionality**: PDF, PowerPoint, Excel export capabilities
- ✅ **Mobile-Responsive Design**: Adaptive layouts for all screen sizes
- ✅ **Dark Mode Support**: Toggle between light and dark themes
- ✅ **Accessibility Features**: ARIA labels, keyboard navigation, high-contrast modes

### Visualization Components
- ✅ **Plotly Integration**: Interactive charts with animations
- ✅ **Bokeh Support**: Advanced interactive plots
- ✅ **3D Visualizations**: Interactive 3D scatter plots
- ✅ **Animated Time Series**: Play/pause controls for temporal data
- ✅ **Sankey Diagrams**: Flow visualization
- ✅ **Sunburst Charts**: Hierarchical data representation
- ✅ **Parallel Coordinates**: Multi-dimensional analysis
- ✅ **Data Storytelling Templates**: Narrative-driven visualizations

### API Infrastructure
- ✅ **REST API**: Full CRUD operations with authentication
- ✅ **GraphQL Endpoint**: Flexible data queries
- ✅ **WebSocket Support**: Real-time bidirectional communication
- ✅ **JWT Authentication**: Secure token-based auth
- ✅ **Rate Limiting**: API protection and throttling
- ✅ **Swagger/OpenAPI Documentation**: Interactive API docs

### Testing Suite
- ✅ **Unit Tests**: Component-level testing
- ✅ **End-to-End Tests**: Selenium and Playwright integration
- ✅ **Performance Testing**: Locust-based load testing
- ✅ **Visual Regression Tests**: Automated screenshot comparison
- ✅ **Cross-Browser Testing**: Chrome, Firefox, Edge compatibility

## 📁 Project Structure

```
dashboard_enhanced/
├── dashboard_framework.py       # Core dashboard framework
├── visualization_components.py  # Interactive visualization library
├── api_infrastructure.py       # REST, GraphQL, WebSocket APIs
├── testing_suite.py            # Comprehensive testing framework
├── example_dashboard.py        # Complete example application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── tests/                      # Test artifacts
    ├── visual_baselines/       # Visual regression baselines
    └── visual_diffs/           # Visual differences
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd dashboard_enhanced
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install Redis** (optional, for real-time features):
```bash
# On Ubuntu/Debian
sudo apt-get install redis-server

# On macOS
brew install redis

# On Windows
# Download from https://github.com/microsoftarchive/redis/releases
```

5. **Install browser drivers** (for testing):
```bash
# For Selenium
pip install webdriver-manager

# For Playwright
playwright install
```

### Running the Dashboard

#### Basic Usage

```python
from dashboard_framework import EnhancedDashboard, DashboardConfig

# Configure dashboard
config = DashboardConfig(
    app_name="My Dashboard",
    enable_realtime=True,
    enable_dark_mode=True
)

# Create dashboard
dashboard = EnhancedDashboard(config)

# Register data source
def get_data(filters=None):
    return pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

dashboard.register_data_source('my_data', get_data)

# Run
dashboard.run()
```

#### Run Example Dashboard

```bash
python example_dashboard.py
```

Then open:
- Dashboard: http://localhost:8050
- API: http://localhost:5000
- API Docs: http://localhost:5000/api/docs

## 📖 API Usage

### REST API

```python
import requests

# Register user
response = requests.post('http://localhost:5000/api/auth/register', json={
    'username': 'user',
    'password': 'pass123'
})

# Login
response = requests.post('http://localhost:5000/api/auth/login',
    auth=('user', 'pass123'))
token = response.json()['token']

# Get data
headers = {'Authorization': f'Bearer {token}'}
response = requests.get('http://localhost:5000/api/v1/data', headers=headers)
```

### GraphQL

```python
query = """
{
  allMetrics {
    name
    value
    timestamp
  }
}
"""

response = requests.post('http://localhost:5000/graphql',
    json={'query': query}, headers=headers)
```

### WebSocket

```javascript
const socket = io('http://localhost:5000');

socket.on('connect', () => {
    console.log('Connected');
    socket.emit('subscribe', {channel: 'metrics'});
});

socket.on('realtime_data', (data) => {
    console.log('Received:', data);
});
```

## 🧪 Testing

### Run All Tests

```bash
python testing_suite.py
```

### Run Specific Test Categories

```bash
# Unit tests
pytest testing_suite.py::TestDashboardFramework -v

# E2E tests with Selenium
python -m unittest testing_suite.SeleniumE2ETests

# E2E tests with Playwright
pytest testing_suite.py::PlaywrightE2ETests -v

# Performance tests with Locust
locust -f testing_suite.py DashboardUser --headless -u 10 -r 2 -t 30s

# Visual regression tests
python -c "from testing_suite import VisualRegressionTests; vrt = VisualRegressionTests(); vrt.test_component_visual_regression('dashboard')"
```

## 🎨 Creating Custom Visualizations

```python
from visualization_components import InteractiveVisualizations

viz = InteractiveVisualizations()

# Create animated time series
fig = viz.create_animated_time_series(
    df, x_col='date', y_cols=['metric1', 'metric2'],
    title='My Animated Chart'
)

# Create 3D scatter
fig_3d = viz.create_interactive_3d_scatter(
    df, x_col='x', y_col='y', z_col='z',
    color_col='category', title='3D Visualization'
)

# Create accessible chart
fig_accessible = viz.create_accessibility_chart(
    df, chart_type='bar', title='Accessible Chart'
)
```

## 📤 Export Options

### PDF Export

```python
figures = [fig1, fig2, fig3]
dashboard.export_to_pdf(figures, 'report.pdf')
```

### PowerPoint Export

```python
dashboard.export_to_powerpoint(figures, 'presentation.pptx')
```

### Excel Export

```python
with pd.ExcelWriter('data.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')
```

## ⚙️ Configuration Options

### Dashboard Configuration

```python
config = DashboardConfig(
    # Application settings
    app_name="My Dashboard",
    host="127.0.0.1",
    port=8050,
    debug=True,

    # Features
    enable_realtime=True,
    enable_export=True,
    enable_filtering=True,
    enable_dark_mode=True,
    enable_mobile=True,
    enable_accessibility=True,

    # Performance
    cache_timeout=300,
    data_refresh_interval=5000,
    max_data_points=10000,

    # Redis settings
    redis_host="localhost",
    redis_port=6379
)
```

### API Configuration

```python
api_config = APIConfig(
    host="0.0.0.0",
    port=5000,
    secret_key="your-secret-key",
    jwt_secret="your-jwt-secret",
    rate_limit_default="100 per hour",
    cors_origins=["http://localhost:3000"]
)
```

## 🔒 Security Best Practices

1. **Environment Variables**: Store sensitive configuration in `.env` files
2. **HTTPS**: Use SSL certificates in production
3. **Authentication**: Implement proper user authentication
4. **Rate Limiting**: Configure appropriate rate limits
5. **CORS**: Restrict origins in production
6. **Input Validation**: Validate all user inputs
7. **SQL Injection**: Use parameterized queries
8. **XSS Protection**: Sanitize user-generated content

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8050 5000

CMD ["python", "example_dashboard.py"]
```

### Production Checklist

- [ ] Set `debug=False` in configuration
- [ ] Configure proper secret keys
- [ ] Set up SSL/HTTPS
- [ ] Configure production database
- [ ] Set up monitoring and logging
- [ ] Configure backup strategy
- [ ] Implement rate limiting
- [ ] Set up CDN for static assets
- [ ] Configure auto-scaling
- [ ] Set up health checks

## 📊 Performance Optimization

1. **Data Caching**: Use Redis for frequently accessed data
2. **Lazy Loading**: Load components on demand
3. **Data Sampling**: Limit data points for large datasets
4. **Pagination**: Implement server-side pagination
5. **Compression**: Enable gzip compression
6. **CDN**: Use CDN for static assets
7. **Database Indexing**: Optimize database queries
8. **Connection Pooling**: Use connection pools for databases

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact the development team
- Check the documentation

## 🎯 Roadmap

- [ ] Add more chart types (treemaps, violin plots, etc.)
- [ ] Implement dashboard templates
- [ ] Add data connectors (SQL, NoSQL, APIs)
- [ ] Create dashboard builder UI
- [ ] Add machine learning integration
- [ ] Implement advanced caching strategies
- [ ] Add multi-language support
- [ ] Create mobile app wrapper
- [ ] Add voice control features
- [ ] Implement collaborative features

## 📚 Resources

- [Dash Documentation](https://dash.plotly.com/)
- [Plotly Documentation](https://plotly.com/python/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [GraphQL Documentation](https://graphql.org/)
- [WebSocket Documentation](https://socket.io/)

---

Built with ❤️ by the Data Science Portfolio Team