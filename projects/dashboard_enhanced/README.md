# Dashboard Components Reference

This folder contains a reference implementation for data dashboard patterns used
across the portfolio: Dash layouts, interactive Plotly components, REST/GraphQL
API sketches, export helpers, and focused dashboard tests.

The code is useful as a review target for application structure and interface
design. It is not presented as a deployed product or production service.

## What To Review

- `dashboard_framework.py`: dashboard configuration, layout assembly, filtering,
  export helpers, and optional real-time hooks.
- `visualization_components.py`: reusable Plotly chart builders for common
  analytics views.
- `api_infrastructure.py` and `graphql_api.py`: API patterns for dashboard data
  access, authentication, and query surfaces.
- `app.py` and `example_dashboard.py`: runnable examples that connect the pieces.
- `test_*.py` and `testing_suite.py`: dashboard-specific validation examples.

## Local Use

Install the project-specific requirements when working inside this folder:

```bash
pip install -r projects/dashboard_enhanced/requirements.txt
```

Run the example dashboard from the repository root:

```bash
python projects/dashboard_enhanced/example_dashboard.py
```

Optional features such as Redis-backed real-time updates, browser automation, and
export formats require the matching services or system packages to be installed
locally.

## Portfolio Role

This project demonstrates how analytical results can be wrapped in a usable
interface:

- clear configuration boundaries;
- reusable visualization builders;
- API surfaces for downstream consumers;
- export paths for stakeholder reporting;
- tests that exercise dashboard behavior separately from notebooks.

For the shortest reviewer path, start with `dashboard_framework.py`,
`visualization_components.py`, and `example_dashboard.py`.
