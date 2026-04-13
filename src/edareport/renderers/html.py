"""HTML renderer — Jinja2 template + Plotly CDN embed."""

from __future__ import annotations

from jinja2 import Environment, BaseLoader

from edareport._types import ReportData, ColumnProfile

_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ data.title }}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" crossorigin="anonymous"></script>
<style>
  :root {
    --bg: #ffffff; --bg2: #f8f8f7; --border: #e0dfd8;
    --text: #1a1a18; --text2: #5f5e5a; --accent: #185FA5;
    --warn: #b85400; --radius: 8px;
  }
  [data-theme="dark"] {
    --bg: #1a1a18; --bg2: #242422; --border: #3a3a38;
    --text: #e8e6de; --text2: #9c9a92; --accent: #85B7EB;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font: 14px/1.6 system-ui, sans-serif; }
  .container { max-width: 1100px; margin: 0 auto; padding: 24px 16px; }
  header { margin-bottom: 28px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }
  header h1 { font-size: 22px; font-weight: 500; }
  header .meta { color: var(--text2); font-size: 13px; margin-top: 4px; }
  .section { margin-bottom: 32px; }
  .section-title { font-size: 16px; font-weight: 500; margin-bottom: 14px;
    border-left: 3px solid var(--accent); padding-left: 10px; }
  /* Overview cards */
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
  .card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 14px 16px; }
  .card .label { font-size: 11px; color: var(--text2); text-transform: uppercase; letter-spacing: .04em; }
  .card .value { font-size: 20px; font-weight: 500; margin-top: 4px; }
  /* Warnings */
  .warnings { background: #fff8f0; border: 1px solid #f5c77a; border-radius: var(--radius);
    padding: 12px 16px; }
  [data-theme="dark"] .warnings { background: #2a1f0e; border-color: #7a4d10; }
  .warnings ul { padding-left: 18px; font-size: 13px; color: var(--warn); }
  [data-theme="dark"] .warnings ul { color: #EF9F27; }
  /* Column table */
  .col-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .col-table th { background: var(--bg2); border-bottom: 1px solid var(--border);
    padding: 8px 10px; text-align: left; font-weight: 500; }
  .col-table td { padding: 7px 10px; border-bottom: 1px solid var(--border); }
  .col-table tr:last-child td { border-bottom: none; }
  .badge { display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 99px; }
  .badge-numeric   { background: #E6F1FB; color: #0C447C; }
  .badge-category  { background: #E1F5EE; color: #085041; }
  .badge-datetime  { background: #FAEEDA; color: #633806; }
  .badge-text      { background: #F1EFE8; color: #444441; }
  .badge-other     { background: #F1EFE8; color: #444441; }
  [data-theme="dark"] .badge-numeric   { background: #042C53; color: #85B7EB; }
  [data-theme="dark"] .badge-category  { background: #04342C; color: #5DCAA5; }
  [data-theme="dark"] .badge-datetime  { background: #412402; color: #FAC775; }
  [data-theme="dark"] .badge-text      { background: #2C2C2A; color: #B4B2A9; }
  /* Plots grid */
  .plots-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }
  .plot-card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 8px; overflow: hidden; }
  /* Top correlations */
  .corr-list { list-style: none; font-size: 13px; }
  .corr-list li { display: flex; justify-content: space-between; padding: 5px 0;
    border-bottom: 1px solid var(--border); }
  .corr-list li:last-child { border-bottom: none; }
  .corr-val { font-weight: 500; font-variant-numeric: tabular-nums; }
  .corr-pos { color: #D85A30; }
  .corr-neg { color: #185FA5; }
  /* Toggle theme */
  .theme-toggle { float: right; font-size: 12px; cursor: pointer; color: var(--text2);
    background: var(--bg2); border: 1px solid var(--border); border-radius: 4px;
    padding: 4px 10px; }
  footer { margin-top: 40px; padding-top: 12px; border-top: 1px solid var(--border);
    font-size: 11px; color: var(--text2); text-align: center; }
</style>
</head>
<body data-theme="{{ theme }}">
<div class="container">
  <header>
    <button class="theme-toggle" onclick="toggleTheme()">toggle theme</button>
    <h1>{{ data.title }}</h1>
    <div class="meta">{{ data.n_rows | format_number }} rows &nbsp;·&nbsp;
      {{ data.n_cols }} columns &nbsp;·&nbsp;
      {{ "%.2f"|format(data.memory_mb) }} MB in memory</div>
  </header>

  {# Overview cards #}
  <div class="section">
    <div class="section-title">Overview</div>
    <div class="cards">
      <div class="card"><div class="label">Rows</div><div class="value">{{ data.n_rows | format_number }}</div></div>
      <div class="card"><div class="label">Columns</div><div class="value">{{ data.n_cols }}</div></div>
      <div class="card"><div class="label">Numeric</div><div class="value">{{ col_counts.numeric }}</div></div>
      <div class="card"><div class="label">Categorical</div><div class="value">{{ col_counts.categorical }}</div></div>
      <div class="card"><div class="label">Missing cells</div><div class="value">{{ total_missing | format_number }}</div></div>
      <div class="card"><div class="label">Memory</div><div class="value">{{ "%.1f"|format(data.memory_mb) }} MB</div></div>
    </div>
  </div>

  {# Warnings #}
  {% if data.warnings %}
  <div class="section">
    <div class="section-title">Warnings ({{ data.warnings | length }})</div>
    <div class="warnings">
      <ul>{% for w in data.warnings %}<li>{{ w }}</li>{% endfor %}</ul>
    </div>
  </div>
  {% endif %}

  {# Column summary table #}
  <div class="section">
    <div class="section-title">Column summary</div>
    <table class="col-table">
      <thead><tr>
        <th>Column</th><th>Type</th><th>Missing</th>
        <th>Unique</th><th>Mean / Top value</th>
      </tr></thead>
      <tbody>
      {% for cp in data.columns %}
      <tr>
        <td>{{ cp.name }}</td>
        <td><span class="badge badge-{{ cp.dtype }}">{{ cp.dtype }}</span></td>
        <td>{{ "%.1f"|format(cp.missing_pct * 100) }}%</td>
        <td>{{ cp.n_unique | format_number }}</td>
        <td>
          {% if cp.dtype == "numeric" and cp.mean is not none %}
            {{ "%.4g"|format(cp.mean) }}
          {% elif cp.dtype == "categorical" and cp.top_values %}
            {{ cp.top_values.keys() | list | first }}
          {% else %}—{% endif %}
        </td>
      </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>

  {# Univariate plots #}
  {% if uni_plots %}
  <div class="section">
    <div class="section-title">Distributions</div>
    <div class="plots-grid">
      {% for col_name, plot_json in uni_plots.items() %}
      <div class="plot-card">
        <div id="uni_{{ loop.index }}"></div>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  {# Correlation heatmap #}
  {% if "correlation_heatmap" in bi_plots %}
  <div class="section">
    <div class="section-title">Correlation matrix</div>
    <div class="plot-card" style="display:inline-block">
      <div id="bi_heatmap"></div>
    </div>
  </div>
  {% endif %}

  {# Top correlations #}
  {% if data.top_correlations %}
  <div class="section">
    <div class="section-title">Top correlations</div>
    <ul class="corr-list">
      {% for a, b, r in data.top_correlations %}
      <li>
        <span>{{ a }} &nbsp;↔&nbsp; {{ b }}</span>
        <span class="corr-val {% if r >= 0 %}corr-pos{% else %}corr-neg{% endif %}">
          {{ "%+.3f"|format(r) }}
        </span>
      </li>
      {% endfor %}
    </ul>
  </div>
  {% endif %}

  {# Scatter plots #}
  {% set scatter_items = bi_plots.items() | selectattr('0', 'startswith', 'scatter') | list %}
  {% if scatter_items %}
  <div class="section">
    <div class="section-title">Top correlated pairs</div>
    <div class="plots-grid">
      {% for key, _ in scatter_items %}
      <div class="plot-card"><div id="bi_{{ loop.index }}"></div></div>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  <footer>generated by <strong>edareport</strong> {{ version }}</footer>
</div>

<script>
var _theme = document.body.dataset.theme || "light";
function toggleTheme() {
  _theme = _theme === "light" ? "dark" : "light";
  document.body.dataset.theme = _theme;
}
var _plotBg = "rgba(0,0,0,0)";
function _patch(json) {
  var fig = JSON.parse(json);
  fig.layout = fig.layout || {};
  fig.layout.paper_bgcolor = _plotBg;
  fig.layout.plot_bgcolor  = _plotBg;
  return fig;
}

// Univariate
var _uniPlots = {{ uni_plot_json }};
Object.entries(_uniPlots).forEach(function([col, json], i) {
  var fig = _patch(json);
  Plotly.newPlot("uni_" + (i + 1), fig.data, fig.layout, {responsive:true, displayModeBar:false});
});

// Bivariate heatmap
{% if "correlation_heatmap" in bi_plots %}
(function() {
  var fig = _patch({{ bi_plots["correlation_heatmap"] | safe }});
  Plotly.newPlot("bi_heatmap", fig.data, fig.layout, {responsive:true, displayModeBar:false});
})();
{% endif %}

// Scatter plots
var _scatterPlots = {{ scatter_plot_json }};
Object.entries(_scatterPlots).forEach(function([key, json], i) {
  var fig = _patch(json);
  Plotly.newPlot("bi_" + (i + 1), fig.data, fig.layout, {responsive:true, displayModeBar:false});
});
</script>
</body>
</html>
"""


class HtmlRenderer:
    def __init__(self, theme: str = "light") -> None:
        self._theme = theme
        self._env = Environment(loader=BaseLoader(), autoescape=True)
        self._env.filters["format_number"] = lambda v: f"{int(v):,}"

    def render(
        self,
        data: ReportData,
        uni_plots: dict[str, str],
        bi_plots: dict[str, str],
    ) -> str:
        import json
        from edareport import __version__

        col_counts = {
            "numeric": sum(1 for c in data.columns if c.dtype == "numeric"),
            "categorical": sum(1 for c in data.columns if c.dtype == "categorical"),
            "datetime": sum(1 for c in data.columns if c.dtype == "datetime"),
            "text": sum(1 for c in data.columns if c.dtype == "text"),
        }
        total_missing = sum(c.n_missing for c in data.columns)

        # Pisahkan scatter dari heatmap agar Jinja tidak perlu selectattr 'startswith'
        scatter_plots = {k: v for k, v in bi_plots.items() if k.startswith("scatter")}

        template = self._env.from_string(_TEMPLATE)
        return template.render(
            data=data,
            theme=self._theme,
            version=__version__,
            col_counts=col_counts,
            total_missing=total_missing,
            uni_plots=uni_plots,
            bi_plots=bi_plots,
            uni_plot_json=json.dumps(uni_plots),
            scatter_plot_json=json.dumps(scatter_plots),
        )