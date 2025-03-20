# Simple Supply and Demand Visualization

A lightweight, interactive tool for generating supply-demand curve visualizations with Plotly.

## Features

- Create interactive supply and demand plots
- Calculate and visualize equilibrium points
- Show step-by-step solution process
- Generate complete explanations with text and audio
- Handle supply and demand curve shifts
- Export to HTML (interactive) or PNG (static)

## Getting Started

### Prerequisites

- Python 3.7+
- Plotly
- NumPy
- gTTS (Google Text-to-Speech)

```bash
pip install plotly numpy gtts
```

### Usage

Run the example script:

```bash
python run_example.py --scene equilibrium --open
```

This will:
1. Generate a supply and demand visualization
2. Calculate the equilibrium point
3. Create HTML, PNG, text, and audio explanations
4. Open the interactive HTML visualization

### Available Scenes

- `equilibrium`: Basic market equilibrium
- `supply_shift`: Demonstration of a supply shift
- `demand_shift`: Demonstration of a demand shift

## Example Code

```python
from supply_demand import SupplyDemandPlot

# Create a supply and demand plot
config = {
    "supply_config": {
        "slope": 0.5,
        "intercept": 2,
        "color": "blue"
    },
    "demand_config": {
        "slope": -0.5,
        "intercept": 8,
        "color": "red"
    }
}

sd_plot = SupplyDemandPlot(config)

# Calculate equilibrium
eq_x, eq_y = sd_plot.solve_equilibrium()
print(f"Equilibrium: Q = {eq_x:.2f}, P = {eq_y:.2f}")

# Save as interactive HTML
sd_plot.save_html("my_plot.html")
```

## Customization

The visualization can be extensively customized:

```python
config = {
    "x_range": [0, 15],  # Range for x-axis
    "y_range": [0, 15],  # Range for y-axis
    "supply_config": {
        "slope": 0.8,
        "intercept": 3,
        "color": "rgb(0, 100, 200)",  # Any valid CSS color
        "name": "Supply"
    },
    "demand_config": {
        "slope": -0.7,
        "intercept": 10,
        "color": "rgb(200, 50, 50)",
        "name": "Demand"
    },
    "layout_config": {
        "title": "Custom Market Analysis",
        "width": 1000,       # Width in pixels
        "height": 600,       # Height in pixels
        "show_equilibrium": True,
        "show_steps": True,
        "equilibrium_color": "green"
    }
}
```

## Output Files

The `create_explanation()` method generates:

- Interactive HTML file with the visualization
- Static PNG image
- Text file with the step-by-step explanation
- MP3 audio file with spoken explanation
- JSON file with calculation steps

## Comparison with Manim Approach

This Plotly-based implementation has several advantages:

1. **Interactive**: Users can zoom, pan, and hover for more information
2. **Faster**: No video rendering, immediate visualization
3. **Simpler**: Less code, easier to understand and modify
4. **Web-ready**: Works everywhere with no plugins
5. **Better visibility**: Clear axis, grid lines, and colors

## Further Development

Future enhancements could include:
- Support for multiple supply/demand curves
- More economic scenarios (price controls, taxes, etc.)
- Export to additional formats
- Integration with web frameworks 