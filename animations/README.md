# Maenigna - Reusable Economics Animation Components

Maenigna is a library of reusable economic and financial animation components for [Manim](https://www.manim.community/). The goal is to provide standardized, high-quality components that can be easily customized and reused in economics and finance educational content.

## Current Components

### SupplyDemandCurve

A reusable supply and demand curve component that includes:
- Customizable axes, labels, and title
- Supply and demand curves with customizable slopes and intercepts
- Automatic equilibrium calculation and visualization
- Methods to shift supply or demand curves and update equilibrium

## Usage Examples

### Basic Market Equilibrium

```python
from manim import Scene
from lib.supply_demand import SupplyDemandCurve

class MarketEquilibriumDemo(Scene):
    def construct(self):
        # Create custom configuration for our specific problem
        sd_config = {
            "supply_config": {
                "slope": 0.5,
                "intercept": 2,
                "label_text": "Supply: P = 2 + 0.5Q"
            },
            "demand_config": {
                "slope": -0.5,
                "intercept": 10,
                "label_text": "Demand: P = 10 - 0.5Q"
            },
            "labels_config": {
                "title": "Market Equilibrium" 
            }
        }
        
        # Create the supply-demand component
        sd_curve = SupplyDemandCurve(self, sd_config)
        
        # Introduce the supply-demand diagram
        sd_curve.introduce_curves()
```

### Supply Shift Example

```python
from manim import Scene
from lib.supply_demand import SupplyDemandCurve

class SupplyShiftDemo(Scene):
    def construct(self):
        # Create custom configuration
        sd_config = {
            "supply_config": {
                "slope": 0.4,
                "intercept": 3,
                "label_text": "Supply: P = 3 + 0.4Q"
            },
            "demand_config": {
                "slope": -0.6,
                "intercept": 9,
                "label_text": "Demand: P = 9 - 0.6Q"
            }
        }
        
        # Create the component
        sd_curve = SupplyDemandCurve(self, sd_config)
        
        # Introduce the diagram
        sd_curve.introduce_curves()
        
        # Shift the supply curve
        shift_anim = sd_curve.shift_supply(new_intercept=5)  # Same slope, new intercept
        self.play(shift_anim)
        
        # Update the equilibrium point
        self.play(*sd_curve.get_updated_equilibrium())
```

## Running the Examples

To run the Market Equilibrium example:

```
conda activate manim-env
manim -pql animations/market_equilibrium_demo.py MarketEquilibriumDemo
```

To run the Supply Shift example:

```
conda activate manim-env
manim -pql animations/supply_shift_demo.py SupplyShiftDemo
```

## Component Customization

Each component accepts a configuration dictionary that allows for customizing various aspects of the animation. For example, the `SupplyDemandCurve` component accepts:

```python
{
    "x_range": [0, 10, 1],  # Range for x-axis [min, max, step]
    "y_range": [0, 10, 1],  # Range for y-axis [min, max, step]
    "axes_config": {
        "color": BLUE,
        "x_length": 6,
        "y_length": 6,
        "axis_config": {"include_tip": True},
    },
    "supply_config": {
        "color": GREEN,
        "slope": 0.5,
        "intercept": 2,
        "label_text": "Supply",
        "stroke_width": 3,
    },
    "demand_config": {
        "color": RED,
        "slope": -0.5,
        "intercept": 8,
        "label_text": "Demand",
        "stroke_width": 3,
    },
    "labels_config": {
        "x_label": "Quantity",
        "y_label": "Price",
        "title": None,
        "show_equilibrium": True,
        "equilibrium_color": YELLOW,
    }
}
```

## Benefits of the Component Library

1. **Standardization**: Consistent look and feel across all animations
2. **Reusability**: Create complex animations quickly by reusing components
3. **Efficiency**: Less computing power needed as components are optimized
4. **Flexibility**: Components can be customized to fit different scenarios
5. **Modularity**: Components can be combined to create more complex animations

## Future Additions

Potential components to add to the library:
- Time Value of Money visualizations
- Financial statement relationships
- Game theory matrices
- Indifference curves and budget constraints
- Production functions
- Macroeconomic models 