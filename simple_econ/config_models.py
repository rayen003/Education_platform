"""
Pydantic models for structured configuration of economic visualizations.

These models enforce validation rules to ensure the generated configurations
will produce valid and accurate visualizations.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal, Dict, Union, Any

class SupplyDemandConfig(BaseModel):
    """Configuration for supply and demand curve visualization"""
    supply_slope: float = Field(..., description="Slope of supply curve (must be positive)")
    supply_intercept: float = Field(..., description="Y-intercept of supply curve")
    demand_slope: float = Field(..., description="Slope of demand curve (must be negative)")
    demand_intercept: float = Field(..., description="Y-intercept of demand curve")
    x_range: List[float] = Field([0, 10], description="Range for x-axis [min, max]")
    y_range: List[float] = Field([0, 10], description="Range for y-axis [min, max]")
    show_equilibrium: bool = Field(True, description="Whether to show equilibrium point")
    equilibrium_color: str = Field("green", description="Color for the equilibrium point and lines")
    supply_color: str = Field("blue", description="Color for the supply curve")
    demand_color: str = Field("red", description="Color for the demand curve")
    
    @validator('supply_slope')
    def supply_slope_positive(cls, v):
        if v <= 0:
            raise ValueError('Supply slope must be positive')
        return v
    
    @validator('demand_slope')
    def demand_slope_negative(cls, v):
        if v >= 0:
            raise ValueError('Demand slope must be negative')
        return v
    
    @validator('x_range', 'y_range')
    def validate_range(cls, v):
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError('Range must be [min, max] with min < max')
        return v
    
    def calculate_equilibrium(self):
        """Calculate the equilibrium point where supply equals demand"""
        # Solve: supply_slope * x + supply_intercept = demand_slope * x + demand_intercept
        # (supply_slope - demand_slope) * x = demand_intercept - supply_intercept
        # x = (demand_intercept - supply_intercept) / (supply_slope - demand_slope)
        
        eq_x = (self.demand_intercept - self.supply_intercept) / (self.supply_slope - self.demand_slope)
        eq_y = self.supply_slope * eq_x + self.supply_intercept
        
        return {"quantity": eq_x, "price": eq_y}

class SupplyShiftConfig(SupplyDemandConfig):
    """Configuration for supply shift visualization"""
    new_supply_slope: Optional[float] = Field(None, description="New slope after shift (if None, use same slope)")
    new_supply_intercept: float = Field(..., description="New intercept after shift")
    shift_style: Literal["parallel", "pivot", "both"] = Field("parallel", description="Type of shift to visualize")
    frames: int = Field(30, description="Number of animation frames")
    
    @validator('new_supply_slope')
    def validate_new_slope(cls, v, values):
        if v is not None and v <= 0:
            raise ValueError('New supply slope must be positive')
        return v

class DemandShiftConfig(SupplyDemandConfig):
    """Configuration for demand shift visualization"""
    new_demand_slope: Optional[float] = Field(None, description="New slope after shift (if None, use same slope)")
    new_demand_intercept: float = Field(..., description="New intercept after shift")
    shift_style: Literal["parallel", "pivot", "both"] = Field("parallel", description="Type of shift to visualize")
    frames: int = Field(30, description="Number of animation frames")
    
    @validator('new_demand_slope')
    def validate_new_slope(cls, v, values):
        if v is not None and v >= 0:
            raise ValueError('New demand slope must be negative')
        return v

class TimeValueConfig(BaseModel):
    """Configuration for time value of money visualization"""
    cash_flows: List[float] = Field(..., description="List of cash flows")
    time_periods: List[int] = Field(..., description="Time periods for cash flows")
    interest_rate: float = Field(..., description="Interest rate as decimal (e.g., 0.05 for 5%)")
    
    @validator('interest_rate')
    def validate_rate(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError('Interest rate should be between 0 and 1')
        return v
    
    @validator('time_periods', 'cash_flows')
    def validate_lists(cls, v, values, **kwargs):
        field_name = kwargs.get('field').name
        other_field = 'cash_flows' if field_name == 'time_periods' else 'time_periods'
        
        if other_field in values and len(values[other_field]) != len(v):
            raise ValueError(f'Length of {field_name} must match length of {other_field}')
        
        return v

class PerpetuityConfig(BaseModel):
    """Configuration for perpetuity visualization"""
    payment: float = Field(..., description="Regular payment amount")
    interest_rate: float = Field(..., description="Interest rate as decimal (e.g., 0.05 for 5%)")
    periods_to_show: int = Field(10, description="Number of periods to show in visualization")
    
    @validator('interest_rate')
    def validate_rate(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError('Interest rate should be between 0 and 1')
        return v
    
    @validator('payment')
    def validate_payment(cls, v):
        if v <= 0:
            raise ValueError('Payment must be positive')
        return v
    
    def calculate_present_value(self):
        """Calculate the present value of the perpetuity"""
        return self.payment / self.interest_rate

class AnimationConfig(BaseModel):
    """Main configuration for economic animation"""
    visualization_type: Literal["supply_demand", "supply_shift", "demand_shift", 
                               "time_value", "perpetuity"] = Field(...)
    title: str = Field(..., description="Title of the animation")
    subtitle: Optional[str] = Field(None, description="Subtitle with additional information")
    width: int = Field(950, description="Width of the visualization in pixels")
    height: int = Field(600, description="Height of the visualization in pixels")
    supply_demand_config: Optional[SupplyDemandConfig] = None
    supply_shift_config: Optional[SupplyShiftConfig] = None
    demand_shift_config: Optional[DemandShiftConfig] = None
    time_value_config: Optional[TimeValueConfig] = None
    perpetuity_config: Optional[PerpetuityConfig] = None
    
    @validator('supply_demand_config')
    def validate_supply_demand(cls, v, values):
        if values.get('visualization_type') == 'supply_demand' and v is None:
            raise ValueError('supply_demand_config is required for supply/demand visualizations')
        return v
    
    @validator('supply_shift_config')
    def validate_supply_shift(cls, v, values):
        if values.get('visualization_type') == 'supply_shift' and v is None:
            raise ValueError('supply_shift_config is required for supply shift visualizations')
        return v
    
    @validator('demand_shift_config')
    def validate_demand_shift(cls, v, values):
        if values.get('visualization_type') == 'demand_shift' and v is None:
            raise ValueError('demand_shift_config is required for demand shift visualizations')
        return v
    
    @validator('time_value_config')
    def validate_time_value(cls, v, values):
        if values.get('visualization_type') == 'time_value' and v is None:
            raise ValueError('time_value_config is required for time value visualizations')
        return v
    
    @validator('perpetuity_config')
    def validate_perpetuity(cls, v, values):
        if values.get('visualization_type') == 'perpetuity' and v is None:
            raise ValueError('perpetuity_config is required for perpetuity visualizations')
        return v 