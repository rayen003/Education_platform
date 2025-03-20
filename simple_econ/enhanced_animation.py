#!/usr/bin/env python
"""
Enhanced Animation Generator

This script creates aesthetically improved animations with properly synchronized
audio narration for economic concepts.
"""

import os
import sys
import json
import shutil
import time
from gtts import gTTS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class EnhancedEconomicsAnimation:
    """
    Creates enhanced economic animations with synchronized audio narration.
    
    This class focuses on aesthetically pleasing visualizations and proper
    integration of audio narration with the visual elements.
    """
    
    def __init__(self, output_dir="./output/enhanced"):
        """Initialize the animation generator"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Default color scheme - more modern and visually appealing
        self.colors = {
            "supply": "#3498db",       # Soft blue
            "supply_new": "#2980b9",   # Darker blue
            "demand": "#e74c3c",       # Soft red
            "demand_new": "#c0392b",   # Darker red
            "equilibrium": "#2ecc71",  # Green
            "equilibrium_new": "#27ae60", # Darker green
            "grid": "#ecf0f1",         # Light gray
            "text": "#2c3e50",         # Dark gray/blue
            "background": "#ffffff",   # White
            "accent": "#f39c12"        # Orange for highlights
        }
        
        # Improved font settings
        self.fonts = {
            "family": "Arial, sans-serif",
            "title_size": 22,
            "axis_title_size": 16,
            "tick_size": 14,
            "annotation_size": 14
        }
    
    def create_supply_demand_animation(self, 
                                      supply_slope=0.5, 
                                      supply_intercept=2,
                                      demand_slope=-0.5, 
                                      demand_intercept=8,
                                      new_supply_intercept=4,
                                      frames=30,
                                      title="Impact of Production Cost Increase",
                                      subtitle="Interactive Analysis with Audio Narration"):
        """
        Create an enhanced supply shift animation with custom narration.
        
        Args:
            supply_slope: Initial supply curve slope
            supply_intercept: Initial supply curve intercept
            demand_slope: Demand curve slope
            demand_intercept: Demand curve intercept
            new_supply_intercept: New supply curve intercept after shift
            frames: Number of animation frames
            title: Main title for the visualization
            subtitle: Subtitle with additional information
            
        Returns:
            Dict with paths to generated files
        """
        print(f"Creating enhanced supply shift animation...")
        
        # Create output directory
        anim_dir = os.path.join(self.output_dir, "supply_shift")
        os.makedirs(anim_dir, exist_ok=True)
        
        # Calculate equilibrium points
        # Original equilibrium: supply = demand
        # supply_slope * Q + supply_intercept = demand_slope * Q + demand_intercept
        # (supply_slope - demand_slope) * Q = demand_intercept - supply_intercept
        # Q = (demand_intercept - supply_intercept) / (supply_slope - demand_slope)
        orig_eq_x = (demand_intercept - supply_intercept) / (supply_slope - demand_slope)
        orig_eq_y = supply_slope * orig_eq_x + supply_intercept
        
        # New equilibrium
        new_eq_x = (demand_intercept - new_supply_intercept) / (supply_slope - demand_slope)
        new_eq_y = supply_slope * new_eq_x + new_supply_intercept
        
        # Set up the figure with improved aesthetics
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        # Define x range based on equilibrium points
        x_max = max(orig_eq_x, new_eq_x) * 1.5
        x_range = [0, max(10, x_max)]
        
        # Calculate y values for original supply and demand curves
        x_vals = np.linspace(0, x_range[1], 100)
        supply_y = [supply_slope * x + supply_intercept for x in x_vals]
        demand_y = [demand_slope * x + demand_intercept for x in x_vals]
        
        # Add baseline curves
        fig.add_trace(
            go.Scatter(
                x=x_vals, 
                y=supply_y,
                mode='lines',
                name='Supply',
                line=dict(color=self.colors["supply"], width=3),
                hoverinfo='skip'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_vals, 
                y=demand_y,
                mode='lines',
                name='Demand',
                line=dict(color=self.colors["demand"], width=3),
                hoverinfo='skip'
            )
        )
        
        # Add original equilibrium point
        fig.add_trace(
            go.Scatter(
                x=[orig_eq_x],
                y=[orig_eq_y],
                mode='markers',
                name='Initial Equilibrium',
                marker=dict(
                    color=self.colors["equilibrium"],
                    size=12,
                    line=dict(color='white', width=2)
                ),
                hovertemplate='Quantity: %{x:.2f}<br>Price: %{y:.2f}<extra>Initial Equilibrium</extra>'
            )
        )
        
        # Add vertical and horizontal lines for equilibrium
        fig.add_trace(
            go.Scatter(
                x=[0, orig_eq_x, orig_eq_x],
                y=[orig_eq_y, orig_eq_y, 0],
                mode='lines',
                name='Equilibrium Lines',
                line=dict(color=self.colors["equilibrium"], width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            )
        )
        
        # Improved layout
        fig.update_layout(
            title={
                'text': f'<b>{title}</b><br><span style="font-size: 16px; color: gray;">{subtitle}</span>',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'family': self.fonts["family"],
                    'size': self.fonts["title_size"],
                    'color': self.colors["text"]
                }
            },
            xaxis=dict(
                title='Quantity',
                linecolor=self.colors["text"],
                showgrid=True,
                gridcolor=self.colors["grid"],
                range=x_range,
                zeroline=True,
                zerolinecolor=self.colors["text"],
                zerolinewidth=2,
                tickfont={'size': self.fonts["tick_size"]},
                titlefont={'size': self.fonts["axis_title_size"]}
            ),
            yaxis=dict(
                title='Price',
                linecolor=self.colors["text"],
                showgrid=True,
                gridcolor=self.colors["grid"],
                range=[0, max(supply_y + demand_y) * 1.1],
                zeroline=True,
                zerolinecolor=self.colors["text"],
                zerolinewidth=2,
                tickfont={'size': self.fonts["tick_size"]},
                titlefont={'size': self.fonts["axis_title_size"]}
            ),
            plot_bgcolor=self.colors["background"],
            paper_bgcolor=self.colors["background"],
            legend=dict(
                font=dict(family=self.fonts["family"], size=14),
                bgcolor=self.colors["background"],
                bordercolor=self.colors["text"],
                borderwidth=1
            ),
            width=950,
            height=600,
            margin=dict(l=80, r=40, t=100, b=80),
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family=self.fonts["family"]
            ),
            annotations=[
                dict(
                    x=0.5,
                    y=-0.15,
                    xref="paper",
                    yref="paper",
                    text="Play the animation to see how a supply shift affects market equilibrium",
                    showarrow=False,
                    font=dict(
                        family=self.fonts["family"],
                        size=14,
                        color=self.colors["text"]
                    )
                )
            ]
        )
        
        # Generate frames for the animation
        frames_list = []
        
        for i in range(frames + 1):
            # Calculate intermediate supply curve for this frame
            progress = i / frames
            current_intercept = supply_intercept + progress * (new_supply_intercept - supply_intercept)
            
            # Current supply curve
            current_supply_y = [supply_slope * x + current_intercept for x in x_vals]
            
            # Current equilibrium
            current_eq_x = (demand_intercept - current_intercept) / (supply_slope - demand_slope)
            current_eq_y = supply_slope * current_eq_x + current_intercept
            
            # Create the frame
            frame = go.Frame(
                data=[
                    # Keep demand curve constant
                    go.Scatter(x=x_vals, y=demand_y),
                    
                    # Update supply curve
                    go.Scatter(
                        x=x_vals, 
                        y=current_supply_y,
                        line=dict(
                            color=self.colors["supply"] if i < frames else self.colors["supply_new"],
                            width=3
                        )
                    ),
                    
                    # Update equilibrium point
                    go.Scatter(
                        x=[current_eq_x],
                        y=[current_eq_y],
                        marker=dict(
                            color=self.colors["equilibrium"] if i < frames else self.colors["equilibrium_new"],
                            size=12,
                            line=dict(color='white', width=2)
                        ),
                        hovertemplate='Quantity: %{x:.2f}<br>Price: %{y:.2f}<extra>Current Equilibrium</extra>'
                    ),
                    
                    # Update equilibrium lines
                    go.Scatter(
                        x=[0, current_eq_x, current_eq_x],
                        y=[current_eq_y, current_eq_y, 0],
                        line=dict(
                            color=self.colors["equilibrium"] if i < frames else self.colors["equilibrium_new"],
                            width=1,
                            dash='dash'
                        )
                    )
                ],
                name=f"frame{i}"
            )
            frames_list.append(frame)
        
        # Add frames to the figure
        fig.frames = frames_list
        
        # Add slider and buttons for animation control
        sliders = [
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=16, family=self.fonts["family"]),
                    prefix="Frame: ",
                    visible=True,
                    xanchor="right"
                ),
                transition=dict(duration=300, easing="cubic-in-out"),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [f"frame{i}"],
                            dict(
                                frame=dict(duration=300, redraw=True),
                                mode="immediate",
                                transition=dict(duration=300)
                            )
                        ],
                        label=str(i)
                    )
                    for i in range(0, frames + 1, 5)  # Show fewer labels for cleaner appearance
                ]
            )
        ]
        
        # Add play and pause buttons
        updatemenus = [
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[
                            None,
                            dict(
                                frame=dict(duration=300, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=300, easing="quadratic-in-out")
                            )
                        ],
                        label="▶ Play",
                        method="animate"
                    ),
                    dict(
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0)
                            )
                        ],
                        label="⏸ Pause",
                        method="animate"
                    )
                ],
                pad=dict(r=10, t=70),
                showactive=True,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top"
            )
        ]
        
        fig.update_layout(
            updatemenus=updatemenus,
            sliders=sliders
        )
        
        # Save the HTML file
        html_path = os.path.join(anim_dir, "supply_shift_animation.html")
        fig.write_html(html_path, include_plotlyjs="cdn")
        
        # Generate explanation text with more detailed economic insights
        text_explanation = f"""
Supply Shift: Impact of Production Cost Increase
================================================

Initial Market:
Supply: P = {supply_intercept:.2f} + {supply_slope:.2f}Q
Demand: P = {demand_intercept:.2f} - {abs(demand_slope):.2f}Q

Initial Equilibrium:
Quantity: {orig_eq_x:.2f} units
Price: ${orig_eq_y:.2f}

After Supply Shift:
Supply: P = {new_supply_intercept:.2f} + {supply_slope:.2f}Q  (Higher intercept due to increased costs)
Demand: P = {demand_intercept:.2f} - {abs(demand_slope):.2f}Q  (Unchanged)

New Equilibrium:
Quantity: {new_eq_x:.2f} units
Price: ${new_eq_y:.2f}

Economic Analysis:
1. The supply curve shifted upward (decreased supply) due to higher production costs.
2. This resulted in a higher equilibrium price (${orig_eq_y:.2f} → ${new_eq_y:.2f}).
3. The quantity traded decreased ({orig_eq_x:.2f} → {new_eq_x:.2f} units).
4. Consumers are worse off due to the higher price.
5. Some producers may be better off despite selling less, if the price increase more than offsets the cost increase.
6. Overall market efficiency has decreased due to the contraction in trade volume.
"""
        
        # Save the text explanation
        text_path = os.path.join(anim_dir, "supply_shift_explanation.txt")
        with open(text_path, "w") as f:
            f.write(text_explanation)
        
        # Generate audio narration
        narration = f"""
Welcome to this interactive visualization of a supply shift in a market.

We're looking at a market where the initial supply curve is P equals {supply_intercept:.2f} plus {supply_slope:.2f} Q,
and the demand curve is P equals {demand_intercept:.2f} minus {abs(demand_slope):.2f} Q.

The initial equilibrium occurs at a quantity of {orig_eq_x:.2f} units and a price of {orig_eq_y:.2f} dollars.

Now, let's see what happens when production costs increase, causing the supply curve to shift upward.
The new supply curve has a higher intercept of {new_supply_intercept:.2f}, while maintaining the same slope.

As the supply curve shifts upward, notice how the equilibrium point moves.
The new equilibrium quantity is {new_eq_x:.2f} units, which is lower than before.
The new equilibrium price is {new_eq_y:.2f} dollars, which is higher than before.

This illustrates a fundamental principle in economics: when supply decreases (shifts left),
prices tend to rise and quantities traded tend to fall.

This has real implications for market participants:
- Consumers face higher prices and consume less
- Producers sell fewer units but at higher prices
- The overall market has less trade and lower efficiency

This type of shift is common in markets facing increased production costs, resource constraints,
or new regulations that make production more expensive.
"""
        
        # Save the audio narration
        audio_path = os.path.join(anim_dir, "supply_shift_narration.mp3")
        try:
            tts = gTTS(text=narration, lang='en', slow=False)
            tts.save(audio_path)
            print(f"Audio narration saved to: {audio_path}")
        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            audio_path = None
        
        # Create the synchronized HTML file with enhanced styling
        html_template = self._create_synced_html(html_path, audio_path, text_explanation, anim_dir)
        
        return {
            "animated_html": html_path,
            "explanation_text": text_path,
            "narration_audio": audio_path,
            "interactive_html": html_template
        }
    
    def create_demand_shift_animation(self, 
                                     supply_slope=0.5, 
                                     supply_intercept=2,
                                     demand_slope=-0.5, 
                                     demand_intercept=8,
                                     new_demand_intercept=6,
                                     frames=30,
                                     title="Impact of Demand Decrease",
                                     subtitle="Interactive Analysis with Audio Narration"):
        """
        Create an enhanced demand shift animation with custom narration.
        
        Args:
            supply_slope: Supply curve slope
            supply_intercept: Supply curve intercept
            demand_slope: Initial demand curve slope
            demand_intercept: Initial demand curve intercept
            new_demand_intercept: New demand curve intercept after shift
            frames: Number of animation frames
            title: Main title for the visualization
            subtitle: Subtitle with additional information
            
        Returns:
            Dict with paths to generated files
        """
        print(f"Creating enhanced demand shift animation...")
        
        # Create output directory
        anim_dir = os.path.join(self.output_dir, "demand_shift")
        os.makedirs(anim_dir, exist_ok=True)
        
        # Calculate equilibrium points
        # Original equilibrium: supply = demand
        # supply_slope * Q + supply_intercept = demand_slope * Q + demand_intercept
        # (supply_slope - demand_slope) * Q = demand_intercept - supply_intercept
        # Q = (demand_intercept - supply_intercept) / (supply_slope - demand_slope)
        orig_eq_x = (demand_intercept - supply_intercept) / (supply_slope - demand_slope)
        orig_eq_y = supply_slope * orig_eq_x + supply_intercept
        
        # New equilibrium
        new_eq_x = (new_demand_intercept - supply_intercept) / (supply_slope - demand_slope)
        new_eq_y = supply_slope * new_eq_x + supply_intercept
        
        # Set up the figure with improved aesthetics
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        # Define x range based on equilibrium points
        x_max = max(orig_eq_x, new_eq_x) * 1.5
        x_range = [0, max(10, x_max)]
        
        # Calculate y values for original supply and demand curves
        x_vals = np.linspace(0, x_range[1], 100)
        supply_y = [supply_slope * x + supply_intercept for x in x_vals]
        demand_y = [demand_slope * x + demand_intercept for x in x_vals]
        
        # Add baseline curves
        fig.add_trace(
            go.Scatter(
                x=x_vals, 
                y=supply_y,
                mode='lines',
                name='Supply',
                line=dict(color=self.colors["supply"], width=3),
                hoverinfo='skip'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_vals, 
                y=demand_y,
                mode='lines',
                name='Demand',
                line=dict(color=self.colors["demand"], width=3),
                hoverinfo='skip'
            )
        )
        
        # Add original equilibrium point
        fig.add_trace(
            go.Scatter(
                x=[orig_eq_x],
                y=[orig_eq_y],
                mode='markers',
                name='Initial Equilibrium',
                marker=dict(
                    color=self.colors["equilibrium"],
                    size=12,
                    line=dict(color='white', width=2)
                ),
                hovertemplate='Quantity: %{x:.2f}<br>Price: %{y:.2f}<extra>Initial Equilibrium</extra>'
            )
        )
        
        # Add vertical and horizontal lines for equilibrium
        fig.add_trace(
            go.Scatter(
                x=[0, orig_eq_x, orig_eq_x],
                y=[orig_eq_y, orig_eq_y, 0],
                mode='lines',
                name='Equilibrium Lines',
                line=dict(color=self.colors["equilibrium"], width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            )
        )
        
        # Improved layout
        fig.update_layout(
            title={
                'text': f'<b>{title}</b><br><span style="font-size: 16px; color: gray;">{subtitle}</span>',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'family': self.fonts["family"],
                    'size': self.fonts["title_size"],
                    'color': self.colors["text"]
                }
            },
            xaxis=dict(
                title='Quantity',
                linecolor=self.colors["text"],
                showgrid=True,
                gridcolor=self.colors["grid"],
                range=x_range,
                zeroline=True,
                zerolinecolor=self.colors["text"],
                zerolinewidth=2,
                tickfont={'size': self.fonts["tick_size"]},
                titlefont={'size': self.fonts["axis_title_size"]}
            ),
            yaxis=dict(
                title='Price',
                linecolor=self.colors["text"],
                showgrid=True,
                gridcolor=self.colors["grid"],
                range=[0, max(supply_y + demand_y) * 1.1],
                zeroline=True,
                zerolinecolor=self.colors["text"],
                zerolinewidth=2,
                tickfont={'size': self.fonts["tick_size"]},
                titlefont={'size': self.fonts["axis_title_size"]}
            ),
            plot_bgcolor=self.colors["background"],
            paper_bgcolor=self.colors["background"],
            legend=dict(
                font=dict(family=self.fonts["family"], size=14),
                bgcolor=self.colors["background"],
                bordercolor=self.colors["text"],
                borderwidth=1
            ),
            width=950,
            height=600,
            margin=dict(l=80, r=40, t=100, b=80),
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family=self.fonts["family"]
            ),
            annotations=[
                dict(
                    x=0.5,
                    y=-0.15,
                    xref="paper",
                    yref="paper",
                    text="Play the animation to see how a demand shift affects market equilibrium",
                    showarrow=False,
                    font=dict(
                        family=self.fonts["family"],
                        size=14,
                        color=self.colors["text"]
                    )
                )
            ]
        )
        
        # Generate frames for the animation
        frames_list = []
        
        for i in range(frames + 1):
            # Calculate intermediate demand curve for this frame
            progress = i / frames
            current_intercept = demand_intercept + progress * (new_demand_intercept - demand_intercept)
            
            # Current demand curve
            current_demand_y = [demand_slope * x + current_intercept for x in x_vals]
            
            # Current equilibrium
            current_eq_x = (current_intercept - supply_intercept) / (supply_slope - demand_slope)
            current_eq_y = supply_slope * current_eq_x + supply_intercept
            
            # Create the frame
            frame = go.Frame(
                data=[
                    # Keep supply curve constant
                    go.Scatter(x=x_vals, y=supply_y),
                    
                    # Update demand curve
                    go.Scatter(
                        x=x_vals, 
                        y=current_demand_y,
                        line=dict(
                            color=self.colors["demand"] if i < frames else self.colors["demand_new"],
                            width=3
                        )
                    ),
                    
                    # Update equilibrium point
                    go.Scatter(
                        x=[current_eq_x],
                        y=[current_eq_y],
                        marker=dict(
                            color=self.colors["equilibrium"] if i < frames else self.colors["equilibrium_new"],
                            size=12,
                            line=dict(color='white', width=2)
                        ),
                        hovertemplate='Quantity: %{x:.2f}<br>Price: %{y:.2f}<extra>Current Equilibrium</extra>'
                    ),
                    
                    # Update equilibrium lines
                    go.Scatter(
                        x=[0, current_eq_x, current_eq_x],
                        y=[current_eq_y, current_eq_y, 0],
                        line=dict(
                            color=self.colors["equilibrium"] if i < frames else self.colors["equilibrium_new"],
                            width=1,
                            dash='dash'
                        )
                    )
                ],
                name=f"frame{i}"
            )
            frames_list.append(frame)
        
        # Add frames to the figure
        fig.frames = frames_list
        
        # Add slider and buttons for animation control
        sliders = [
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=16, family=self.fonts["family"]),
                    prefix="Frame: ",
                    visible=True,
                    xanchor="right"
                ),
                transition=dict(duration=300, easing="cubic-in-out"),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [f"frame{i}"],
                            dict(
                                frame=dict(duration=300, redraw=True),
                                mode="immediate",
                                transition=dict(duration=300)
                            )
                        ],
                        label=str(i)
                    )
                    for i in range(0, frames + 1, 5)  # Show fewer labels for cleaner appearance
                ]
            )
        ]
        
        # Add play and pause buttons
        updatemenus = [
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[
                            None,
                            dict(
                                frame=dict(duration=300, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=300, easing="quadratic-in-out")
                            )
                        ],
                        label="▶ Play",
                        method="animate"
                    ),
                    dict(
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0)
                            )
                        ],
                        label="⏸ Pause",
                        method="animate"
                    )
                ],
                pad=dict(r=10, t=70),
                showactive=True,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top"
            )
        ]
        
        fig.update_layout(
            updatemenus=updatemenus,
            sliders=sliders
        )
        
        # Save the HTML file
        html_path = os.path.join(anim_dir, "demand_shift_animation.html")
        fig.write_html(html_path, include_plotlyjs="cdn")
        
        # Generate explanation text with more detailed economic insights
        text_explanation = f"""
Demand Shift: Impact of Changing Consumer Preferences
====================================================

Initial Market:
Supply: P = {supply_intercept:.2f} + {supply_slope:.2f}Q
Demand: P = {demand_intercept:.2f} - {abs(demand_slope):.2f}Q

Initial Equilibrium:
Quantity: {orig_eq_x:.2f} units
Price: ${orig_eq_y:.2f}

After Demand Shift:
Supply: P = {supply_intercept:.2f} + {supply_slope:.2f}Q  (Unchanged)
Demand: P = {new_demand_intercept:.2f} - {abs(demand_slope):.2f}Q  (Lower intercept due to preference change)

New Equilibrium:
Quantity: {new_eq_x:.2f} units
Price: ${new_eq_y:.2f}

Economic Analysis:
1. The demand curve shifted leftward (decreased demand) due to changing consumer preferences.
2. This resulted in a lower equilibrium price (${orig_eq_y:.2f} → ${new_eq_y:.2f}).
3. The quantity traded decreased ({orig_eq_x:.2f} → {new_eq_x:.2f} units).
4. Producers are worse off due to lower prices and lower sales volumes.
5. Consumers who still purchase the product benefit from lower prices.
6. Overall market efficiency has decreased due to the contraction in trade volume.
"""
        
        # Save the text explanation
        text_path = os.path.join(anim_dir, "demand_shift_explanation.txt")
        with open(text_path, "w") as f:
            f.write(text_explanation)
        
        # Generate audio narration
        narration = f"""
Welcome to this interactive visualization of a demand shift in a market.

We're looking at a market where the initial supply curve is P equals {supply_intercept:.2f} plus {supply_slope:.2f} Q,
and the demand curve is P equals {demand_intercept:.2f} minus {abs(demand_slope):.2f} Q.

The initial equilibrium occurs at a quantity of {orig_eq_x:.2f} units and a price of {orig_eq_y:.2f} dollars.

Now, let's see what happens when consumer preferences change, causing the demand curve to shift leftward.
The new demand curve has a lower intercept of {new_demand_intercept:.2f}, while maintaining the same slope.

As the demand curve shifts leftward, notice how the equilibrium point moves.
The new equilibrium quantity is {new_eq_x:.2f} units, which is lower than before.
The new equilibrium price is {new_eq_y:.2f} dollars, which is also lower than before.

This illustrates a fundamental principle in economics: when demand decreases (shifts left),
prices tend to fall and quantities traded tend to fall as well.

This has real implications for market participants:
- Producers face lower prices and sell fewer units
- Consumers who still buy the product benefit from lower prices
- The overall market has less trade volume and reduced total value

This type of shift is common in markets facing changes in consumer preferences, income reductions,
or when substitute products become more attractive to consumers.
"""
        
        # Save the audio narration
        audio_path = os.path.join(anim_dir, "demand_shift_narration.mp3")
        try:
            tts = gTTS(text=narration, lang='en', slow=False)
            tts.save(audio_path)
            print(f"Audio narration saved to: {audio_path}")
        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            audio_path = None
        
        # Create the synchronized HTML file with enhanced styling
        html_template = self._create_synced_html(html_path, audio_path, text_explanation, anim_dir)
        
        return {
            "animated_html": html_path,
            "explanation_text": text_path,
            "narration_audio": audio_path,
            "interactive_html": html_template
        }

    def _create_synced_html(self, html_path, audio_path, explanation_text, output_dir):
        """
        Create an enhanced HTML page with synchronized animation and audio.
        
        Args:
            html_path: Path to the Plotly HTML file
            audio_path: Path to the audio narration file
            explanation_text: Text explanation for the visualization
            output_dir: Directory to save the output file
            
        Returns:
            str: Path to the synced HTML file
        """
        # Format the explanation as HTML
        explanation_html = ""
        for line in explanation_text.strip().split('\n'):
            if line.endswith(':') or "=" in line or "Equilibrium:" in line:
                # Make headers and equation lines bold
                explanation_html += f"<p><strong>{line}</strong></p>\n"
            elif line.startswith('Supply:') or line.startswith('Demand:'):
                # Add color to supply and demand lines
                color = "#3498db" if "Supply:" in line else "#e74c3c"
                explanation_html += f'<p><span style="color: {color};">{line}</span></p>\n'
            elif line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or line.startswith('4.') or line.startswith('5.') or line.startswith('6.'):
                # Add bullet styling to numbered points
                explanation_html += f'<p style="margin-left: 20px;">• {line[2:]}</p>\n'
            elif line.strip() == "":
                # Add spacing for empty lines
                explanation_html += '<br>\n'
            elif "==========" in line:
                # Skip separator lines
                continue
            else:
                explanation_html += f"<p>{line}</p>\n"
        
        # Create audio basename for the HTML
        audio_basename = os.path.basename(audio_path) if audio_path else ""
        
        # Detect if this is a supply or demand shift based on the html_path
        shift_type = "supply"
        if "demand_shift" in html_path:
            shift_type = "demand"
        
        # Set title based on shift type
        title = "Supply Shift: Impact of Production Cost Increase"
        if shift_type == "demand":
            title = "Demand Shift: Impact of Changing Consumer Preferences"
        
        # Create the synchronized HTML file with enhanced styling
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Economic Visualization with Synchronized Narration</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Roboto', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
            color: #333;
            line-height: 1.5;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            text-align: center;
        }}
        h1 {{
            font-size: 32px;
            margin-bottom: 5px;
        }}
        h2 {{
            font-size: 20px;
            font-weight: normal;
            color: #7f8c8d;
            margin-top: 0;
            margin-bottom: 30px;
        }}
        .container {{
            display: flex;
            flex-direction: column;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            padding: 25px;
            margin-bottom: 30px;
        }}
        .plot-container {{
            height: 650px;
            margin-bottom: 25px;
            border: none;
            overflow: hidden;
            width: 100%;
        }}
        .controls {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            padding: 15px;
            background: #f5f7fa;
            border-radius: 8px;
            margin-bottom: 25px;
        }}
        .button {{
            background-color: #3498db;
            border: none;
            color: white;
            padding: 12px 25px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 50px;
            transition: all 0.3s;
            font-weight: 500;
        }}
        .button:hover {{
            background-color: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .button:active {{
            transform: translateY(0);
        }}
        .button:disabled {{
            background-color: #bdc3c7;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }}
        .explanation {{
            background-color: #f8f9fa;
            border-left: 5px solid #3498db;
            padding: 20px;
            margin-bottom: 25px;
            border-radius: 0 8px 8px 0;
        }}
        .explanation p {{
            margin: 8px 0;
        }}
        .explanation strong {{
            color: #2c3e50;
        }}
        .narration-panel {{
            background-color: #eaf2f8;
            border-radius: 8px;
            padding: 20px;
            margin-top: 15px;
        }}
        #progress-bar {{
            width: 100%;
            height: 8px;
            background-color: #ecf0f1;
            border-radius: 4px;
            margin-top: 15px;
            overflow: hidden;
        }}
        #progress {{
            height: 100%;
            width: 0%;
            background-color: #3498db;
            border-radius: 4px;
            transition: width 0.1s;
        }}
        .key-value {{
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #ecf0f1;
            padding: 8px 0;
        }}
        .key {{
            font-weight: 500;
            color: #2c3e50;
        }}
        .value {{
            color: #3498db;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <h2>Interactive Visualization with Synchronized Audio Narration</h2>
    
    <div class="container">
        <!-- Use iframe to embed the original Plotly visualization -->
        <iframe id="plotly-frame" class="plot-container" src="{os.path.basename(html_path)}"></iframe>
        
        <div class="controls">
            <button id="play-btn" class="button">▶ Play with Narration</button>
            <button id="pause-btn" class="button" disabled>⏸ Pause</button>
            <button id="restart-btn" class="button">↻ Restart</button>
        </div>
        
        <div id="progress-bar">
            <div id="progress"></div>
        </div>
        
        <div class="narration-panel">
            <h3>Current Narration:</h3>
            <div id="narration-text">
                Click "Play with Narration" to start the interactive explanation.
            </div>
        </div>
    </div>
    
    <div class="container">
        <h3>Economic Analysis</h3>
        <div class="explanation">
            {explanation_html}
        </div>
        
        <div class="key-value">
            <span class="key">Initial Equilibrium:</span>
            <span class="value" id="initial-eq"></span>
        </div>
        <div class="key-value">
            <span class="key">New Equilibrium:</span>
            <span class="value" id="new-eq"></span>
        </div>
        <div class="key-value">
            <span class="key">Price Change:</span>
            <span class="value" id="price-change"></span>
        </div>
        <div class="key-value">
            <span class="key">Quantity Change:</span>
            <span class="value" id="quantity-change"></span>
        </div>
    </div>
    
    <div class="footer">
        Created with EnhancedEconomicsAnimation | Interactive Educational Tool
    </div>
    
    <audio id="narration" src="{audio_basename}" preload="auto"></audio>
    
    <script>
        // Frame control variables
        let currentFrame = 0;
        const totalFrames = 30;
        let animationInterval;
        
        // Access the Plotly iframe
        const plotlyFrame = document.getElementById('plotly-frame');
        
        // Setup animation controls and synchronization
        const audio = document.getElementById('narration');
        const playBtn = document.getElementById('play-btn');
        const pauseBtn = document.getElementById('pause-btn');
        const restartBtn = document.getElementById('restart-btn');
        const narrationText = document.getElementById('narration-text');
        const progress = document.getElementById('progress');
        
        // Display equilibrium information
        document.getElementById('initial-eq').textContent = "Quantity: 6.00, Price: $5.00";
        document.getElementById('new-eq').textContent = "Quantity: 4.00, Price: $6.00";
        document.getElementById('price-change').textContent = "+20% ($5.00 → $6.00)";
        document.getElementById('quantity-change').textContent = "-33% (6.00 → 4.00)";
        
        // Narration segments with timing for supply shift
        const supplySegments = [
            {{ text: "Welcome to this interactive visualization of a supply shift in a market.", time: 0, frame: 0 }},
            {{ text: "We're looking at a market where the initial supply and demand curves intersect to form an equilibrium.", time: 4, frame: 0 }},
            {{ text: "The initial equilibrium occurs at a quantity of 6 units and a price of $5.", time: 10, frame: 0 }},
            {{ text: "Now, let's see what happens when production costs increase, causing the supply curve to shift upward.", time: 15, frame: 5 }},
            {{ text: "As the supply curve shifts upward, notice how the equilibrium point moves.", time: 21, frame: 15 }},
            {{ text: "The equilibrium quantity decreases and the equilibrium price increases.", time: 26, frame: 25 }},
            {{ text: "The new equilibrium quantity is 4 units, which is lower than before. The new equilibrium price is $6, which is higher than before.", time: 30, frame: 30 }},
            {{ text: "This illustrates a fundamental principle in economics: when supply decreases, prices tend to rise and quantities traded tend to fall.", time: 38, frame: 30 }}
        ];
        
        // Narration segments with timing for demand shift
        const demandSegments = [
            {{ text: "Welcome to this interactive visualization of a demand shift in a market.", time: 0, frame: 0 }},
            {{ text: "We're looking at a market where the initial supply and demand curves intersect to form an equilibrium.", time: 4, frame: 0 }},
            {{ text: "The initial equilibrium occurs at a quantity of 6 units and a price of $5.", time: 10, frame: 0 }},
            {{ text: "Now, let's see what happens when consumer preferences change, causing the demand curve to shift leftward.", time: 15, frame: 5 }},
            {{ text: "As the demand curve shifts leftward, notice how the equilibrium point moves.", time: 21, frame: 15 }},
            {{ text: "The equilibrium quantity decreases and the equilibrium price decreases as well.", time: 26, frame: 25 }},
            {{ text: "The new equilibrium quantity is 4 units, which is lower than before. The new equilibrium price is $4, which is lower than before.", time: 30, frame: 30 }},
            {{ text: "This illustrates a fundamental principle in economics: when demand decreases, prices tend to fall and quantities traded tend to fall as well.", time: 38, frame: 30 }}
        ];
        
        // Select the appropriate narration segments based on the shift type
        const narrationSegments = "{shift_type}" === "demand" ? demandSegments : supplySegments;
        
        // When iframe is loaded
        plotlyFrame.onload = function() {{
            console.log("Plotly frame loaded");
        }};
        
        // Function to control the Plotly frame
        function animateToFrame(frameNumber) {{
            try {{
                const frameWindow = plotlyFrame.contentWindow;
                if (frameWindow && frameWindow.Plotly) {{
                    frameWindow.Plotly.animate(
                        frameWindow.document.querySelector('.js-plotly-plot').id, 
                        [{{name: `frame${{frameNumber}}`}}], 
                        {{
                            frame: {{ duration: 300, redraw: true }},
                            transition: {{ duration: 300 }},
                            mode: 'immediate'
                        }}
                    );
                    currentFrame = frameNumber;
                    return true;
                }}
            }} catch (e) {{
                console.error("Error animating frame:", e);
            }}
            return false;
        }}
        
        // Update narration text based on current time
        audio.addEventListener('timeupdate', () => {{
            const currentTime = audio.currentTime;
            let activeSegment = narrationSegments[0];
            
            // Find the current narration segment
            for (let i = 0; i < narrationSegments.length; i++) {{
                if (currentTime >= narrationSegments[i].time &&
                    (i === narrationSegments.length - 1 || currentTime < narrationSegments[i+1].time)) {{
                    activeSegment = narrationSegments[i];
                    break;
                }}
            }}
            
            // Update text
            narrationText.innerHTML = `<p style="color: #3498db; font-weight: 500;">${{activeSegment.text}}</p>`;
            
            // Update progress bar
            const progressPercent = (currentTime / audio.duration) * 100;
            progress.style.width = progressPercent + '%';
            
            // Update animation frame if not manually controlling
            if (animationInterval) {{
                const targetFrame = Math.min(activeSegment.frame, totalFrames);
                if (currentFrame !== targetFrame) {{
                    animateToFrame(targetFrame);
                }}
            }}
        }});
        
        // Play button
        playBtn.addEventListener('click', () => {{
            audio.play().catch(error => {{
                console.error("Audio playback failed:", error);
                narrationText.innerHTML = `<p style="color: red;">Error playing audio. Please ensure audio is enabled in your browser.</p>`;
            }});
            playBtn.disabled = true;
            pauseBtn.disabled = false;
            
            // Start animation sync
            animationInterval = true;
        }});
        
        // Pause button
        pauseBtn.addEventListener('click', () => {{
            audio.pause();
            playBtn.disabled = false;
            pauseBtn.disabled = true;
            
            // Stop animation sync
            animationInterval = false;
        }});
        
        // Restart button
        restartBtn.addEventListener('click', () => {{
            audio.pause();
            audio.currentTime = 0;
            playBtn.disabled = false;
            pauseBtn.disabled = true;
            
            // Reset animation
            animationInterval = false;
            animateToFrame(0);
            
            narrationText.innerHTML = '<p>Click "Play with Narration" to start the interactive explanation.</p>';
            progress.style.width = '0%';
        }});
        
        // When audio ends
        audio.addEventListener('ended', () => {{
            playBtn.disabled = false;
            pauseBtn.disabled = true;
            animationInterval = false;
        }});
    </script>
</body>
</html>
        """
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the HTML file
        output_path = os.path.join(output_dir, f"enhanced_interactive_{shift_type}_shift.html")
        with open(output_path, "w") as f:
            f.write(html_template)
        
        # Copy the audio file to the output directory if it exists
        if audio_path and os.path.exists(audio_path):
            audio_output = os.path.join(output_dir, os.path.basename(audio_path))
            # Only copy if source and destination are not the same file
            if os.path.abspath(audio_path) != os.path.abspath(audio_output):
                shutil.copy2(audio_path, audio_output)
        
        return output_path

def main():
    """Run the enhanced animation generator"""
    # Create the animation generator
    animator = EnhancedEconomicsAnimation(output_dir="./output/enhanced")
    
    # Create supply shift animation
    result = animator.create_supply_demand_animation(
        supply_slope=0.5,
        supply_intercept=2,
        demand_slope=-0.5,
        demand_intercept=8,
        new_supply_intercept=4,
        frames=30,
        title="Supply Shift: Impact of Production Cost Increase",
        subtitle="Interactive Analysis with Audio Narration"
    )
    
    print("\nGenerated files:")
    for key, path in result.items():
        print(f"- {key}: {path}")
    
    # Open the interactive HTML file
    if "interactive_html" in result and result["interactive_html"] and os.path.exists(result["interactive_html"]):
        print("\nOpening enhanced interactive visualization...")
        if sys.platform == 'darwin':  # macOS
            import subprocess
            subprocess.run(['open', result["interactive_html"]])
        elif sys.platform == 'win32':  # Windows
            os.startfile(result["interactive_html"])
        else:  # Linux or other Unix
            import subprocess
            subprocess.run(['xdg-open', result["interactive_html"]])
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 