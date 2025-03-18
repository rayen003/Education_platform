# UI Components for Math Services

This directory contains UI components for displaying math assessment results and confidence metrics.

## Overview

The UI components provide consistent styling and interactive elements for the math assessment system, with a special focus on visualizing confidence metrics and ensuring users understand the reliability of AI-generated responses.

## Confidence Display

The `confidence_display.py` module provides reusable components for visualizing confidence levels:

### Components

1. **Confidence Bars** (`display_confidence_bar`):
   - Full-width progress bars with coloring based on confidence level
   - Green for high confidence (>= 0.8)
   - Orange for medium confidence (>= 0.6)
   - Red for low confidence (< 0.6)
   - Includes percentage and categorical label (High/Medium/Low)

2. **Confidence Badges** (`display_confidence_badge`):
   - Compact inline indicators for showing confidence
   - Color-coded labels with percentage values
   - Designed for use with individual items (like hints)

3. **Confidence Tooltips** (`display_confidence_tooltip`):
   - Contextual tooltips with confidence icons
   - Provides explanation alongside confidence level
   - Checkmark for high confidence, info icon for medium, warning for low

4. **Confidence Explanations** (`confidence_explanation`):
   - Generates human-readable explanations of confidence levels
   - Explains factors that might affect confidence

### Usage Examples

```python
# Display a confidence bar with a label
display_confidence_bar(0.85, "Analysis Confidence")

# Display a compact confidence badge
display_confidence_badge(0.70)

# Display text with a confidence tooltip
display_confidence_tooltip(0.90, "This calculation step is correct.")

# Get explanatory text for a confidence score
explanation = confidence_explanation(0.65)
```

## Integration with Math Assessment

These components integrate with the math assessment workflow to provide:

1. Confidence indicators for problem solutions
2. Confidence levels for analysis of student work
3. Confidence ratings for feedback
4. Reliability scores for hints
5. Verification confidence for reasoning steps
6. Accuracy ratings for chat responses

## Best Practices

- Always include confidence indicators for AI-generated content
- Provide explanations for confidence levels with expanders
- Use consistent color coding across the application
- Consider user-customizable confidence thresholds
- Log confidence metrics for calibration and improvement

## Development

To update or extend these components:

1. Ensure consistent styling with the main application
2. Follow the established color scheme for confidence levels
3. Test with various screen sizes for responsive design
4. Add comprehensive documentation for new components 