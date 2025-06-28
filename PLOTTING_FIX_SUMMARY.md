# Plotting Fix Summary - Non-blocking Plot Display

## Issue
The plotting functions were calling `plt.show()` which blocks script execution, causing scripts to hang while waiting for plot windows to be closed. This was particularly problematic for `plot_error_summary()` which is commonly used in analysis scripts.

## Solution
Implemented intelligent plot display handling that auto-detects the execution environment:
- **Jupyter/IPython**: Shows plots inline (interactive mode)
- **Scripts**: Saves plots without displaying (non-blocking mode)
- **Manual Override**: New `show` parameter for explicit control

## Changes Made

### 1. Added Helper Function (`locator/plotting.py`)
```python
def _handle_plot_display(show=None):
    """Handle whether to display a plot based on environment."""
    if show is None:
        # Auto-detect: show only if in interactive environment
        try:
            get_ipython()  # Defined in IPython/Jupyter
            plt.show()
        except NameError:
            # Not in interactive environment, don't show
            pass
    elif show:
        plt.show()
```

### 2. Updated Plotting Functions
Added `show=None` parameter to:
- `plot_predictions()`
- `plot_error_summary()` 
- `plot_sample_weights()`

### 3. Fixed plt.show() Calls
Replaced all `plt.show()` with `_handle_plot_display(show)` and added `plt.close()`

### 4. Updated predict_holdout()
Added `show=True` when calling `plot_error_summary()` from notebook context

## Benefits
- Scripts run to completion without hanging
- Plots still saved to disk (.png files)
- Notebooks continue to show plots inline
- Backward compatible (default behavior is auto-detect)
- Manual control available when needed

## Usage Examples

### In Scripts (Non-blocking)
```python
# Auto-detects script environment, won't show plot window
plot_error_summary(predictions, sample_data, out_prefix="results")

# Explicitly never show
plot_error_summary(predictions, sample_data, show=False)
```

### In Notebooks (Interactive)
```python
# Auto-detects notebook environment, shows plot inline
plot_error_summary(predictions, sample_data)

# Explicitly always show
plot_error_summary(predictions, sample_data, show=True)
```

## Files Modified
- `locator/plotting.py`: Added helper function and updated all plotting functions
- `locator/prediction.py`: Added show=True for notebook context