"""
Utility functions for deception detection system
"""

import plotly.graph_objects as go

def get_color_for_score(score, is_hex=True):
    """Get color based on deception score"""
    if score >= 60:
        return '#dc3545' if is_hex else 'red'
    elif score >= 40:
        return '#ffc107' if is_hex else 'orange'
    else:
        return '#28a745' if is_hex else 'green'

def get_classification_for_score(score):
    """Get classification label based on score"""
    if score >= 60:
        return 'HIGH PROBABILITY OF DECEPTION'
    elif score >= 40:
        return 'POSSIBLE DECEPTION'
    else:
        return 'LOW PROBABILITY OF DECEPTION'

def create_gauge_chart(score, title="Deception Score", bar_color="#667eea"):
    """Create a gauge chart for score visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': bar_color},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 40], 'color': "rgba(40,167,69,0.3)"},
                {'range': [40, 60], 'color': "rgba(255,193,7,0.3)"},
                {'range': [60, 100], 'color': "rgba(220,53,69,0.3)"}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white'},
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig