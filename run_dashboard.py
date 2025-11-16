"""
Quick launcher script for the web dashboard.
Run this file to start the Flask + Dash web dashboard.
"""

from src.web_dashboard import app, server

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  🚀 Traffic Congestion Prediction Web Dashboard")
    print("="*70)
    print("  📍 Dashboard URL: http://127.0.0.1:8050")
    print("  📊 API Base URL: http://127.0.0.1:8050/api/")
    print("  ⚠️  Make sure you've trained the model first!")
    print("  💡 Run 'python src/model_training.py' if you haven't")
    print("="*70 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=8050)

