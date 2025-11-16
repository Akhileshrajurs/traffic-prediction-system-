# Traffic Congestion Prediction using GPS and Big Data

A comprehensive Python mini-project that demonstrates traffic congestion prediction using machine learning, GPS data, and advanced visualizations. Perfect for college-level projects and learning data science workflows.

## 🚀 Project Overview

This project provides a complete end-to-end solution for traffic congestion prediction:

- **Data Processing**: Cleans and preprocesses GPS traffic data from CSV files
- **Machine Learning**: Trains a RandomForest classifier to predict congestion levels
- **Visualizations**: Creates interactive maps, statistical dashboards, and analysis charts
- **Export**: Saves predictions and statistics in multiple formats (CSV, JSON)
- **Analysis**: Provides peak hour analysis, feature importance, and correlation matrices

## ✨ Key Features

### 🗺️ Enhanced Interactive Map
- **Color-coded markers**: Green (Low), Orange (Medium), Red (High) congestion
- **Variable marker sizes**: Based on vehicle count
- **Heatmap overlay**: Visual representation of congestion hotspots
- **Multiple tile layers**: OpenStreetMap, CartoDB Positron, Dark Matter
- **Interactive popups**: Detailed information for each location
- **Statistics panel**: Real-time summary of congestion distribution
- **Legend**: Clear explanation of congestion levels

### 📊 Advanced Visualizations
- **Feature Importance Plot**: Shows which features matter most for predictions
- **Correlation Matrix**: Identifies relationships between variables
- **Statistical Dashboard**: 4-panel visualization with:
  - Congestion distribution pie chart
  - Speed distribution histogram
  - Speed vs Vehicle count scatter plot
  - Time-based congestion analysis
- **Diagnostic Plots**: Speed vs vehicle count by congestion level

### 📈 Statistical Analysis
- **Comprehensive Statistics**: Average speeds, vehicle counts, ranges
- **Peak Hour Analysis**: Identifies busiest traffic hours
- **Congestion Distribution**: Breakdown by level and percentage
- **Confidence Scores**: Prediction reliability for each location

### 💾 Export Capabilities
- **CSV Export**: Predictions in spreadsheet format
- **JSON Export**: Structured data for APIs or further processing
- **Statistics Export**: Complete analysis results in JSON

## 📁 Project Structure

```
.
├── data/
│   └── traffic_data.csv          # Training dataset (200+ samples)
├── models/
│   └── traffic_congestion_model.joblib  # Trained model (created after training)
├── reports/
│   ├── speed_vs_vehicle.png      # Diagnostic scatter plot
│   ├── feature_importance.png    # Feature importance visualization
│   ├── correlation_matrix.png    # Correlation heatmap
│   └── statistical_dashboard.png # 4-panel analysis dashboard
├── exports/
│   ├── predictions.csv           # Exported predictions
│   ├── predictions.json          # JSON format predictions
│   └── statistics.json           # Analysis statistics
├── src/
│   ├── data_preprocessing.py     # Data loading and cleaning
│   ├── model_training.py         # Model training and evaluation
│   ├── gps_input.py              # GPS simulation and prediction
│   ├── map_display.py            # Enhanced Folium map creation
│   ├── visualizations.py         # Charts and plots
│   ├── analysis.py               # Statistical analysis
│   ├── export_utils.py           # Export functionality
│   └── main.py                   # Main execution script
├── traffic_map.html              # Interactive map (created after running)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- 16 GB RAM recommended (tested on Intel i7)
- Windows/macOS/Linux

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### Step 1: Train the Model

```bash
python src/model_training.py
```

This will:
- ✅ Load and clean the dataset
- ✅ Train the RandomForest classifier
- ✅ Display accuracy and classification report
- ✅ Generate diagnostic plots:
  - `reports/speed_vs_vehicle.png`
  - `reports/feature_importance.png`
  - `reports/correlation_matrix.png`
- ✅ Save the trained model to `models/traffic_congestion_model.joblib`

**Expected Output:**
```
Model Evaluation
----------------
Accuracy: 0.85+

Classification Report:
              precision    recall  f1-score   support
        High        ...
         Low        ...
      Medium        ...
```

### Step 2: Run Predictions and Generate Visualizations

```bash
python src/main.py
```

This will:
- ✅ Generate 30 balanced GPS points (10 Low, 10 Medium, 10 High)
- ✅ Predict congestion levels with confidence scores
- ✅ Display detailed statistics and analysis
- ✅ Create enhanced interactive map (`traffic_map.html`)
- ✅ Generate statistical dashboard (`reports/statistical_dashboard.png`)
- ✅ Export predictions and statistics to CSV/JSON

**Expected Output:**
```
Live Congestion Predictions
---------------------------
1. Location (12.973069, 77.602634) - Speed: 24.01 km/h - Vehicles: 42 - Congestion: High (Confidence: 0.92)
...

==================================================
Congestion Level Summary:
==================================================
🟢 Low: 10 locations
🟡 Medium: 10 locations
🔴 High: 10 locations
==================================================

📊 TRAFFIC CONGESTION ANALYSIS REPORT
...
```

### Step 3: View Results

1. **Interactive Map**: Open `traffic_map.html` in your web browser
   - Click markers for details
   - Toggle heatmap overlay
   - Switch between map styles
   - View statistics panel

2. **Visualizations**: Check the `reports/` folder for:
   - Feature importance chart
   - Correlation matrix
   - Statistical dashboard

3. **Exports**: Check the `exports/` folder for:
   - `predictions.csv` - Spreadsheet format
   - `predictions.json` - JSON format
   - `statistics.json` - Complete analysis

## 📊 Dataset Information

### Dataset Structure
The `traffic_data.csv` contains 200+ samples with the following columns:

| Column | Description | Range/Format |
|--------|-------------|--------------|
| `latitude` | GPS latitude | 12.96 - 12.98 |
| `longitude` | GPS longitude | 77.59 - 77.61 |
| `speed(kmph)` | Vehicle speed | 5 - 60 km/h |
| `timestamp` | Date and time | YYYY-MM-DD HH:MM:SS |
| `vehicle_count` | Number of vehicles | 10 - 55 |

### Congestion Level Labels

Labels are automatically created based on speed:

- **Low** 🟢: Speed > 40 km/h (Free flow)
- **Medium** 🟡: Speed 20-40 km/h (Moderate traffic)
- **High** 🔴: Speed < 20 km/h (Heavy congestion)

## 🎨 Features Explained

### Enhanced Map Features

1. **Heatmap Overlay**: Shows congestion intensity across the area
   - Blue = Low congestion
   - Green = Medium congestion
   - Orange/Red = High congestion

2. **Variable Marker Sizes**: Larger markers indicate more vehicles

3. **Multiple Tile Layers**: Switch between different map styles

4. **Statistics Panel**: Real-time summary in the top-right corner

### Statistical Analysis

The project provides comprehensive analysis including:

- **Average speeds and vehicle counts**
- **Speed and vehicle count ranges**
- **Peak hour identification**
- **Congestion distribution percentages**
- **Prediction confidence scores**

### Export Formats

**CSV Format** (`exports/predictions.csv`):
```csv
latitude,longitude,speed(kmph),vehicle_count,congestion_level,confidence
12.9715,77.5946,48.2,20,Low,0.95
...
```

**JSON Format** (`exports/predictions.json`):
```json
[
  {
    "latitude": 12.9715,
    "longitude": 77.5946,
    "speed(kmph)": 48.2,
    "vehicle_count": 20,
    "congestion_level": "Low",
    "confidence": 0.95
  }
]
```

## 🔧 Customization

### Adjust Number of Predictions

Edit `src/main.py`:
```python
simulated_points = simulate_balanced_gps_points(
    num_per_level=15,  # Change from 10 to 15 for 45 total points
    bounds=model_bundle.training_data_bounds,
)
```

### Modify Model Parameters

Edit `src/model_training.py`:
```python
clf = RandomForestClassifier(
    n_estimators=300,  # Increase trees for better accuracy
    max_depth=10,      # Control tree depth
    random_state=42,
)
```

### Add Your Own Data

1. Add rows to `data/traffic_data.csv` with your GPS data
2. Retrain the model: `python src/model_training.py`
3. Run predictions: `python src/main.py`

## 📈 Model Performance

The RandomForest model typically achieves:
- **Accuracy**: 80-90% (depending on data quality)
- **Features Used**: latitude, longitude, speed, vehicle_count, hour_of_day
- **Training Time**: < 5 seconds on standard hardware

## 🎓 Learning Outcomes

This project demonstrates:

1. **Data Preprocessing**: Cleaning, feature engineering, labeling
2. **Machine Learning**: Classification with RandomForest
3. **Model Evaluation**: Accuracy, precision, recall, F1-score
4. **Visualization**: Maps, charts, dashboards
5. **Data Export**: Multiple format support
6. **Statistical Analysis**: Descriptive statistics and insights

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError
**Solution**: Make sure virtual environment is activated and dependencies are installed:
```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: Map not displaying
**Solution**: 
- Ensure internet connection (map tiles load from web)
- Try different browser (Chrome, Firefox, Edge)
- Check browser console for errors (F12)

### Issue: Low model accuracy
**Solution**:
- Add more training data to `traffic_data.csv`
- Ensure balanced distribution of congestion levels
- Try adjusting RandomForest hyperparameters

## 📝 System Requirements

- **Python**: 3.10+
- **RAM**: 16 GB recommended
- **Storage**: < 100 MB for project files
- **Internet**: Required for map tiles (first load)

## 🚀 Future Enhancements

Potential improvements:
- Real-time GPS data integration
- Route optimization suggestions
- Historical trend analysis
- Web dashboard interface
- API endpoint for predictions
- Mobile app integration

## 📄 License

This project is provided as-is for educational purposes.

## 🙏 Acknowledgments

Built with:
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning
- **folium**: Interactive maps
- **matplotlib/seaborn**: Visualizations
- **numpy**: Numerical computing

---

**Enjoy exploring traffic analytics!** 🚗📊🗺️
