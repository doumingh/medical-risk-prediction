import os
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils
import json
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration Section ====================
# Configure your model file path here
# Method 1: Direct filename (Recommended, model file in project root)
MODEL_FILE_PATH = "best_catboost_model_new.cbm"

# Method 2: Relative path
# MODEL_FILE_PATH = "models/best_catboost_model.cbm"

# Method 3: Absolute path (Windows example)
# MODEL_FILE_PATH = "E:/private/mingfei/best_catboost_model.cbm"
# MODEL_FILE_PATH = "C:\\Users\\YourName\\Desktop\\best_catboost_model.cbm"

# Method 4: Absolute path (Linux example)
# MODEL_FILE_PATH = "/home/username/projects/mingfei/best_catboost_model.cbm"

# Method 5: Dynamic path (Most flexible, auto-detection)
# import os
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_FILE_PATH = os.path.join(CURRENT_DIR, "best_catboost_model.cbm")

# Other configuration options
DEFAULT_AGE = 48
DEFAULT_POLYPS = 1
DEFAULT_LONG_DIAMETER = 10.0
DEFAULT_SHORT_DIAMETER = 6.0
DEFAULT_FUNDUS = 1  # 1=Pedicle, 2=Broad base
# ================================================

class MedicalRiskPredictor:
    def __init__(self):
        self.model = None
        self.features = ["Age", "Number of polyps", "Long diameter", "Short diameter", "Base"]
        
        # Risk stratification configuration
        self.bins = [
            (0.00, 0.25, "Low Risk", "Low Risk"),
            (0.25, 0.50, "Moderate Risk", "Moderate Risk"),
            (0.50, 0.75, "High Risk", "High Risk"),
            (0.75, 1.00, "Very High Risk", "Very High Risk"),
        ]
        
        # Treatment recommendations
        self.advice = {
            "Low Risk": "Follow-up is not required",
            "Moderate Risk": "Follow-up ultrasound is recommended at 6 months, 1 year, and 2 years;\n Follow-up should be discontinued after 2 years in the absence of growth.",
            "High Risk": "Cholecystectomy is recommended if the patient is fit for, and accepts, surgery;\n MDT discussion may be considered",
            "Very High Risk": "Cholecystectomy is strongly recommended if the patient is fit for, and accepts, surgery"
        }
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load CatBoost model"""
        try:
            if not os.path.exists(MODEL_FILE_PATH):
                print(f"Warning: Model file does not exist: {MODEL_FILE_PATH}")
                print("Please check the MODEL_FILE_PATH setting in the configuration section")
                return False
            
            self.model = CatBoostClassifier()
            self.model.load_model(MODEL_FILE_PATH)
            print(f"Model loaded successfully: {MODEL_FILE_PATH}")
            return True
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            return False
    
    def predict_risk(self, age, polyps, long_diameter, short_diameter, Base):
        """Predict risk probability"""
        try:
            if self.model is None:
                return None, "Model not loaded"
            
            # Prepare input data
            # Base: 1=Pedicle, 2=Broad base, use the value sent from frontend directly
            input_data = np.array([[age, polyps, long_diameter, short_diameter, Base]])
            
            # Predict probability
            probability = self.model.predict_proba(input_data)[0][1]
            
            # Determine risk level
            risk_level = None
            risk_level_cn = None
            for min_prob, max_prob, level, level_cn in self.bins:
                if min_prob <= probability < max_prob:
                    risk_level = level
                    risk_level_cn = level_cn
                    break
            
            # Get treatment recommendations
            advice = self.advice.get(risk_level, "No advice")
            
            return {
                'probability': probability,
                'risk_level': risk_level,
                'risk_level_cn': risk_level_cn,
                'advice': advice
            }, None
            
        except Exception as e:
            return None, f"Prediction failed: {str(e)}"
    
    def _to_base_value(self, ev):
        """Convert TreeExplainer.expected_value to scalar (compatible with array/list cases)"""
        if isinstance(ev, (list, tuple, np.ndarray)):
            return float(np.ravel(ev)[0])
        return float(ev)
    
    def _cleanup_old_requests(self, keep_count=10):
        """Clean up old request ID directories, keep only the recent ones"""
        try:
            result_dir = "result"
            if not os.path.exists(result_dir):
                return
            
            # Get all request ID directories
            request_dirs = []
            for item in os.listdir(result_dir):
                item_path = os.path.join(result_dir, item)
                if os.path.isdir(item_path) and len(item) == 36:  # UUID length
                    # Get directory creation time
                    creation_time = os.path.getctime(item_path)
                    request_dirs.append((item_path, creation_time))
            
            # Sort by creation time, keep the latest
            request_dirs.sort(key=lambda x: x[1], reverse=True)
            
            # Delete excess old directories
            for dir_path, _ in request_dirs[keep_count:]:
                try:
                    import shutil
                    shutil.rmtree(dir_path)
                    print(f"🗑️ Cleaned up old request directory: {os.path.basename(dir_path)}")
                except Exception as e:
                    print(f"⚠️ Failed to clean up old directory {dir_path}: {str(e)}")
            
            print(f"✅ Request directory cleanup completed, keeping {min(len(request_dirs), keep_count)} directories")
            
        except Exception as e:
            print(f"⚠️ Error occurred while cleaning up old request directories: {str(e)}")
    
    def generate_shap_plots(self, age, polyps, long_diameter, short_diameter, Base, request_id=None):
        """Generate two SHAP charts according to original design, using matplotlib to ensure exact match"""
        try:
            print("=" * 50)
            print("Starting SHAP chart generation...")
            print("=" * 50)
            
            # Set matplotlib font to Times New Roman
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Times New Roman']
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.titlesize'] = 14
            plt.rcParams['axes.labelsize'] = 12
            plt.rcParams['xtick.labelsize'] = 10
            plt.rcParams['ytick.labelsize'] = 10
            plt.rcParams['legend.fontsize'] = 10
            plt.rcParams['figure.titlesize'] = 16
            plt.rcParams['font.weight'] = 'normal'
            plt.rcParams['axes.titleweight'] = 'normal'
            plt.rcParams['axes.labelweight'] = 'normal'
            print("✅ Font settings completed: Times New Roman")
            
            if self.model is None:
                print("❌ Model not loaded, cannot generate SHAP charts")
                return None, None
            
            print("✅ Model loaded")
            
            # Prepare input data
            # Base: 1=Pedicle, 2=Broad base, use the value sent from frontend directly
            input_data = np.array([[age, polyps, long_diameter, short_diameter, Base]])
            input_df = pd.DataFrame(input_data, columns=self.features)
            
            print(f"📊 Input data: {input_data}")
            print(f"🏷️ Feature names: {self.features}")
            
            # Calculate SHAP values
            print("🔄 Calculating SHAP values...")
            try:
                explainer = shap.TreeExplainer(self.model)
                print("✅ SHAP explainer created successfully")
                
                shap_vals = explainer.shap_values(input_df)
                print(f"📈 SHAP raw values: {shap_vals}")
                
                if isinstance(shap_vals, list):  # Some versions return list
                    shap_vals = shap_vals[0]
                    print("✅ SHAP values extracted from list")
                
                shap_vals = np.array(shap_vals).reshape(-1)  # (n_features,)
                base_value = self._to_base_value(explainer.expected_value)
                
                print(f"🎯 Final SHAP values: {shap_vals}")
                print(f"🔢 Base value: {base_value}")
                print(f"📏 SHAP values shape: {shap_vals.shape}")
                
            except Exception as shap_e:
                print(f"❌ SHAP calculation failed: {str(shap_e)}")
                import traceback
                traceback.print_exc()
                raise shap_e
            
            # Create result directory and request ID subdirectory
            if request_id:
                result_dir = os.path.join("result", request_id)
                print(f"📁 Creating request ID directory: {result_dir}")
                os.makedirs(result_dir, exist_ok=True)
                print(f"✅ Request ID directory path: {os.path.abspath(result_dir)}")
            else:
                result_dir = "result"
                print("📁 Creating result directory...")
                os.makedirs(result_dir, exist_ok=True)
                print(f"✅ Result directory path: {os.path.abspath(result_dir)}")
            
            # Clean up old request ID directories (keep recent 10)
            if request_id:
                self._cleanup_old_requests()
            
            # Generate waterfall chart
            print("🌊 Generating waterfall chart...")
            waterfall_filename = "shap_waterfall.png"
            waterfall_path = os.path.join(result_dir, waterfall_filename)
            print(f"📍 Waterfall chart save path: {waterfall_path}")
            
            try:
                plt.figure(figsize=(8, 6), dpi=300)
                print("✅ matplotlib figure created successfully")
                
                # Try to use SHAP native waterfall chart
                try:
                    print("🔄 Trying SHAP native waterfall chart...")
                    shap.plots._waterfall.waterfall_legacy(
                        base_value,
                        shap_vals,
                        feature_names=self.features,
                        max_display=min(12, len(self.features)),
                        show=False
                    )
                    print("✅ Using SHAP native waterfall chart successfully")
                except Exception as e:
                    print(f"⚠️ SHAP native waterfall chart failed, using fallback: {str(e)}")
                    # Fallback: bar chart sorted by absolute values
                    order = np.argsort(-np.abs(shap_vals))
                    plt.bar(range(len(self.features)), shap_vals[order])
                    plt.xticks(range(len(self.features)), np.array(self.features)[order], rotation=45, ha="right")
                    plt.title("SHAP (fallback bar plot)", fontfamily='Times New Roman', fontsize=14)
                    plt.xlabel("Features", fontfamily='Times New Roman', fontsize=12)
                    plt.ylabel("SHAP Values", fontfamily='Times New Roman', fontsize=12)
                    print("✅ Fallback waterfall chart created successfully")
                
                plt.tight_layout()
                print("✅ Figure layout adjustment completed")
                
                plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
                print(f"✅ Waterfall chart saved successfully: {waterfall_path}")
                
                plt.close()
                print("✅ Waterfall chart matplotlib object closed")
                
            except Exception as waterfall_e:
                print(f"❌ Waterfall chart generation failed: {str(waterfall_e)}")
                import traceback
                traceback.print_exc()
                raise waterfall_e
            
            # Generate force chart
            print("💪 Generating force chart...")
            force_filename = "shap_force.png"
            force_path = os.path.join(result_dir, force_filename)
            print(f"📍 Force chart save path: {force_path}")
            
            try:
                plt.figure(figsize=(10, 2.6), dpi=300)
                print("✅ matplotlib force chart figure created successfully")
                
                # Use SHAP native force chart
                try:
                    print("🔄 Trying SHAP native force chart...")
                    shap.force_plot(
                        base_value,
                        shap_vals,
                        input_df.iloc[0, :],
                        matplotlib=True,
                        contribution_threshold=0.01,
                        figsize=(12, 3),
                        show=False
                    )
                    print("✅ Using SHAP native force chart successfully")
                except Exception as e:
                    print(f"⚠️ SHAP native force chart failed, using fallback: {str(e)}")
                    # Fallback: simple line chart
                    pred_like = base_value + shap_vals.sum()
                    plt.axhline(0)
                    plt.plot([0, 1], [base_value, pred_like])
                    plt.title("Force (fallback line)", fontfamily='Times New Roman', fontsize=14)
                    plt.xlabel("Prediction", fontfamily='Times New Roman', fontsize=12)
                    plt.ylabel("Value", fontfamily='Times New Roman', fontsize=12)
                    print("✅ Fallback force chart created successfully")
                
                plt.tight_layout()
                plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
                plt.gca().tick_params(axis='x', labelsize=16)
                print("✅ Force chart layout adjustment completed")
                
                plt.savefig(force_path, dpi=300, bbox_inches="tight")
                print(f"✅ Force chart saved successfully: {force_path}")
                
                plt.close()
                print("✅ Force chart matplotlib object closed")
                
            except Exception as force_e:
                print(f"❌ Force chart generation failed: {str(force_e)}")
                import traceback
                traceback.print_exc()
                raise force_e
            
            # Verify if files were actually created
            if os.path.exists(waterfall_path):
                print(f"✅ Waterfall chart file actually exists: {waterfall_path}")
                print(f"📏 File size: {os.path.getsize(waterfall_path)} bytes")
            else:
                print(f"❌ Waterfall chart file does not exist: {waterfall_path}")
            
            if os.path.exists(force_path):
                print(f"✅ Force chart file actually exists: {force_path}")
                print(f"📏 File size: {os.path.getsize(force_path)} bytes")
            else:
                print(f"❌ Force chart file does not exist: {force_path}")
            
            print("🎉 Two SHAP charts generated successfully!")
            print("=" * 50)
            return waterfall_path, force_path
            
        except Exception as e:
            print(f"❌ SHAP chart generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            print("=" * 50)
            return None, None

# Create FastAPI application
app = FastAPI(title="Medical Risk Prediction System", version="1.0.0")

# Mount result directory, allowing SHAP images to be accessed
app.mount("/result", StaticFiles(directory="result"), name="result")

# Set templates
templates = Jinja2Templates(directory="templates")

# Create predictor instance
predictor = MedicalRiskPredictor()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page - only show form, don't show historical results"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "default_age": DEFAULT_AGE,
        "default_polyps": DEFAULT_POLYPS,
        "default_long_diameter": DEFAULT_LONG_DIAMETER,
        "default_short_diameter": DEFAULT_SHORT_DIAMETER,
        "default_fundus": DEFAULT_FUNDUS,
        "model_loaded": predictor.model is not None,
        # Don't pass result-related parameters, ensure page displays initial state
        "result": None,
        "waterfall_path": None,
        "force_path": None,
        "error": None
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: int = Form(...),
    polyps: int = Form(...),
    long_diameter: float = Form(...),
    short_diameter: float = Form(...),
    Base: int = Form(...),  # Changed to int type, accepts 1 or 2
    request_id: str = Form(...)
):
    """Execute prediction"""
    # Execute prediction
    result, error = predictor.predict_risk(age, polyps, long_diameter, short_diameter, Base)
    
    if error:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": error,
            "request_id": request_id,
            "default_age": age,
            "default_polyps": polyps,
            "default_long_diameter": long_diameter,
            "default_short_diameter": short_diameter,
            "default_fundus": Base,
            "model_loaded": predictor.model is not None
        })
    
    # Generate SHAP charts
    print("Generating SHAP charts...")
    waterfall_path, force_path = predictor.generate_shap_plots(age, polyps, long_diameter, short_diameter, Base, request_id)
    
    print(f"Waterfall chart path: {waterfall_path}")
    print(f"Force chart path: {force_path}")
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "waterfall_path": waterfall_path,
        "force_path": force_path,
        "request_id": request_id,
        "default_age": age,
        "default_polyps": polyps,
        "default_long_diameter": long_diameter,
        "default_short_diameter": short_diameter,
        "default_fundus": Base,
        "model_loaded": predictor.model is not None
    })

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "model_path": MODEL_FILE_PATH
    }

if __name__ == "__main__":
    print("=" * 50)
    print("🏥 Medical Risk Prediction System starting...")
    print("=" * 50)
    
    # Check model file
    if not os.path.exists(MODEL_FILE_PATH):
        print(f"❌ Model file does not exist: {MODEL_FILE_PATH}")
        print("Please check the MODEL_FILE_PATH setting in the configuration section")
        print("=" * 50)
    else:
        print(f"✅ Model file path: {MODEL_FILE_PATH}")
    
    # Start service
    print("🚀 Starting Web service...")
    print("📱 Access address: http://localhost:8001")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
