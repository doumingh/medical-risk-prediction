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

# ==================== 配置区域 ====================
# 在这里配置你的模型文件路径
# 方法1：直接指定文件名（推荐，模型文件放在项目根目录）
MODEL_FILE_PATH = "best_catboost_model_new.cbm"

# 方法2：指定相对路径
# MODEL_FILE_PATH = "models/best_catboost_model.cbm"

# 方法3：指定绝对路径（Windows示例）
# MODEL_FILE_PATH = "E:/private/mingfei/best_catboost_model.cbm"
# MODEL_FILE_PATH = "C:\\Users\\YourName\\Desktop\\best_catboost_model.cbm"

# 方法4：指定绝对路径（Linux示例）
# MODEL_FILE_PATH = "/home/username/projects/mingfei/best_catboost_model.cbm"

# 方法5：动态路径（最灵活，自动检测）
# import os
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_FILE_PATH = os.path.join(CURRENT_DIR, "best_catboost_model.cbm")

# 其他配置选项
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
        
        # 风险分层配置
        self.bins = [
            (0.00, 0.25, "Low Risk", "Low Risk"),
            (0.25, 0.50, "Moderate Risk", "Moderate Risk"),
            (0.50, 0.75, "High Risk", "High Risk"),
            (0.75, 1.00, "Very High Risk", "Very High Risk"),
        ]
        
        # 诊疗建议
        self.advice = {
            "Low Risk": "Follow-up is not required",
            "Moderate Risk": "Follow-up ultrasound is recommended at 6 months, 1 year, and 2 years;\n Follow-up should be discontinued after 2 years in the absence of growth.",
            "High Risk": "Cholecystectomy is recommended if the patient is fit for, and accepts, surgery;\n MDT discussion may be considered",
            "Very High Risk": "Cholecystectomy is strongly recommended if the patient is fit for, and accepts, surgery"
        }
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """加载CatBoost模型"""
        try:
            if not os.path.exists(MODEL_FILE_PATH):
                print(f"警告: 模型文件不存在: {MODEL_FILE_PATH}")
                print("请检查配置区域中的MODEL_FILE_PATH设置")
                return False
            
            self.model = CatBoostClassifier()
            self.model.load_model(MODEL_FILE_PATH)
            print(f"模型加载成功: {MODEL_FILE_PATH}")
            return True
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return False
    
    def predict_risk(self, age, polyps, long_diameter, short_diameter, Base):
        """预测风险概率"""
        try:
            if self.model is None:
                return None, "模型未加载"
            
            # 准备输入数据
            # Base: 1=Pedicle, 2=Broad base，直接使用前端发送的值
            input_data = np.array([[age, polyps, long_diameter, short_diameter, Base]])
            
            # 预测概率
            probability = self.model.predict_proba(input_data)[0][1]
            
            # 确定风险等级
            risk_level = None
            risk_level_cn = None
            for min_prob, max_prob, level, level_cn in self.bins:
                if min_prob <= probability < max_prob:
                    risk_level = level
                    risk_level_cn = level_cn
                    break
            
            # 获取诊疗建议
            advice = self.advice.get(risk_level, "No advice")
            
            return {
                'probability': probability,
                'risk_level': risk_level,
                'risk_level_cn': risk_level_cn,
                'advice': advice
            }, None
            
        except Exception as e:
            return None, f"预测失败: {str(e)}"
    
    def _to_base_value(self, ev):
        """将 TreeExplainer.expected_value 统一转成标量（兼容数组/列表情况）"""
        if isinstance(ev, (list, tuple, np.ndarray)):
            return float(np.ravel(ev)[0])
        return float(ev)
    
    def _cleanup_old_requests(self, keep_count=10):
        """清理旧的请求ID目录，只保留最近的几个"""
        try:
            result_dir = "result"
            if not os.path.exists(result_dir):
                return
            
            # 获取所有请求ID目录
            request_dirs = []
            for item in os.listdir(result_dir):
                item_path = os.path.join(result_dir, item)
                if os.path.isdir(item_path) and len(item) == 36:  # UUID长度
                    # 获取目录创建时间
                    creation_time = os.path.getctime(item_path)
                    request_dirs.append((item_path, creation_time))
            
            # 按创建时间排序，保留最新的
            request_dirs.sort(key=lambda x: x[1], reverse=True)
            
            # 删除多余的旧目录
            for dir_path, _ in request_dirs[keep_count:]:
                try:
                    import shutil
                    shutil.rmtree(dir_path)
                    print(f"🗑️ 已清理旧请求目录: {os.path.basename(dir_path)}")
                except Exception as e:
                    print(f"⚠️ 清理旧目录失败 {dir_path}: {str(e)}")
            
            print(f"✅ 请求目录清理完成，保留 {min(len(request_dirs), keep_count)} 个")
            
        except Exception as e:
            print(f"⚠️ 清理旧请求目录时出错: {str(e)}")
    
    def generate_shap_plots(self, age, polyps, long_diameter, short_diameter, Base, request_id=None):
        """按照原始设计生成两个SHAP图表，使用matplotlib确保一模一样"""
        try:
            print("=" * 50)
            print("开始生成SHAP图表...")
            print("=" * 50)
            
            # 设置matplotlib字体为Times New Roman
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
            print("✅ 字体设置完成：Times New Roman")
            
            if self.model is None:
                print("❌ 模型未加载，无法生成SHAP图")
                return None, None
            
            print("✅ 模型已加载")
            
            # 准备输入数据
            # Base: 1=Pedicle, 2=Broad base，直接使用前端发送的值
            input_data = np.array([[age, polyps, long_diameter, short_diameter, Base]])
            input_df = pd.DataFrame(input_data, columns=self.features)
            
            print(f"📊 输入数据: {input_data}")
            print(f"🏷️ 特征名称: {self.features}")
            
            # 计算SHAP值
            print("🔄 正在计算SHAP值...")
            try:
                explainer = shap.TreeExplainer(self.model)
                print("✅ SHAP解释器创建成功")
                
                shap_vals = explainer.shap_values(input_df)
                print(f"📈 SHAP原始值: {shap_vals}")
                
                if isinstance(shap_vals, list):  # 某些版本会返回 list
                    shap_vals = shap_vals[0]
                    print("✅ SHAP值从列表中提取")
                
                shap_vals = np.array(shap_vals).reshape(-1)  # (n_features,)
                base_value = self._to_base_value(explainer.expected_value)
                
                print(f"🎯 最终SHAP值: {shap_vals}")
                print(f"🔢 基础值: {base_value}")
                print(f"📏 SHAP值形状: {shap_vals.shape}")
                
            except Exception as shap_e:
                print(f"❌ SHAP计算失败: {str(shap_e)}")
                import traceback
                traceback.print_exc()
                raise shap_e
            
            # 创建result目录和请求ID子目录
            if request_id:
                result_dir = os.path.join("result", request_id)
                print(f"📁 创建请求ID目录: {result_dir}")
                os.makedirs(result_dir, exist_ok=True)
                print(f"✅ 请求ID目录路径: {os.path.abspath(result_dir)}")
            else:
                result_dir = "result"
                print("📁 创建result目录...")
                os.makedirs(result_dir, exist_ok=True)
                print(f"✅ result目录路径: {os.path.abspath(result_dir)}")
            
            # 清理旧的请求ID目录（保留最近10个）
            if request_id:
                self._cleanup_old_requests()
            
            # 生成瀑布图
            print("🌊 正在生成瀑布图...")
            waterfall_filename = "shap_waterfall.png"
            waterfall_path = os.path.join(result_dir, waterfall_filename)
            print(f"📍 瀑布图保存路径: {waterfall_path}")
            
            try:
                plt.figure(figsize=(8, 6), dpi=300)
                print("✅ matplotlib图形创建成功")
                
                # 尝试使用SHAP原生瀑布图
                try:
                    print("🔄 尝试SHAP原生瀑布图...")
                    shap.plots._waterfall.waterfall_legacy(
                        base_value,
                        shap_vals,
                        feature_names=self.features,
                        max_display=min(12, len(self.features)),
                        show=False
                    )
                    print("✅ 使用SHAP原生瀑布图成功")
                except Exception as e:
                    print(f"⚠️ SHAP原生瀑布图失败，使用备用方案: {str(e)}")
                    # 备用方案：按绝对值排序的条形图
                    order = np.argsort(-np.abs(shap_vals))
                    plt.bar(range(len(self.features)), shap_vals[order])
                    plt.xticks(range(len(self.features)), np.array(self.features)[order], rotation=45, ha="right")
                    plt.title("SHAP (fallback bar plot)", fontfamily='Times New Roman', fontsize=14)
                    plt.xlabel("Features", fontfamily='Times New Roman', fontsize=12)
                    plt.ylabel("SHAP Values", fontfamily='Times New Roman', fontsize=12)
                    print("✅ 备用瀑布图创建成功")
                
                plt.tight_layout()
                print("✅ 图形布局调整完成")
                
                plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
                print(f"✅ 瀑布图保存成功: {waterfall_path}")
                
                plt.close()
                print("✅ 瀑布图matplotlib对象关闭")
                
            except Exception as waterfall_e:
                print(f"❌ 瀑布图生成失败: {str(waterfall_e)}")
                import traceback
                traceback.print_exc()
                raise waterfall_e
            
            # 生成力图
            print("💪 正在生成力图...")
            force_filename = "shap_force.png"
            force_path = os.path.join(result_dir, force_filename)
            print(f"📍 力图保存路径: {force_path}")
            
            try:
                plt.figure(figsize=(10, 2.6), dpi=300)
                print("✅ matplotlib力图图形创建成功")
                
                # 使用SHAP原生力图
                try:
                    print("🔄 尝试SHAP原生力图...")
                    shap.force_plot(
                        base_value,
                        shap_vals,
                        input_df.iloc[0, :],
                        matplotlib=True,
                        contribution_threshold=0.01,
                        figsize=(12, 3),
                        show=False
                    )
                    print("✅ 使用SHAP原生力图成功")
                except Exception as e:
                    print(f"⚠️ SHAP原生力图失败，使用备用方案: {str(e)}")
                    # 备用方案：简单的线图
                    pred_like = base_value + shap_vals.sum()
                    plt.axhline(0)
                    plt.plot([0, 1], [base_value, pred_like])
                    plt.title("Force (fallback line)", fontfamily='Times New Roman', fontsize=14)
                    plt.xlabel("Prediction", fontfamily='Times New Roman', fontsize=12)
                    plt.ylabel("Value", fontfamily='Times New Roman', fontsize=12)
                    print("✅ 备用力图创建成功")
                
                plt.tight_layout()
                plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
                plt.gca().tick_params(axis='x', labelsize=16)
                print("✅ 力图布局调整完成")
                
                plt.savefig(force_path, dpi=300, bbox_inches="tight")
                print(f"✅ 力图保存成功: {force_path}")
                
                plt.close()
                print("✅ 力图matplotlib对象关闭")
                
            except Exception as force_e:
                print(f"❌ 力图生成失败: {str(force_e)}")
                import traceback
                traceback.print_exc()
                raise force_e
            
            # 验证文件是否真的创建了
            if os.path.exists(waterfall_path):
                print(f"✅ 瀑布图文件确实存在: {waterfall_path}")
                print(f"📏 文件大小: {os.path.getsize(waterfall_path)} 字节")
            else:
                print(f"❌ 瀑布图文件不存在: {waterfall_path}")
            
            if os.path.exists(force_path):
                print(f"✅ 力图文件确实存在: {force_path}")
                print(f"📏 文件大小: {os.path.getsize(force_path)} 字节")
            else:
                print(f"❌ 力图文件不存在: {force_path}")
            
            print("🎉 两个SHAP图表生成成功！")
            print("=" * 50)
            return waterfall_path, force_path
            
        except Exception as e:
            print(f"❌ SHAP图生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            print("=" * 50)
            return None, None

# 创建FastAPI应用
app = FastAPI(title="医疗风险预测系统", version="1.0.0")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 挂载result目录，让SHAP图片可以访问
app.mount("/result", StaticFiles(directory="result"), name="result")

# 设置模板
templates = Jinja2Templates(directory="templates")

# 创建预测器实例
predictor = MedicalRiskPredictor()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """主页面 - 只显示表单，不显示历史结果"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "default_age": DEFAULT_AGE,
        "default_polyps": DEFAULT_POLYPS,
        "default_long_diameter": DEFAULT_LONG_DIAMETER,
        "default_short_diameter": DEFAULT_SHORT_DIAMETER,
        "default_fundus": DEFAULT_FUNDUS,
        "model_loaded": predictor.model is not None,
        # 不传递result相关参数，确保页面显示为初始状态
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
    Base: int = Form(...),  # 改为int类型，接受1或2
    request_id: str = Form(...)
):
    """执行预测"""
    # 执行预测
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
    
    # 生成SHAP图
    print("正在生成SHAP图表...")
    waterfall_path, force_path = predictor.generate_shap_plots(age, polyps, long_diameter, short_diameter, Base, request_id)
    
    print(f"瀑布图路径: {waterfall_path}")
    print(f"力图路径: {force_path}")
    
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
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "model_path": MODEL_FILE_PATH
    }

if __name__ == "__main__":
    print("=" * 50)
    print("🏥 医疗风险预测系统启动中...")
    print("=" * 50)
    
    # 检查模型文件
    if not os.path.exists(MODEL_FILE_PATH):
        print(f"❌ 模型文件不存在: {MODEL_FILE_PATH}")
        print("请检查配置区域中的MODEL_FILE_PATH设置")
        print("=" * 50)
    else:
        print(f"✅ 模型文件路径: {MODEL_FILE_PATH}")
    
    # 启动服务
    print("🚀 启动Web服务...")
    print("📱 访问地址: http://localhost:8001")
    print("📚 API文档: http://localhost:8001/docs")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
