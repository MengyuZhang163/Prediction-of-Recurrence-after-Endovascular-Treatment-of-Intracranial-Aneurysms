import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os
import xgboost as xgb

# ==========================================
# 1. 配置与模型加载
# ==========================================
st.set_page_config(
    page_title="血栓风险预测系统",
    page_icon="🧠",
    layout="wide"
)

# --- 请修改模型路径 ---
MODEL_PATH = 'XGB.pkl'


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"找不到模型文件，请检查路径: {MODEL_PATH}")
        return None
    model = load(MODEL_PATH)
    return model


model = load_model()

# ==========================================
# 2. 定义映射字典 (CRITICAL: 必须与训练时的编码一致)
# ==========================================
# 警告：下方的 0, 1, 2 只是示例。
# 你必须查看你训练前的 X_train 数据，确认每个文本对应的具体数字是多少！

mapping_dict = {
    "ThrombolysisAfterTirofiban": {"无 (No)": 0, "有 (Yes)": 1},

    # 假设你在训练时：LVIS=0, Enterprise=1, Solitaire=2... 请根据实际情况修改
    "StentType": {
        "LVIS": 0,
        "Enterprise": 1,
        "Solitaire": 2,
        "Flow Diverter": 3,
        "Other": 4
    },

    # 假设：Saccular=0, Irregular=1, Fusiform=2...
    "Morphology": {
        "Saccular (囊状)": 0,
        "Irregular (不规则)": 1,
        "Fusiform (梭形)": 2
    },

    "Rupture": {"未破裂 (Unruptured)": 0, "破裂 (Ruptured)": 1},

    # 假设：Simple=0, Balloon=1, Stent=2...
    "EmbolizationTechnique": {
        "Simple Coiling": 0,
        "Balloon-Assisted": 1,
        "Stent-Assisted": 2
    },

    # AngioAndTreatment 具体分类
    "AngioAndTreatment": {
        "Type A": 0,
        "Type B": 1,
        "Type C": 2
    },

    # 肝素化时机
    "HeparinTiming": {
        "Pre-operative (术前)": 0,
        "Intra-operative (术中)": 1,
        "Post-operative (术后)": 2
    }
}

# ==========================================
# 3. 侧边栏：输入参数
# ==========================================
st.sidebar.header("📝 患者临床参数输入")

# 1. ThrombolysisAfterTirofiban
input_thrombolysis = st.sidebar.selectbox(
    "替罗非班后溶栓 (ThrombolysisAfterTirofiban)",
    options=list(mapping_dict["ThrombolysisAfterTirofiban"].keys())
)

# 2. StentType
input_stent = st.sidebar.selectbox(
    "支架类型 (StentType)",
    options=list(mapping_dict["StentType"].keys())
)

# 3. Morphology
input_morphology = st.sidebar.selectbox(
    "动脉瘤形态 (Morphology)",
    options=list(mapping_dict["Morphology"].keys())
)

# 4. Rupture
input_rupture = st.sidebar.selectbox(
    "是否破裂 (Rupture)",
    options=list(mapping_dict["Rupture"].keys())
)

# 5. EmbolizationTechnique
input_technique = st.sidebar.selectbox(
    "栓塞技术 (EmbolizationTechnique)",
    options=list(mapping_dict["EmbolizationTechnique"].keys())
)

# 6. Width (数值型)
# 请根据训练数据的最大最小值调整 min_value, max_value
input_width = st.sidebar.number_input(
    "瘤体宽度 (Width, mm)",
    min_value=0.0, max_value=50.0, value=5.0, step=0.1
)

# 7. Neck (数值型)
input_neck = st.sidebar.number_input(
    "瘤颈宽度 (Neck, mm)",
    min_value=0.0, max_value=30.0, value=3.0, step=0.1
)

# 8. AngioAndTreatment
input_angio = st.sidebar.selectbox(
    "造影与治疗 (AngioAndTreatment)",
    options=list(mapping_dict["AngioAndTreatment"].keys())
)

# 9. HeparinTiming
input_heparin = st.sidebar.selectbox(
    "肝素时机 (HeparinTiming)",
    options=list(mapping_dict["HeparinTiming"].keys())
)

# ==========================================
# 4. 主界面：预测逻辑
# ==========================================
st.title("🧠 颅内动脉瘤血管内治疗 - 血栓风险预测")
st.markdown("---")

# 将输入转换为模型需要的 DataFrame 格式
# 注意：特征的顺序必须与模型训练时完全一致！
input_data = {
    'ThrombolysisAfterTirofiban': mapping_dict["ThrombolysisAfterTirofiban"][input_thrombolysis],
    'StentType': mapping_dict["StentType"][input_stent],
    'Morphology': mapping_dict["Morphology"][input_morphology],
    'Rupture': mapping_dict["Rupture"][input_rupture],
    'EmbolizationTechnique': mapping_dict["EmbolizationTechnique"][input_technique],
    'Width': input_width,
    'Neck': input_neck,
    'AngioAndTreatment': mapping_dict["AngioAndTreatment"][input_angio],
    'HeparinTiming': mapping_dict["HeparinTiming"][input_heparin]
}

df_input = pd.DataFrame([input_data])

# 展示当前输入
with st.expander("查看当前输入的模型特征值 (Encoded Data)"):
    st.dataframe(df_input)

# 预测按钮
if st.button("🚀 开始预测 (Predict)", type="primary"):
    if model:
        try:
            # 预测概率
            prob = model.predict_proba(df_input)[:, 1][0]

            # 设定阈值 (这里使用你在验证集中算出的最佳阈值，例如 0.4 或 0.5)
            # 你需要将下面这个数值改成你刚才代码跑出来的 best_threshold
            best_threshold = 0.5

            prediction_class = 1 if prob >= best_threshold else 0

            # 结果展示区
            col1, col2 = st.columns(2)

            with col1:
                st.metric(label="血栓发生概率 (Probability)", value=f"{prob:.2%}")

            with col2:
                if prediction_class == 1:
                    st.error(f"⚠️ 高风险 (High Risk) \n(> {best_threshold})")
                else:
                    st.success(f"✅ 低风险 (Low Risk) \n(< {best_threshold})")

            # 进度条可视化
            st.progress(prob, text="风险指数")

            # 解释性文字
            if prediction_class == 1:
                st.warning("提示：模型预测该患者发生血栓相关并发症的风险较高，建议密切监测或调整抗凝策略。")

        except Exception as e:
            st.error(f"预测过程中发生错误: {e}")
            st.write("请检查输入数据的格式是否与训练数据一致。")
    else:

        st.error("模型未加载，无法预测。")

