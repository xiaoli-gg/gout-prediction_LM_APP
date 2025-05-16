import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
#["U-Ba_ln", "U-Pb_ln", "B-Pb_ln", "BMI", "Sex", "Race/ethnicity", "Age", "Marital status", "Alcohol drink" ]
#df['Sex'] = df['Sex'].map({'Female': 0, 'Male': 1})
#df['Race/ethnicity'] = df['Race/ethnicity'].map({'Mexican American': 0, 'Non-Hispanic Black': 1, 'Non-Hispanic White': 2, 'Other Hispanic': 3, 'Other Race - Including Multi-Racial': 4})
#df['Age'] = df['Age'].map({'≥60': 0, '20–59': 1})
#df['Marital status'] = df['Marital status'].map({'Married/cohabiting': 0, 'Never married': 1, 'Widowed/divorced/separated': 2, 'missing': 3})
#df['Alcohol drink'] = df['Alcohol drink'].map({'Ever': 0, 'Never': 1, 'missing': 1})

# 加载保存的随机森林模型
model = joblib.load('RF_model.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "U-Ba_ln": {"type": "numerical", "min": -4.045554, "max": 7.347204, "default": 0.144627},
    "U-Pb_ln": {"type": "numerical", "min": -4.494239, "max": 3.914041, "default": -0.859132},
    "B-Pb_ln": {"type": "numerical", "min": -2.995732, "max": 3.516607, "default": 0.122218},
    "BMI": {"type": "numerical", "min": 13.18, "max": 82.90, "default": 28.10},
    "Sex": {"type": "categorical", "options": [0, 1]},
    "Race/ethnicity": {"type": "categorical", "options": [0, 1,2,3,4]},
    "Age": {"type": "categorical", "options": [0, 1]},
    "Marital status": {"type": "categorical", "options": [0, 1,2,3]},
    "Alcohol drink": {"type": "categorical", "options": [0, 1]},
}

# 设置标题
st.set_page_config(page_title='AI-assisted Gout Prediction Tool')
st.title("AI-Assisted Prediction of Gout")

# 左侧输入面板
st.sidebar.header("Enter the following feature values:")

feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.sidebar.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.sidebar.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 页面底部版权信息
st.sidebar.markdown("---")
st.sidebar.markdown("##### All rights reserved") 
st.sidebar.markdown("##### Contact: mengpanli163@163.com (Mengpan Li, Shanghai Jiao Tong University School of Medicine)")


# 转换为模型输入格式
features = np.array([feature_values])

# 预测与解释
if st.button("Predict"):
    # 构造输入
    features = np.array([feature_values])
    feature_df = pd.DataFrame(features, columns=feature_ranges.keys())

    # 预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[1] * 100  # 二分类中通常 [1] 为“阳性”类

    # 显示预测结果
    text = f"Predicted probability of Gout: {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(0.5, 0.5, text, fontsize=16, ha='center', va='center',
            fontname='Times New Roman', transform=ax.transAxes)
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 模拟预测概率（替换成你的实际结果）
    sample_prob = {
        'Class_0': predicted_proba[0],  # 未患病概率
        'Class_1': predicted_proba[1]   # 患病概率
    }

    # 设置图形大小
    plt.figure(figsize=(10, 3))

    # 创建水平条形图
    bars = plt.barh(['Not Gout', 'Gout'], 
                    [sample_prob['Class_0'], sample_prob['Class_1']], 
                    color=['#512b58', '#fe346e'])

    # 添加标题和轴标签
    plt.title("Prediction Probability for Patient", fontsize=20, fontweight='bold')
    plt.xlabel("Probability", fontsize=14, fontweight='bold')
    plt.ylabel("Classes", fontsize=14, fontweight='bold')
    
    # 添加数值标签
    for i, v in enumerate([sample_prob['Class_0'], sample_prob['Class_1']]):
        plt.text(
            v + 0.02,  # 添加一点右移以避免文字重叠
            i, 
            f"{v:.2f}", 
            va='center', 
            fontsize=14, 
            color='black', 
            fontweight='bold'
        )

    # 美化边框（隐藏上右边框）
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 限制 x 轴范围为 0 到 1，避免比例超界时图变形
    plt.xlim(0, 1)

    # 展示图像
    st.pyplot(plt)
    
    # SHAP 分析，注意“这里所用的模型是Bagclassifier，所以需要background数据，如果是tree模型的话就不需要”
    background = pd.read_csv("shap_background.csv")
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(feature_df)

    # 因为feature_df只有一个样本，所以a只能为0
    a = 0
    shap_fig = shap.plots.force(
        explainer.expected_value,  
        shap_values[a], 
        feature_df.iloc[a, :],
        matplotlib=True,
        show=False
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")


