import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os


# ✅ 피클 파일 다운로드 (최초 실행 시 한 번만 다운로드됨)
file_id = '1t3CzygdUX2uajH_Cwkx5RUiahjhNQX15'  # 파일 ID만 추출
output = 'model_0310.pkl'

# 이미 파일이 존재하면 다운로드 생략
if not os.path.exists(output):
    gdown.download(id=file_id, output=output, quiet=False)
else:
    print(f"✅ '{output}' 파일이 이미 존재합니다. 다운로드 생략.")


# ✅ 모델 로드 함수
@st.cache_resource
def load_model():
    model_path = "model_0310.pkl"  # 다운로드된 모델 경로
    
    # 파일 존재 여부 확인
    if not os.path.exists(model_path):
        st.error(f"🚨 모델 파일이 존재하지 않습니다: {model_path}")
        return None

    try:
        with open(model_path, "rb") as f:
            model = joblib.load(f)

        if isinstance(model, dict):  # 모델이 dict 형태일 경우
            model = model.get("model", None)

        return model
    except Exception as e:
        st.error(f"⚠️ 모델 로드 실패: {e}")
        return None

### Main 공간 ###
st.title("머신러닝 예측")

model = load_model()  # 모델 로드

# 모델이 정상적으로 로드되었는지 확인
if model is None:
    st.error("🚨 모델을 불러오지 못했습니다. 파일을 확인하세요.")
    st.stop()  # 실행 중단

# ✅ 사용자 입력 폼
st.write("## 🔧 입력 변수 설정")
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        battery_current = st.number_input("Battery Current A", value=10.0, step=0.1)
        battery_power = st.number_input("Battery Power (W)", value=5000.0, step=10.0)

    with col2:
        battery_capacity = st.number_input("Battery Capacity (Wh)_shift", value=40000.0, step=10.0)
        hvac_power = st.number_input("HVAC Power Consumption", value=2.0, step=0.1)

    submit_button = st.form_submit_button("예측 시작")

# ✅ 예측 실행
if submit_button:
    try:
        input_data = np.array([[battery_current, battery_power, hvac_power, battery_capacity]])
        
        # 학습 당시 사용한 컬럼명과 동일하게 맞추기
        features_df = pd.DataFrame(input_data, columns=[
            "Battery Current A",  # 기존 "Battery Current [A]" -> "Battery Current A"
            "Battery Power (W)",
            "HVAC Power Consumption",
            "Battery Capacity (Wh)_shift",  # 기존 "Battery Capacity (Wh)" -> "Battery Capacity (Wh)_shift"
        ])
        
        # 예측
        prediction = model.predict(features_df)
        
        st.success(f"🔮 예측된 SoC_diff: {prediction[0]:.4f}")

        # 예측 결과에 대한 추가 정보 (시각화나 설명)
        st.write("### 예측된 결과")
        st.write(f"**배터리 전류 (A):** {battery_current} A")
        st.write(f"**배터리 전력 (W):** {battery_power} W")
        st.write(f"**HVAC 전력 소비량:** {hvac_power} kW")
        st.write(f"**배터리 용량 (Wh):** {battery_capacity} Wh")

        # 예측 결과를 보다 직관적으로 보여주기 위해 결과 시각화 추가 가능
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.bar(["Predicted SoC_diff"], [prediction[0]], color="blue")
        ax.set_ylabel("SoC_diff")
        st.pyplot(fig)

    except AttributeError as e:
        st.error(f"⚠️ 모델 예측 중 오류 발생: {e}")


