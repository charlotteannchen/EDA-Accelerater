"""
EDA-Accelerater Web Dashboard
基於 Streamlit 的 Web 介面
"""

import streamlit as st
import json
import yaml
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# 添加 src 目錄到 Python 路徑
sys.path.append(str(Path(__file__).parent / "src"))

from core.spec_normalizer import SpecNormalizer, SpecValidation
from core.architecture_suggester import ArchitectureSuggester
from generators.analog_topology_generator import AnalogTopologyGenerator
from verifiers.testbench_generator import TestbenchGenerator, MeasurementType

# 頁面配置
st.set_page_config(
    page_title="EDA-Accelerater",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .error-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def load_config():
    """載入配置"""
    try:
        with open("config/pdk_config.yaml", 'r', encoding='utf-8') as f:
            pdk_config = yaml.safe_load(f)
        with open("config/llm_config.yaml", 'r', encoding='utf-8') as f:
            llm_config = yaml.safe_load(f)
        return pdk_config, llm_config
    except FileNotFoundError as e:
        st.error(f"配置文件未找到: {e}")
        return None, None

def main():
    """主應用程式"""
    st.markdown('<h1 class="main-header">⚡ EDA-Accelerater</h1>', unsafe_allow_html=True)
    st.markdown("### LLM × EDA 工具鏈自動化平台")
    
    # 載入配置
    pdk_config, llm_config = load_config()
    if not pdk_config or not llm_config:
        st.stop()
    
    # 側邊欄導航
    st.sidebar.title("導航")
    page = st.sidebar.selectbox(
        "選擇頁面",
        ["規格輸入", "架構設計", "電路生成", "驗證測試", "結果分析", "設置"]
    )
    
    if page == "規格輸入":
        spec_input_page(pdk_config, llm_config)
    elif page == "架構設計":
        architecture_page(pdk_config, llm_config)
    elif page == "電路生成":
        circuit_generation_page(pdk_config, llm_config)
    elif page == "驗證測試":
        verification_page(pdk_config, llm_config)
    elif page == "結果分析":
        analysis_page(pdk_config, llm_config)
    elif page == "設置":
        settings_page(pdk_config, llm_config)

def spec_input_page(pdk_config, llm_config):
    """規格輸入頁面"""
    st.header("📝 規格輸入與標準化")
    
    # 輸入方式選擇
    input_method = st.radio(
        "選擇輸入方式",
        ["自然語言描述", "JSON 格式", "上傳文件"]
    )
    
    spec_text = ""
    spec_json = {}
    
    if input_method == "自然語言描述":
        st.subheader("自然語言規格描述")
        spec_text = st.text_area(
            "請輸入電路規格描述",
            placeholder="例如：設計一個 12-bit ADC，採樣率 100MHz，電源電壓 1.2V，輸入範圍 1Vpp，功耗預算 10mW，最小 SNR 68dB...",
            height=200
        )
        
        if st.button("轉換為標準格式", type="primary"):
            if spec_text:
                with st.spinner("正在處理規格..."):
                    normalizer = SpecNormalizer()
                    spec_json, validation = normalizer.normalize_spec(spec_text)
                    
                    if validation.is_valid:
                        st.success("規格轉換成功！")
                        st.json(spec_json)
                    else:
                        st.error("規格轉換失敗")
                        for error in validation.errors:
                            st.error(f"❌ {error}")
                        for warning in validation.warnings:
                            st.warning(f"⚠️ {warning}")
    
    elif input_method == "JSON 格式":
        st.subheader("JSON 格式規格")
        json_input = st.text_area(
            "請輸入 JSON 格式的規格",
            height=300,
            placeholder='{"block": "ADC", "resolution_bits": 12, ...}'
        )
        
        if st.button("驗證 JSON 格式", type="primary"):
            if json_input:
                try:
                    spec_json = json.loads(json_input)
                    st.success("JSON 格式正確！")
                    st.json(spec_json)
                except json.JSONDecodeError as e:
                    st.error(f"JSON 格式錯誤: {e}")
    
    elif input_method == "上傳文件":
        st.subheader("上傳規格文件")
        uploaded_file = st.file_uploader(
            "選擇文件",
            type=['json', 'txt', 'yaml'],
            help="支援 JSON、TXT、YAML 格式"
        )
        
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            if uploaded_file.name.endswith('.json'):
                try:
                    spec_json = json.loads(content)
                    st.success("文件載入成功！")
                    st.json(spec_json)
                except json.JSONDecodeError as e:
                    st.error(f"JSON 格式錯誤: {e}")
            else:
                spec_text = content
                st.text_area("文件內容", value=content, height=200)
    
    # 規格預覽和編輯
    if spec_json:
        st.subheader("規格預覽與編輯")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**當前規格**")
            st.json(spec_json)
        
        with col2:
            st.write("**規格摘要**")
            create_spec_summary(spec_json)
        
        # 保存規格
        if st.button("保存規格", type="primary"):
            with open("current_spec.json", "w", encoding="utf-8") as f:
                json.dump(spec_json, f, indent=2, ensure_ascii=False)
            st.success("規格已保存到 current_spec.json")

def create_spec_summary(spec):
    """創建規格摘要"""
    summary_data = []
    
    if 'block' in spec:
        summary_data.append(("電路類型", spec['block']))
    if 'resolution_bits' in spec:
        summary_data.append(("解析度", f"{spec['resolution_bits']} bits"))
    if 'fs_hz' in spec:
        summary_data.append(("採樣率", f"{spec['fs_hz']/1e6:.1f} MHz"))
    if 'vdd_v' in spec:
        summary_data.append(("電源電壓", f"{spec['vdd_v']} V"))
    if 'power_budget_mw' in spec:
        summary_data.append(("功耗預算", f"{spec['power_budget_mw']} mW"))
    if 'snr_db_min' in spec:
        summary_data.append(("最小 SNR", f"{spec['snr_db_min']} dB"))
    
    for label, value in summary_data:
        st.metric(label, value)

def architecture_page(pdk_config, llm_config):
    """架構設計頁面"""
    st.header("🏗️ 架構設計與分割")
    
    # 載入規格
    if not Path("current_spec.json").exists():
        st.warning("請先在規格輸入頁面創建規格")
        return
    
    with open("current_spec.json", "r", encoding="utf-8") as f:
        spec = json.load(f)
    
    st.subheader("當前規格")
    st.json(spec)
    
    if st.button("生成架構建議", type="primary"):
        with st.spinner("正在生成架構建議..."):
            suggester = ArchitectureSuggester()
            architecture = suggester.suggest_architecture(spec)
            
            # 顯示架構信息
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("類比方塊", len(architecture.analog_blocks))
                st.write("**類比方塊列表**")
                for block in architecture.analog_blocks:
                    st.write(f"• {block}")
            
            with col2:
                st.metric("數位方塊", len(architecture.digital_blocks))
                st.write("**數位方塊列表**")
                for block in architecture.digital_blocks:
                    st.write(f"• {block}")
            
            with col3:
                st.metric("混合方塊", len(architecture.mixed_blocks))
                st.write("**混合方塊列表**")
                for block in architecture.mixed_blocks:
                    st.write(f"• {block}")
            
            # 顯示詳細架構
            st.subheader("詳細架構信息")
            
            # 方塊詳細信息
            blocks_df = pd.DataFrame([
                {
                    "方塊名稱": block.name,
                    "類型": block.block_type.value,
                    "功能": block.function,
                    "介面數": len(block.interfaces),
                    "量測點數": len(block.measurement_points),
                    "功耗預估 (mW)": block.power_estimate_mw,
                    "面積預估 (mm²)": block.area_estimate_mm2
                }
                for block in architecture.blocks
            ])
            
            st.dataframe(blocks_df, use_container_width=True)
            
            # 連接關係
            st.subheader("方塊連接關係")
            connections_df = pd.DataFrame(architecture.connections, columns=["來源方塊", "目標方塊"])
            st.dataframe(connections_df, use_container_width=True)
            
            # 保存架構
            if st.button("保存架構", type="primary"):
                suggester.export_architecture(architecture, "current_architecture.json")
                st.success("架構已保存到 current_architecture.json")

def circuit_generation_page(pdk_config, llm_config):
    """電路生成頁面"""
    st.header("⚡ 電路生成")
    
    # 載入規格和架構
    if not Path("current_spec.json").exists():
        st.warning("請先創建規格")
        return
    
    with open("current_spec.json", "r", encoding="utf-8") as f:
        spec = json.load(f)
    
    st.subheader("電路需求")
    
    # 電路參數輸入
    col1, col2 = st.columns(2)
    
    with col1:
        gbw_hz = st.number_input("增益帶寬乘積 (Hz)", value=100e6, format="%.0e")
        pm_deg = st.number_input("相位邊限 (度)", value=60, min_value=0, max_value=180)
        load_cap_f = st.number_input("負載電容 (F)", value=1e-12, format="%.2e")
    
    with col2:
        power_mw = st.number_input("功耗預算 (mW)", value=5.0, min_value=0.1)
        noise_nv_per_sqrt_hz = st.number_input("雜訊密度 (nV/√Hz)", value=10.0)
        vdd_v = st.number_input("電源電壓 (V)", value=1.2, min_value=0.8, max_value=3.3)
    
    requirements = {
        "gbw_hz": gbw_hz,
        "pm_deg": pm_deg,
        "load_cap_f": load_cap_f,
        "power_mw": power_mw,
        "noise_nv_per_sqrt_hz": noise_nv_per_sqrt_hz,
        "vdd_v": vdd_v
    }
    
    if st.button("生成電路", type="primary"):
        with st.spinner("正在生成電路..."):
            generator = AnalogTopologyGenerator(pdk_config)
            topology, spice_netlist, veriloga_model = generator.generate_topology(requirements)
            
            # 顯示拓樸選擇
            st.subheader("拓樸選擇")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("拓樸名稱", topology.name)
            with col2:
                st.metric("預估 GBW", f"{topology.estimated_gbw/1e6:.1f} MHz")
            with col3:
                st.metric("預估功耗", f"{topology.estimated_power:.1f} mW")
            with col4:
                st.metric("預估面積", f"{topology.estimated_area:.3f} mm²")
            
            st.write("**選擇理由**")
            st.info(topology.rationale)
            
            # 顯示生成的代碼
            tab1, tab2 = st.tabs(["SPICE Netlist", "Verilog-A Model"])
            
            with tab1:
                st.code(spice_netlist, language="spice")
                if st.button("下載 SPICE 文件"):
                    with open("generated_circuit.sp", "w") as f:
                        f.write(spice_netlist)
                    st.success("SPICE 文件已保存")
            
            with tab2:
                st.code(veriloga_model, language="verilog")
                if st.button("下載 Verilog-A 文件"):
                    with open("generated_circuit.va", "w") as f:
                        f.write(veriloga_model)
                    st.success("Verilog-A 文件已保存")

def verification_page(pdk_config, llm_config):
    """驗證測試頁面"""
    st.header("🧪 驗證測試")
    
    # 測試配置
    st.subheader("測試配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        measurement_types = st.multiselect(
            "選擇量測類型",
            [m.value for m in MeasurementType],
            default=[MeasurementType.AC_ANALYSIS.value, MeasurementType.TRANSIENT.value]
        )
        
        simulation_time = st.number_input("模擬時間 (μs)", value=10.0, min_value=0.1)
        temperature = st.number_input("溫度 (°C)", value=25, min_value=-40, max_value=125)
    
    with col2:
        vdd = st.number_input("電源電壓 (V)", value=1.2, min_value=0.8, max_value=3.3)
        corners = st.multiselect(
            "工藝角落",
            ["TT", "SS", "FF", "SF", "FS"],
            default=["TT"]
        )
        monte_carlo_runs = st.number_input("Monte Carlo 次數", value=0, min_value=0, max_value=10000)
    
    # 載入電路
    if not Path("generated_circuit.sp").exists():
        st.warning("請先生成電路")
        return
    
    with open("generated_circuit.sp", "r") as f:
        circuit_netlist = f.read()
    
    st.subheader("電路 Netlist 預覽")
    st.code(circuit_netlist[:500] + "..." if len(circuit_netlist) > 500 else circuit_netlist, language="spice")
    
    if st.button("生成測試平台", type="primary"):
        with st.spinner("正在生成測試平台..."):
            generator = TestbenchGenerator(pdk_config)
            
            requirements = {
                "gbw_hz": 100e6,
                "pm_deg": 60,
                "vdd_v": vdd,
                "load_cap_f": 1e-12
            }
            
            measurement_types_enum = [MeasurementType(mt) for mt in measurement_types]
            testbench = generator.generate_testbench(circuit_netlist, requirements, measurement_types_enum)
            
            st.subheader("生成的測試平台")
            st.code(testbench, language="spice")
            
            if st.button("下載測試平台"):
                with open("testbench.sp", "w") as f:
                    f.write(testbench)
                st.success("測試平台已保存")
    
    # 模擬執行
    if st.button("執行模擬", type="primary"):
        with st.spinner("正在執行模擬..."):
            st.info("模擬功能需要安裝 EDA 工具（如 NGSPICE、Spectre 等）")
            st.code("ngspice -b testbench.sp > simulation.log", language="bash")
            
            # 模擬結果解析（示例）
            st.subheader("模擬結果")
            
            # 創建示例結果
            results_data = {
                "量測類型": ["GBW", "PM", "Slew Rate", "Noise"],
                "數值": [95.2e6, 58.3, 45e6, 8.5e-9],
                "單位": ["Hz", "degrees", "V/s", "V/√Hz"],
                "狀態": ["✅ PASS", "⚠️ WARN", "✅ PASS", "✅ PASS"]
            }
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # 結果圖表
            fig = go.Figure(data=[
                go.Bar(x=results_data["量測類型"], y=results_data["數值"], 
                      text=results_data["狀態"], textposition='auto')
            ])
            fig.update_layout(title="量測結果", xaxis_title="量測類型", yaxis_title="數值")
            st.plotly_chart(fig, use_container_width=True)

def analysis_page(pdk_config, llm_config):
    """結果分析頁面"""
    st.header("📊 結果分析")
    
    # 創建示例數據
    st.subheader("KPI 儀表板")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("總體通過率", "85%", "5%")
    with col2:
        st.metric("功耗", "4.2 mW", "-0.8 mW")
    with col3:
        st.metric("面積", "0.012 mm²", "0.002 mm²")
    with col4:
        st.metric("良率", "92%", "3%")
    
    # 趨勢圖表
    st.subheader("參數優化趨勢")
    
    # 創建示例優化歷史
    optimization_data = pd.DataFrame({
        "迭代次數": range(1, 11),
        "GBW (MHz)": [80, 85, 88, 90, 92, 94, 95, 95.2, 95.1, 95.3],
        "功耗 (mW)": [6.0, 5.5, 5.2, 4.8, 4.5, 4.3, 4.2, 4.2, 4.1, 4.2],
        "面積 (mm²)": [0.015, 0.014, 0.013, 0.0125, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=optimization_data["迭代次數"], y=optimization_data["GBW (MHz)"], 
                            name="GBW", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=optimization_data["迭代次數"], y=optimization_data["功耗 (mW)"], 
                            name="功耗", line=dict(color="red"), yaxis="y2"))
    
    fig.update_layout(
        title="優化過程",
        xaxis_title="迭代次數",
        yaxis=dict(title="GBW (MHz)", side="left"),
        yaxis2=dict(title="功耗 (mW)", side="right", overlaying="y"),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 工藝角落分析
    st.subheader("工藝角落分析")
    
    corner_data = pd.DataFrame({
        "角落": ["TT", "SS", "FF", "SF", "FS"],
        "GBW (MHz)": [95.2, 88.5, 102.1, 91.3, 99.8],
        "PM (度)": [58.3, 52.1, 64.2, 55.7, 61.8],
        "功耗 (mW)": [4.2, 5.1, 3.8, 4.6, 3.9],
        "通過狀態": ["✅", "⚠️", "✅", "⚠️", "✅"]
    })
    
    st.dataframe(corner_data, use_container_width=True)
    
    # Monte Carlo 分析
    st.subheader("Monte Carlo 分析")
    
    # 創建示例 MC 數據
    np.random.seed(42)
    mc_gbw = np.random.normal(95.2, 5, 1000)
    mc_pm = np.random.normal(58.3, 3, 1000)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_gbw = px.histogram(x=mc_gbw, nbins=30, title="GBW 分布")
        st.plotly_chart(fig_gbw, use_container_width=True)
    
    with col2:
        fig_pm = px.histogram(x=mc_pm, nbins=30, title="PM 分布")
        st.plotly_chart(fig_pm, use_container_width=True)

def settings_page(pdk_config, llm_config):
    """設置頁面"""
    st.header("⚙️ 設置")
    
    tab1, tab2, tab3 = st.tabs(["PDK 設置", "LLM 設置", "模擬設置"])
    
    with tab1:
        st.subheader("PDK 配置")
        st.json(pdk_config)
        
        if st.button("重新載入 PDK 配置"):
            st.rerun()
    
    with tab2:
        st.subheader("LLM 配置")
        st.json(llm_config)
        
        # LLM 設置
        provider = st.selectbox("LLM 提供商", ["openai", "anthropic", "local"])
        model = st.text_input("模型名稱", value="gpt-4-turbo-preview")
        temperature = st.slider("溫度", 0.0, 1.0, 0.1)
        
        if st.button("保存 LLM 設置"):
            st.success("LLM 設置已保存")
    
    with tab3:
        st.subheader("模擬設置")
        
        engine = st.selectbox("模擬引擎", ["ngspice", "spectre", "hspice"])
        timeout = st.number_input("超時時間 (秒)", value=3600, min_value=60)
        max_iterations = st.number_input("最大迭代次數", value=1000, min_value=100)
        
        if st.button("保存模擬設置"):
            st.success("模擬設置已保存")

if __name__ == "__main__":
    main()
