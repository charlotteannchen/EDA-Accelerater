# LLM × EDA 工具鏈自動化平台

## 項目概述

這是一個基於 LLM 的 EDA 工具鏈自動化平台，實現「系統規格 → 功能設計 → RTL → 電路設計」四階段的自動化流程，特別專注於類比電路設計與驗證。

## 核心架構

### 四階段設計流程
1. **System Spec** - 規格擷取與標準化
2. **Functional Design** - 功能分割與架構建議  
3. **Logic Design/RTL** - 數位電路設計
4. **Circuit Design** - 類比電路設計（SPICE/Verilog-A）

### 核心模組
- **Spec Intake & Normalizer** - 規格擷取與標準化
- **Architecture & Partition Suggester** - 功能分割建議
- **Analog Topology & Code Generator** - 類比拓樸與代碼生成
- **Testbench & Measurement Library** - 量測套件
- **Auto-Verify Orchestrator** - 自動驗證編排
- **Optimizer** - 參數優化
- **PDK/EDA Connectors** - 工具介面

## 快速開始

```bash
# 安裝依賴
pip install -r requirements.txt

# 啟動 Web Dashboard
python app.py

# 使用 CLI
python cli.py --spec examples/adc_spec.json
```

## 目錄結構

```
EDA-Accelerater/
├── README.md
├── requirements.txt
├── app.py                    # Web Dashboard
├── cli.py                    # CLI 介面
├── config/
│   ├── pdk_config.yaml      # PDK 配置
│   └── llm_config.yaml      # LLM 配置
├── src/
│   ├── core/                # 核心模組
│   ├── generators/          # 代碼生成器
│   ├── verifiers/           # 驗證器
│   ├── optimizers/          # 優化器
│   └── connectors/          # EDA 工具連接器
├── examples/                # 範例規格
├── templates/               # 模板文件
├── tests/                   # 測試
└── docs/                    # 文檔
```

## 特色功能

- 🤖 **LLM 驅動設計** - 自然語言規格轉換為電路設計
- 🔄 **自動化閉環** - 設計→驗證→優化→迭代
- 📊 **智能量測** - 自動生成測試平台與 KPI 儀表板
- 🎯 **參數優化** - 貝葉斯優化與 DOE 方法
- 🔒 **安全防護** - 審計日誌與權限管理
- 🌐 **多工具支援** - Spectre/HSPICE/NGSPICE

## 授權

MIT License
