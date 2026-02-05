# 📈 PSX Portfolio Manager

A comprehensive stock portfolio management system for Pakistan Stock Exchange (PSX) with AI-powered price forecasting, technical analysis, and multi-user support.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

### 🔐 Authentication & Multi-User Support
- User registration and login system
- Secure password hashing (SHA-256)
- Isolated portfolios per user
- Session management

### 📊 Portfolio Management
- **Transaction Tracking**: Record BUY/SELL transactions with fees
- **Holdings Summary**: Real-time portfolio value with P&L calculations
- **Allocation Analysis**: Visual breakdown of portfolio allocation
- **Bulk PDF Import**: Upload up to 15 broker contract PDFs at once
  - Supports JSBL, AKD, and other broker formats
  - Auto-extracts symbol, quantity, price, date, fees

### 📈 Technical Analysis
- **Interactive Charts**: Candlestick/Line charts with Plotly
- **Technical Indicators**:
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - MACD (Moving Average Convergence Divergence)
- **Advanced Analysis**:
  - Dynamic Support/Resistance Zones (Pivot clustering algorithm)
  - Pattern Recognition (Head & Shoulders, Double Top/Bottom, Trend patterns)

### 🔮 AI-Powered Price Forecasting
- **Amazon Chronos-2**: Time-series foundation model
  - Max Context: 8,192 days
  - Max Prediction: 1,024 days
- **Datadog Toto**: Open time-series model
  - Max Context: 4,096 days
  - Max Prediction: 1,024 days
- **Features**:
  - Probabilistic forecasts with confidence intervals (50% & 90%)
  - GPU acceleration (CUDA support)
  - Model selection in UI

### 💰 Dividends & P&L Tracking
- Record cash and stock dividends
- Automatic realized P&L calculation on sales
- Historical dividend and P&L reports

### 🔍 Stock Explorer
- Explore any PSX stock (not just portfolio holdings)
- Full technical analysis suite
- AI forecasting capabilities

### 📡 Market Data
- Real-time prices via Yahoo Finance
- Web scraping for KSE100/KMI30 index data
- Local price caching for performance

## 🚀 Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional, for faster AI forecasting)
- Chrome browser (for web scraping)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/psx-portfolio-manager.git
cd psx-portfolio-manager
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
cd dashboard/src
pip install -r requirements.txt
```

### Step 4: Install PyTorch with CUDA (Optional - for GPU acceleration)
```bash
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For latest GPUs (RTX 40/50 series) - use nightly build
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Step 5: Install AI Forecasting Models
```bash
# Amazon Chronos-2
pip install "chronos-forecasting>=2.0"

# Datadog Toto
pip install toto-ts
```

## 🎮 Usage

### Start the Application
```bash
cd dashboard/src
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### First Time Setup
1. Create an account on the login page
2. Login with your credentials
3. Add transactions manually or import broker PDFs
4. Explore your portfolio!

## 📁 Project Structure

```
psx-portfolio-manager/
├── dashboard/
│   ├── data/
│   │   └── portfolio.db          # SQLite database
│   └── src/
│       ├── app.py                # Main Streamlit application
│       ├── auth.py               # Authentication module
│       ├── database.py           # Database operations
│       ├── portfolio_manager.py  # Portfolio calculations
│       ├── market_data.py        # Yahoo Finance integration
│       ├── scraper.py            # Web scraping (Sarmaaya.pk)
│       ├── pdf_parser.py         # Broker PDF parsing
│       ├── advanced_indicators.py # Technical analysis
│       ├── forecasting.py        # AI forecasting (Chronos/Toto)
│       ├── calculations.py       # Allocation calculations
│       ├── data_loader.py        # Data I/O utilities
│       ├── sector_data.py        # Sector information
│       └── requirements.txt      # Python dependencies
└── README.md
```

## 🛠️ Configuration

### Database
The application uses SQLite by default. The database file is created automatically at:
```
dashboard/data/portfolio.db
```

### Supported Broker PDF Formats
- JSBL (JS Bank Limited)
- AKD Securities
- Other brokers with similar contract formats

### Yahoo Finance Symbols
PSX stocks use the `.KA` suffix on Yahoo Finance:
- `ENGRO` → `ENGRO.KA`
- `TRG` → `TRG.KA`
- `HBL` → `HBL.KA`

## 📊 Technical Indicators Explained

| Indicator | Description | Interpretation |
|-----------|-------------|----------------|
| **RSI** | Measures momentum (0-100) | >70 overbought, <30 oversold |
| **Bollinger Bands** | Volatility bands around MA | Price near bands = potential reversal |
| **MACD** | Trend-following momentum | Line crossing signal = trend change |
| **Support/Resistance** | Price levels with historical significance | Potential bounce/breakout points |

## 🔮 AI Forecasting Models

### Amazon Chronos-2
- **Type**: Time-series foundation model
- **Training**: Pre-trained on diverse time-series data
- **Strengths**: General-purpose, handles various patterns

### Datadog Toto
- **Type**: Open time-series model
- **Training**: Optimized for forecasting tasks
- **Strengths**: Fast inference, good for operational metrics

### GPU Requirements
| Model | VRAM Required | Recommended GPU |
|-------|---------------|-----------------|
| Chronos-2 | ~4GB | RTX 3060+ |
| Toto | ~2GB | RTX 3050+ |

## 🔧 Troubleshooting

### Common Issues

**1. "No module named 'chronos'"**
```bash
pip install "chronos-forecasting>=2.0"
```

**2. "CUDA not available"**
```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**3. "PDF parsing failed"**
```bash
pip install pdfplumber PyPDF2
```

**4. "Chrome driver error"**
```bash
pip install webdriver-manager
```

## 🗺️ Roadmap

- [ ] Migration to FastAPI + React/Next.js
- [ ] Real-time price streaming via WebSocket
- [ ] Portfolio alerts and notifications
- [ ] Multi-currency support
- [ ] Options/Futures tracking
- [ ] Mobile responsive design
- [ ] API for third-party integrations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Amazon Chronos](https://github.com/amazon-science/chronos-forecasting) - Time-series forecasting
- [Datadog Toto](https://github.com/DataDog/toto) - Open time-series model
- [Yahoo Finance](https://finance.yahoo.com/) - Market data
- [Streamlit](https://streamlit.io/) - Web application framework
- [Plotly](https://plotly.com/) - Interactive visualizations

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Disclaimer**: This software is for educational and informational purposes only. It is not financial advice. Always do your own research before making investment decisions.
