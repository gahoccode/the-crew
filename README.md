# CrewAI Data Analyst

An intelligent financial analysis system powered by CrewAI and vnstock, designed to provide comprehensive stock analysis with interactive visualizations.

## Features

- **CrewAI Integration**: Intelligent agent-based analysis
- **vnstock Support**: Vietnamese stock market data
- **Interactive Visualizations**: Rich charts and graphs
- **Streamlit Interface**: User-friendly web interface
- **Financial Health Analysis**: Key ratios and metrics
- **Real-time Data**: Latest financial statements and market data

## Installation

### Using UV (Recommended)
```bash
# Install dependencies
uv pip install -r requirements.txt

# Or using pyproject.toml
uv pip install -e .
```

### Using pip
```bash
pip install -r requirements.txt
```

## Usage

1. **Set up environment variables** (optional):
   ```bash
   # Create .env file for API keys if needed
   echo "OPENAI_API_KEY=your_key_here" > .env
   ```

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```

3. **Access the web interface**:
   - Open your browser to `http://localhost:8501`
   - Enter a Vietnamese stock symbol (e.g., VIC, VNM, REE)
   - Select analysis type
   - Click "Run Analysis"

## Stock Symbols

Popular Vietnamese stock symbols you can analyze:
- **VIC**: Vinhomes
- **VNM**: Vinamilk
- **REE**: Refrigeration Electrical Engineering
- **FPT**: FPT Corporation
- **HPG**: Hoa Phat Group
- **MWG**: Mobile World

## Architecture

### Components

1. **CrewAI Agent**: `Senior Financial Data Analyst`
   - Role: Comprehensive financial analysis
   - Tools: Data fetching, visualization, financial analysis
   - Expertise: Vietnamese stock market, fundamental analysis

2. **Custom Tools**:
   - `fetch_stock_data`: Retrieves financial statements and ratios
   - `create_visualization`: Creates interactive charts
   - `analyze_financial_health`: Analyzes key financial metrics

3. **Streamlit Interface**:
   - Real-time data display
   - Interactive visualizations
   - User-friendly controls

### Data Sources

- **Financial Statements**: Income statement, balance sheet, cash flow
- **Key Ratios**: Liquidity, profitability, leverage metrics
- **Dividend History**: Cash dividends and payment dates
- **Market Data**: Historical prices and trading volumes

## Example Analysis

The system provides:
- **Financial Health Score**: Based on key ratios
- **Revenue Trends**: Historical revenue patterns
- **Profitability Analysis**: ROA, ROE, net margins
- **Liquidity Assessment**: Current ratio, quick ratio
- **Leverage Analysis**: Debt-to-equity, interest coverage
- **Dividend Analysis**: Historical dividend payments

## Troubleshooting

### Common Issues

1. **vnstock data not loading**:
   - Check internet connection
   - Verify stock symbol exists
   - Try different data sources (VCI, TCBS)

2. **Streamlit not starting**:
   - Ensure all dependencies are installed
   - Check Python version (>=3.8 required)
   - Verify port 8501 is available

3. **Visualization errors**:
   - Check data availability for selected symbol
   - Ensure required columns exist in datasets

## Development

### Project Structure
```
crewai-data-analyst/
├── app.py                 # Main Streamlit application
├── pyproject.toml         # UV project configuration
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .env                  # Environment variables (optional)
```

### Adding New Features

1. **New Tools**: Add custom tools using `@tool` decorator
2. **New Agents**: Create specialized agents for different analysis types
3. **New Visualizations**: Extend `create_visualization` tool
4. **New Data Sources**: Integrate additional vnstock endpoints

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.
