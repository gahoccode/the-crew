"""
CrewAI Data Analyst Agent with Code Interpreter
This application creates a data analyst agent that can fetch financial data 
using vnstock and visualize it using various plotting libraries.
"""

import os
import warnings
from typing import Dict, Any
from dotenv import load_dotenv

import pandas as pd
from vnstock import Vnstock

from crewai import Agent, Task, Crew, Process
from crewai_tools import CodeInterpreterTool
from crewai.llm import LLM

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

class FinancialDataAnalyst:
    """
    A CrewAI-based financial data analyst that can fetch Vietnamese stock data
    and perform analysis with visualization capabilities.
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the Financial Data Analyst.
        
        Args:
            openai_api_key: OpenAI API key for the LLM
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Initialize tools
        self.code_interpreter = CodeInterpreterTool()
        
        # Initialize LLM
        self.llm = LLM(
            model="gpt-4o-mini",
            api_key=self.openai_api_key
        )
        
        # Create the data analyst agent
        self.data_analyst_agent = self._create_data_analyst_agent()
    
    def _create_data_analyst_agent(self) -> Agent:
        """
        Create a data analyst agent with code interpreter capabilities.
        
        Returns:
            Agent: The configured data analyst agent
        """
        return Agent(
            role="Senior Financial Data Analyst",
            goal="Analyze Vietnamese stock market data and create insightful visualizations to help investors make informed decisions",
            backstory="""You are a seasoned financial data analyst with expertise in Vietnamese stock market analysis.
            You have deep knowledge of financial metrics, market trends, and data visualization techniques.
            You can fetch real-time financial data using vnstock library and create compelling visualizations
            using matplotlib, seaborn, and plotly. You always provide actionable insights based on your analysis.""",
            tools=[self.code_interpreter],
            llm=self.llm,
            allow_code_execution=True,
            verbose=True,
            max_iter=3,
            memory=True
        )
    
    def _process_ratio_dataframe(self, ratios_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the multi-index columns in the financial ratios DataFrame.
        
        Args:
            ratios_df: Raw ratios DataFrame with multi-index columns
            
        Returns:
            pd.DataFrame: Processed DataFrame with flattened column names
        """
        if ratios_df.empty:
            return ratios_df
            
        try:
            # Create a copy to avoid modifying the original
            processed_df = ratios_df.copy()
            
            # Flatten multi-index columns by combining level 0 and level 1
            if isinstance(processed_df.columns, pd.MultiIndex):
                # Create new column names by combining the multi-index levels
                new_columns = []
                for col in processed_df.columns:
                    if col[0] == 'Meta':
                        # Keep meta columns as is
                        new_columns.append(col[1])
                    else:
                        # Combine category and metric name
                        category_mapping = {
                            'Chỉ tiêu cơ cấu nguồn vốn': 'Capital_Structure',
                            'Chỉ tiêu hiệu quả hoạt động': 'Efficiency',
                            'Chỉ tiêu khả năng sinh lợi': 'Profitability',
                            'Chỉ tiêu thanh khoản': 'Liquidity',
                            'Chỉ tiêu định giá': 'Valuation'
                        }
                        
                        metric_mapping = {
                            # Capital Structure
                            '(Vay NH+DH)/VCSH': 'Debt_to_Equity',
                            'Nợ/VCSH': 'Total_Debt_to_Equity',
                            'TSCĐ / Vốn CSH': 'Fixed_Assets_to_Equity',
                            'Vốn CSH/Vốn điều lệ': 'Equity_to_Charter_Capital',
                            
                            # Efficiency
                            'Vòng quay tài sản': 'Asset_Turnover',
                            'Vòng quay TSCĐ': 'Fixed_Asset_Turnover',
                            'Số ngày thu tiền bình quân': 'Days_Sales_Outstanding',
                            'Số ngày tồn kho bình quân': 'Days_Inventory_Outstanding',
                            'Số ngày thanh toán bình quân': 'Days_Payable_Outstanding',
                            'Chu kỳ tiền': 'Cash_Cycle',
                            'Vòng quay hàng tồn kho': 'Inventory_Turnover',
                            
                            # Profitability
                            'Biên EBIT (%)': 'EBIT_Margin_Pct',
                            'Biên lợi nhuận gộp (%)': 'Gross_Margin_Pct',
                            'Biên lợi nhuận ròng (%)': 'Net_Margin_Pct',
                            'ROE (%)': 'ROE_Pct',
                            'ROIC (%)': 'ROIC_Pct',
                            'ROA (%)': 'ROA_Pct',
                            'EBITDA (Tỷ đồng)': 'EBITDA_Billion_VND',
                            'EBIT (Tỷ đồng)': 'EBIT_Billion_VND',
                            'Tỷ suất cổ tức (%)': 'Dividend_Yield_Pct',
                            
                            # Liquidity
                            'Chỉ số thanh toán hiện thời': 'Current_Ratio',
                            'Chỉ số thanh toán tiền mặt': 'Cash_Ratio',
                            'Chỉ số thanh toán nhanh': 'Quick_Ratio',
                            'Khả năng chi trả lãi vay': 'Interest_Coverage_Ratio',
                            'Đòn bẩy tài chính': 'Financial_Leverage',
                            
                            # Valuation
                            'Vốn hóa (Tỷ đồng)': 'Market_Cap_Billion_VND',
                            'Số CP lưu hành (Triệu CP)': 'Shares_Outstanding_Million',
                            'P/E': 'PE_Ratio',
                            'P/B': 'PB_Ratio',
                            'P/S': 'PS_Ratio',
                            'P/Cash Flow': 'P_CashFlow_Ratio',
                            'EPS (VND)': 'EPS_VND',
                            'BVPS (VND)': 'BVPS_VND',
                            'EV/EBITDA': 'EV_EBITDA_Ratio'
                        }
                        
                        category = category_mapping.get(col[0], col[0])
                        metric = metric_mapping.get(col[1], col[1])
                        new_columns.append(f"{category}_{metric}")
                
                # Apply new column names
                processed_df.columns = new_columns
                
                print(f"✅ Processed financial ratios DataFrame with {len(new_columns)} columns")
                
            return processed_df
            
        except Exception as e:
            print(f"⚠️ Warning: Could not process ratios DataFrame: {str(e)}")
            return ratios_df
    
    def fetch_financial_data(self, stock_symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch comprehensive financial data for a given stock symbol.
        
        Args:
            stock_symbol: Vietnamese stock symbol (e.g., 'VIC', 'REE', 'VHM')
            
        Returns:
            Dict containing various financial dataframes
        """
        try:
            # Initialize vnstock
            stock = Vnstock().stock(symbol=stock_symbol, source="VCI")
            company = Vnstock().stock(symbol=stock_symbol, source="TCBS").company
            
            # Fetch raw financial data
            raw_ratios = stock.finance.ratio(period="year", lang="en", dropna=True)
            
            # Process the ratios DataFrame to handle multi-index columns
            processed_ratios = self._process_ratio_dataframe(raw_ratios)
            
            # Fetch other financial data
            financial_data = {
                'cash_flow': stock.finance.cash_flow(period="year"),
                'balance_sheet': stock.finance.balance_sheet(period="year", lang="en", dropna=True),
                'income_statement': stock.finance.income_statement(period="year", lang="en", dropna=True),
                'financial_ratios': processed_ratios,
                'dividend_schedule': company.dividends(),
                'stock_symbol': stock_symbol
            }
            
            print(f"✅ Successfully fetched financial data for {stock_symbol}")
            print(f"   - Processed ratios columns: {list(processed_ratios.columns) if not processed_ratios.empty else 'Empty'}")
            return financial_data
            
        except Exception as e:
            print(f"❌ Error fetching data for {stock_symbol}: {str(e)}")
            return {}
    
    def create_analysis_task(self, stock_symbol: str, analysis_type: str = "comprehensive") -> Task:
        """
        Create a financial analysis task for the agent.
        
        Args:
            stock_symbol: Stock symbol to analyze
            analysis_type: Type of analysis ('comprehensive', 'profitability', 'liquidity', 'visualization')
            
        Returns:
            Task: The configured analysis task
        """
        
        # Fetch the data first
        financial_data = self.fetch_financial_data(stock_symbol)
        
        if not financial_data:
            raise ValueError(f"Could not fetch financial data for {stock_symbol}")
        
        # Create context with the fetched data - INJECT REAL DATA
        data_context = f"""
        IMPORTANT: Use the REAL data provided below for {stock_symbol}, NOT simulated/mock data.
        
        REAL FINANCIAL DATA FOR {stock_symbol}:
        
        Available DataFrames with ACTUAL data:
        - cash_flow: Cash flow statement data
        - balance_sheet: Balance sheet data  
        - income_statement: Income statement data
        - financial_ratios: Key financial ratios (PROCESSED with English column names)
        - financial_ratios_raw: Raw ratios with multi-index columns (for reference)
        - dividend_schedule: Dividend payment history
        
        ACTUAL DATA FOR PANDAS OPERATIONS:
        
        # Income Statement DataFrame (use this exact data):
        income_statement_data = {financial_data['income_statement'].to_dict('records') if not financial_data['income_statement'].empty else []}
        income_statement_columns = {list(financial_data['income_statement'].columns) if not financial_data['income_statement'].empty else []}
        
        # Balance Sheet DataFrame (use this exact data):
        balance_sheet_data = {financial_data['balance_sheet'].to_dict('records') if not financial_data['balance_sheet'].empty else []}
        balance_sheet_columns = {list(financial_data['balance_sheet'].columns) if not financial_data['balance_sheet'].empty else []}
        
        # Financial Ratios DataFrame (use this exact data):
        financial_ratios_data = {financial_data['financial_ratios'].to_dict('records') if not financial_data['financial_ratios'].empty else []}
        financial_ratios_columns = {list(financial_data['financial_ratios'].columns) if not financial_data['financial_ratios'].empty else []}
        
        # Cash Flow DataFrame (use this exact data):
        cash_flow_data = {financial_data['cash_flow'].to_dict('records') if not financial_data['cash_flow'].empty else []}
        cash_flow_columns = {list(financial_data['cash_flow'].columns) if not financial_data['cash_flow'].empty else []}
        
        # Dividend Schedule DataFrame (use this exact data):
        dividend_data = {financial_data['dividend_schedule'].to_dict('records') if not financial_data['dividend_schedule'].empty else []}
        dividend_columns = {list(financial_data['dividend_schedule'].columns) if not financial_data['dividend_schedule'].empty else []}
        
        IMPORTANT: When using the Code Interpreter tool, you MUST specify the libraries_used parameter.
        
        Use the following code to recreate the ACTUAL DataFrames from the real data:
        ```python
        # REQUIRED: Specify libraries_used when calling Code Interpreter
        # libraries_used: ["pandas", "matplotlib", "seaborn", "plotly", "numpy"]
        
        import pandas as pd
        import numpy as np
        # NOTE: Visualization libraries may not be available in Docker environment
        # Focus on data analysis and numerical insights instead
        import warnings
        warnings.filterwarnings("ignore")
        
        # RECREATE ACTUAL DATAFRAMES FROM REAL DATA
        # Income Statement DataFrame
        income_statement = pd.DataFrame(income_statement_data)
        if income_statement_columns:
            income_statement.columns = income_statement_columns
        
        # Balance Sheet DataFrame  
        balance_sheet = pd.DataFrame(balance_sheet_data)
        if balance_sheet_columns:
            balance_sheet.columns = balance_sheet_columns
            
        # Financial Ratios DataFrame (with processed English column names)
        financial_ratios = pd.DataFrame(financial_ratios_data)
        if financial_ratios_columns:
            financial_ratios.columns = financial_ratios_columns
            
        # Cash Flow DataFrame
        cash_flow = pd.DataFrame(cash_flow_data)
        if cash_flow_columns:
            cash_flow.columns = cash_flow_columns
            
        # Dividend Schedule DataFrame
        dividend_schedule = pd.DataFrame(dividend_data)
        if dividend_columns:
            dividend_schedule.columns = dividend_columns
        
        # Function to process ratios DataFrame (already done for you)
        def process_ratios_dataframe(ratios_df):
            if ratios_df.empty or not isinstance(ratios_df.columns, pd.MultiIndex):
                return ratios_df
            
            processed_df = ratios_df.copy()
            new_columns = []
            
            category_mapping = {{
                'Chỉ tiêu cơ cấu nguồn vốn': 'Capital_Structure',
                'Chỉ tiêu hiệu quả hoạt động': 'Efficiency', 
                'Chỉ tiêu khả năng sinh lợi': 'Profitability',
                'Chỉ tiêu thanh khoản': 'Liquidity',
                'Chỉ tiêu định giá': 'Valuation'
            }}
            
            for col in processed_df.columns:
                if col[0] == 'Meta':
                    new_columns.append(col[1])
                else:
                    category = category_mapping.get(col[0], col[0])
                    new_columns.append(f"{{category}}_{{col[1]}}")
            
            processed_df.columns = new_columns
            return processed_df
        
        # Fetch data for {stock_symbol}
        stock = Vnstock().stock(symbol="{stock_symbol}", source="VCI")
        company = Vnstock().stock(symbol="{stock_symbol}", source="TCBS").company
        
        cash_flow = stock.finance.cash_flow(period="year")
        balance_sheet = stock.finance.balance_sheet(period="year", lang="en", dropna=True)
        income_statement = stock.finance.income_statement(period="year", lang="en", dropna=True)
        
        # Get raw ratios and process them
        financial_ratios_raw = stock.finance.ratio(period="year", lang="en", dropna=True)
        financial_ratios = process_ratios_dataframe(financial_ratios_raw)
        
        dividend_schedule = company.dividends()
        
        print(f"Data loaded for {stock_symbol}")
        print(f"Income Statement shape: {{income_statement.shape}}")
        print(f"Balance Sheet shape: {{balance_sheet.shape}}")
        print(f"Cash Flow shape: {{cash_flow.shape}}")
        print(f"Financial Ratios shape: {{financial_ratios.shape}}")
        print(f"Financial Ratios columns: {{list(financial_ratios.columns)}}")
        
        # Example: Access specific ratios
        if not financial_ratios.empty:
            print("\nKey Financial Ratios Available:")
            profitability_cols = [col for col in financial_ratios.columns if 'Profitability' in col]
            liquidity_cols = [col for col in financial_ratios.columns if 'Liquidity' in col]
            valuation_cols = [col for col in financial_ratios.columns if 'Valuation' in col]
            
            print(f"Profitability metrics: {{profitability_cols}}")
            print(f"Liquidity metrics: {{liquidity_cols}}")
            print(f"Valuation metrics: {{valuation_cols}}")
        ```
        """
        
        task_descriptions = {
            "comprehensive": f"""
            CRITICAL: Use ONLY the REAL data provided in the context below for {stock_symbol}. DO NOT create mock/simulated data.
            
            Perform a comprehensive financial analysis of {stock_symbol} including:
            
            1. **Data Overview**: Analyze the ACTUAL financial statements data provided
            2. **Profitability Analysis**: 
               - Use REAL revenue and profit data from income statement
               - Calculate actual profit margins from the data
               - Use actual ROE, ROA from the financial ratios
            3. **Liquidity Analysis**:
               - Use REAL current ratio, quick ratio from financial ratios
               - Analyze actual working capital from balance sheet
            4. **Financial Health**:
               - Use REAL debt-to-equity ratio from the data
               - Calculate actual interest coverage from income statement
            5. **Data Analysis & Insights** (NO VISUALIZATIONS):
               - Perform statistical analysis using pandas on the ACTUAL data
               - Calculate trends, growth rates, and comparative metrics
               - Create detailed data tables showing year-over-year changes
               - Provide numerical trend descriptions instead of charts
               - Focus on data-driven insights from the real Vietnamese stock data
            6. **Investment Insights**: Base recommendations on ACTUAL financial performance
            
            REMEMBER: All analysis must be based on the REAL {stock_symbol} data provided in the context.
            
            {data_context}
            """,
            
            "profitability": f"""
            Focus on profitability analysis of {stock_symbol}:
            
            1. Analyze revenue growth trends
            2. Calculate and visualize profit margins over time
            3. Compare ROE and ROA trends
            4. Create visualizations showing profitability metrics
            5. Provide insights on profitability trends
            
            {data_context}
            """,
            
            "visualization": f"""
            Create comprehensive visualizations for {stock_symbol}:
            
            1. Revenue and profit trend charts
            2. Financial ratios comparison charts
            3. Balance sheet composition analysis
            4. Cash flow analysis charts
            5. Interactive dashboards using plotly
            
            {data_context}
            """
        }
        
        return Task(
            description=task_descriptions.get(analysis_type, task_descriptions["comprehensive"]),
            expected_output=f"""
            A comprehensive financial analysis report for {stock_symbol} including:
            - Executive summary with key findings
            - Detailed financial metrics analysis
            - Professional visualizations and charts
            - Actionable investment insights and recommendations
            - All code used for analysis should be included and executable
            """,
            agent=self.data_analyst_agent,
        )
    
    def run_analysis(self, stock_symbol: str, analysis_type: str = "comprehensive") -> str:
        """
        Run the financial analysis for a given stock.
        
        Args:
            stock_symbol: Stock symbol to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            str: Analysis results
        """
        try:
            # Create the analysis task
            analysis_task = self.create_analysis_task(stock_symbol, analysis_type)
            
            # Create and run the crew
            crew = Crew(
                agents=[self.data_analyst_agent],
                tasks=[analysis_task],
                process=Process.sequential,
                verbose=True,
                memory=True
            )
            
            print(f"🚀 Starting financial analysis for {stock_symbol}...")
            result = crew.kickoff()
            
            print(f"✅ Analysis completed for {stock_symbol}")
            
            # Export findings to report.md
            self._export_to_report(stock_symbol, analysis_type, result)
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error during analysis: {str(e)}"
            print(error_msg)
            return error_msg
    
    def _export_to_report(self, stock_symbol: str, analysis_type: str, result: str) -> None:
        """
        Export the analysis findings to a report.md file.
        
        Args:
            stock_symbol: Stock symbol that was analyzed
            analysis_type: Type of analysis performed
            result: Analysis results from the CrewAI agent
        """
        try:
            import datetime
            
            # Generate timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create report content
            report_content = f"""# Financial Analysis Report

**Stock Symbol:** {stock_symbol}  
**Analysis Type:** {analysis_type.title()}  
**Generated:** {timestamp}  
**Generated by:** CrewAI Financial Data Analyst

---

{result}

---

*This report was automatically generated by the CrewAI Financial Data Analyst agent using Vietnamese stock market data from vnstock.*
"""
            
            # Write to report.md file
            report_path = "report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"📄 Analysis report exported to: {report_path}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not export report to file: {str(e)}")

def main():
    """
    Main function to demonstrate the Financial Data Analyst.
    """
    try:
        # Initialize the analyst
        analyst = FinancialDataAnalyst()
        
        # Example usage - analyze REE stock (from reference file)
        stock_symbol = "REE"
        
        print(f"🔍 Analyzing {stock_symbol} stock...")
        
        # Run comprehensive analysis
        result = analyst.run_analysis(
            stock_symbol=stock_symbol,
            analysis_type="comprehensive"
        )
        
        print("\n" + "="*80)
        print("ANALYSIS RESULTS")
        print("="*80)
        print(result)
        
    except Exception as e:
        print(f"❌ Application error: {str(e)}")
        print("Make sure to set your OPENAI_API_KEY environment variable.")

if __name__ == "__main__":
    main()
