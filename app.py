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
from crewai_tools import CodeInterpreterTool, BraveSearchTool, FileWriterTool
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
        self.brave_search = BraveSearchTool(n_results=3)
        self.file_writer = FileWriterTool()
        
        # Initialize LLM
        self.llm = LLM(
            model="gpt-4o-mini",
            api_key=self.openai_api_key
        )
        
        # Create agents
        self.data_analyst_agent = self._create_data_analyst_agent()
        self.news_research_agent = self._create_news_research_agent()
        self.reporting_agent = self._create_reporting_agent()
    
    def _create_data_analyst_agent(self) -> Agent:
        """
        Create a data analyst agent with code interpreter capabilities.
        
        Returns:
            Agent: The configured data analyst agent
        """
        return Agent(
            role="Senior Financial Data Analyst",
            goal="Analyze Vietnamese stock market data and provide comprehensive financial insights",
            backstory="You are an expert financial analyst specializing in Vietnamese stock market data. "
                     "You have deep knowledge of financial metrics, ratios, and market analysis. "
                     "You excel at interpreting complex financial data and providing actionable insights.",
            tools=[self.code_interpreter],
            llm=self.llm,
            allow_code_execution=True,
            allow_delegation=False,  # Specialist agent focused on financial analysis
            verbose=True,
            max_iter=2,
            memory=True,
            reasoning=True,
            max_reasoning_attempts=2
        )
    
    def _create_news_research_agent(self) -> Agent:
        """
        Create a news research agent that searches for recent company news.
        
        Returns:
            Agent: Configured news research agent
        """
        return Agent(
            role="Financial News Research Analyst",
            goal="Research and analyze recent news about Vietnamese companies to provide market intelligence",
            backstory="You are an expert financial news analyst specializing in Vietnamese market research. "
                     "You excel at finding relevant company news, management commentary, and strategic developments. "
                     "You focus on future plans, investment announcements, and executive insights that impact stock performance.",
            tools=[self.brave_search],
            llm=self.llm,
            allow_delegation=False,  # Specialist agent focused on news research
            verbose=True,
            max_iter=2,
            memory=True,
            reasoning=True,
            max_reasoning_attempts=2
        )
    
    def _create_reporting_agent(self) -> Agent:
        """
        Create a reporting agent that synthesizes financial analysis and news research.
        
        Returns:
            Agent: Configured reporting agent
        """
        return Agent(
            role="Executive Report Writer",
            goal="Synthesize financial analysis and market intelligence into comprehensive executive summaries",
            backstory="You are an expert executive report writer specializing in financial markets and corporate analysis. "
                     "You excel at combining quantitative financial data with qualitative market intelligence to create "
                     "compelling executive summaries. You write clear, actionable reports that help executives make "
                     "informed investment decisions based on both financial metrics and market sentiment.",
            tools=[self.file_writer],
            llm=self.llm,
            allow_delegation=True,  # Lead agent that can coordinate with other agents if needed
            verbose=True,
            max_iter=2,
            memory=True,
            reasoning=True,
            max_reasoning_attempts=2
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
        IMPORTANT: Use the REAL data provided below for {stock_symbol}

        
        FETCHED DATA FOR PANDAS OPERATIONS:
        
        # Income Statement DataFrame (use this exact data):
        income_statement_data = {financial_data['income_statement'].to_dict('records') if not financial_data['income_statement'].empty else []}
        income_statement_columns = {list(financial_data['income_statement'].columns) if not financial_data['income_statement'].empty else []}
        
        # Balance Sheet DataFrame (use this to query key financial metrics):
        balance_sheet_data = {financial_data['balance_sheet'].to_dict('records') if not financial_data['balance_sheet'].empty else []}
        balance_sheet_columns = {list(financial_data['balance_sheet'].columns) if not financial_data['balance_sheet'].empty else []}
        
        # Financial Ratios DataFrame (use this exact data):
        # Available column names (format: Category_Metric):
        # - Capital_Structure_Debt_to_Equity, Capital_Structure_Total_Debt_to_Equity, Capital_Structure_Fixed_Assets_to_Equity, Capital_Structure_Equity_to_Charter_Capital
        # - Efficiency_Asset_Turnover, Efficiency_Fixed_Asset_Turnover, Efficiency_Days_Sales_Outstanding, Efficiency_Days_Inventory_Outstanding, Efficiency_Days_Payable_Outstanding, Efficiency_Cash_Cycle, Efficiency_Inventory_Turnover
        # - Profitability_EBIT_Margin_Pct, Profitability_Gross_Margin_Pct, Profitability_Net_Margin_Pct, Profitability_ROE_Pct, Profitability_ROIC_Pct, Profitability_ROA_Pct, Profitability_EBITDA_Billion_VND, Profitability_EBIT_Billion_VND, Profitability_Dividend_Yield_Pct
        # - Liquidity_Current_Ratio, Liquidity_Cash_Ratio, Liquidity_Quick_Ratio, Liquidity_Interest_Coverage_Ratio, Liquidity_Financial_Leverage
        # - Valuation_Market_Cap_Billion_VND, Valuation_Shares_Outstanding_Million, Valuation_PE_Ratio, Valuation_PB_Ratio, Valuation_PS_Ratio, Valuation_P_CashFlow_Ratio, Valuation_EPS_VND, Valuation_BVPS_VND, Valuation_EV_EBITDA_Ratio
        financial_ratios_data = {financial_data['financial_ratios'].to_dict('records') if not financial_data['financial_ratios'].empty else []}
        financial_ratios_columns = {list(financial_data['financial_ratios'].columns) if not financial_data['financial_ratios'].empty else []}
        
        # Cash Flow DataFrame (use this exact data):
        cash_flow_data = {financial_data['cash_flow'].to_dict('records') if not financial_data['cash_flow'].empty else []}
        cash_flow_columns = {list(financial_data['cash_flow'].columns) if not financial_data['cash_flow'].empty else []}
        
        # Dividend Schedule DataFrame (use this exact data):
        dividend_data = {financial_data['dividend_schedule'].to_dict('records') if not financial_data['dividend_schedule'].empty else []}
        dividend_columns = {list(financial_data['dividend_schedule'].columns) if not financial_data['dividend_schedule'].empty else []}
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
            5. **Data Analysis & Insights**:
               - Perform statistical analysis using pandas on the ACTUAL data
               - Calculate trends, growth rates, and comparative metrics
               - Create detailed data tables showing year-over-year changes
               - Provide numerical trend descriptions and analysis
               - Focus on data-driven insights from the real Vietnamese stock data
            6. **Investment Insights**: Base recommendations on ACTUAL financial performance
            
            REMEMBER: All analysis must be based on the REAL {stock_symbol} data provided in the context.
            
            {data_context}
            """,
            
            "profitability": f"""
            Focus on profitability analysis of {stock_symbol} using REAL data:
            
            1. Analyze revenue growth trends using actual data
            2. Calculate profit margins over time from real financial statements
            3. Compare ROE and ROA trends using actual ratios
            4. Perform statistical analysis on profitability metrics
            5. Provide data-driven insights on profitability trends
            
            {data_context}
            """,
            
            "liquidity": f"""
            Focus on liquidity analysis of {stock_symbol} using REAL data:
            
            1. Analyze current and quick ratios from actual financial data
            2. Calculate working capital trends over time
            3. Assess cash flow patterns using real cash flow statements
            4. Evaluate short-term financial health indicators
            5. Provide insights on liquidity position and trends
            
            {data_context}
            """
        }
        
        return Task(
            description=task_descriptions[analysis_type],
            expected_output="Comprehensive financial analysis report with numerical insights and data-driven recommendations",
            agent=self.data_analyst_agent
        )
    
    def create_news_research_task(self, stock_symbol: str) -> Task:
        """
        Create a news research task for gathering recent company news.
        
        Args:
            stock_symbol: Stock symbol to research
            
        Returns:
            Task: Configured news research task
        """
        # Get company name mapping for better search results
        company_names = {
            "REE": "Refrigeration Electrical Engineering Corporation",
            "VIC": "Vingroup Joint Stock Company",
            "VHM": "Vinhomes Joint Stock Company",
            "VCB": "Vietcombank",
            "HPG": "Hoa Phat Group"
        }
        
        company_name = company_names.get(stock_symbol, stock_symbol)
        
        task_description = f"""
        Research recent news and developments about {company_name} ({stock_symbol}) focusing on:
        
        1. **Future Plans & Strategy**:
           - Search for: "{company_name} future plans 2024 2025"
           - Search for: "{stock_symbol} strategic development Vietnam"
           - Look for expansion plans, new projects, strategic initiatives
        
        2. **Recent Investments & Projects**:
           - Search for: "{company_name} investment projects Vietnam"
           - Search for: "{stock_symbol} new investments acquisitions"
           - Find information about capital expenditure, new ventures, partnerships
        
        3. **Management Commentary & Performance Views**:
           - Search for: "{company_name} CEO management outlook Vietnam"
           - Search for: "{stock_symbol} earnings call management commentary"
           - Look for executive statements, performance assessments, market outlook
        
        4. **Recent Financial News**:
           - Search for: "{company_name} financial results Vietnam stock"
           - Search for: "{stock_symbol} quarterly earnings Vietnam"
           - Find recent financial announcements, analyst coverage, market reactions
        
        **Search Instructions**:
        - Use multiple search queries to gather comprehensive information
        - Focus on news from the last 6-12 months
        - Prioritize Vietnamese financial news sources and official company announcements
        - Look for both Vietnamese and English language sources
        - Summarize key findings with dates and sources
        
        **Output Requirements**:
        - Organize findings by the 4 categories above
        - Include publication dates and source URLs when available
        - Highlight the most significant developments
        - Provide a summary of overall market sentiment
        - Focus on information that could impact stock performance
        """
        
        return Task(
            description=task_description,
            expected_output="Comprehensive news research report organized by categories with sources and dates",
            agent=self.news_research_agent
        )
    
    def create_executive_summary_task(self, stock_symbol: str, analysis_result: str, news_result: str) -> Task:
        """
        Create an executive summary task that synthesizes financial analysis and news research.
        
        Args:
            stock_symbol: Stock symbol analyzed
            analysis_result: Results from financial analysis agent
            news_result: Results from news research agent
            
        Returns:
            Task: Configured executive summary task
        """
        # Get company name mapping
        company_names = {
            "REE": "Refrigeration Electrical Engineering Corporation",
            "VIC": "Vingroup Joint Stock Company",
            "VHM": "Vinhomes Joint Stock Company",
            "VCB": "Vietcombank",
            "HPG": "Hoa Phat Group"
        }
        
        company_name = company_names.get(stock_symbol, stock_symbol)
        
        task_description = f"""
        Create a comprehensive executive summary for {company_name} ({stock_symbol}) by synthesizing the following information:
        
        **FINANCIAL ANALYSIS DATA:**
        {analysis_result}
        
        **NEWS & MARKET INTELLIGENCE:**
        {news_result}
        
        **EXECUTIVE SUMMARY REQUIREMENTS:**
        
        1. **Executive Overview** (2-3 paragraphs):
           - Brief company introduction and current market position
           - Key financial highlights from the analysis
           - Major recent developments from news research
        
        2. **Financial Performance Analysis**:
           - Summarize key financial metrics (profitability, liquidity, efficiency)
           - Highlight trends in revenue, margins, and returns
           - Compare performance against industry standards where possible
        
        3. **Strategic Developments & Market Intelligence**:
           - Synthesize recent news about future plans and strategic initiatives
           - Analyze management commentary and outlook
           - Assess recent investments and their potential impact
        
        4. **Investment Thesis**:
           - Combine quantitative analysis with qualitative insights
           - Identify key strengths and potential risks
           - Provide balanced assessment of growth prospects
        
        5. **Key Recommendations**:
           - Investment recommendation (Buy/Hold/Sell) with rationale
           - Key factors to monitor going forward
           - Risk considerations and mitigation strategies
        
        **FORMATTING REQUIREMENTS:**
        - Use proper Markdown formatting with headers, bullet points, and emphasis
        - Include specific numbers and dates where relevant
        - Keep the summary concise but comprehensive (3-4 pages)
        - Use professional, executive-level language
        - Ensure all claims are supported by the provided data
        
        **FILE OUTPUT:**
        Use the FileWriterTool to save the executive summary as 'executive_summary.md'
        
        The summary should provide executives with a clear, actionable overview that combines 
        both quantitative financial analysis and qualitative market intelligence.
        """
        
        return Task(
            description=task_description,
            expected_output="Comprehensive executive summary in markdown format saved to executive_summary.md",
            agent=self.reporting_agent,
            markdown=True
        )
    
    def run_analysis(self, stock_symbol: str, analysis_type: str = "comprehensive") -> str:
        """
        Run the complete three-stage analysis: financial data, news research, and executive summary.
        
        Args:
            stock_symbol: Stock symbol to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            str: Analysis results
        """
        try:
            # Stage 1 & 2: Run financial analysis and news research
            analysis_task = self.create_analysis_task(stock_symbol, analysis_type)
            news_task = self.create_news_research_task(stock_symbol)
            
            # Create and run the first crew with data and news agents
            data_crew = Crew(
                agents=[self.data_analyst_agent, self.news_research_agent],
                tasks=[analysis_task, news_task],
                process=Process.sequential,
                verbose=True,
                memory=True
            )
            
            print(f"🚀 Stage 1-2: Starting financial analysis and news research for {stock_symbol}...")
            data_result = data_crew.kickoff()
            
            print(f"✅ Financial analysis and news research completed for {stock_symbol}")
            
            # Extract results from both tasks
            analysis_output = str(data_result.tasks_output[0])
            news_output = str(data_result.tasks_output[1])
            
            # Stage 3: Create executive summary
            summary_task = self.create_executive_summary_task(stock_symbol, analysis_output, news_output)
            
            # Create and run the reporting crew
            reporting_crew = Crew(
                agents=[self.reporting_agent],
                tasks=[summary_task],
                process=Process.sequential,
                verbose=True,
                memory=True
            )
            
            print(f"🚀 Stage 3: Creating executive summary for {stock_symbol}...")
            summary_result = reporting_crew.kickoff()
            
            print(f"✅ Executive summary completed for {stock_symbol}")
            
            # Export findings to separate files
            self._export_to_report(stock_symbol, analysis_type, analysis_output)
            self._export_to_news(stock_symbol, news_output)
            
            print(f"📄 All reports exported: report.md, news.md, executive_summary.md")
            
            return str(summary_result)
            
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
            
            print(f"✅ Report exported to report.md")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not export report: {str(e)}")
    
    def _export_to_news(self, stock_symbol: str, news_result: str) -> None:
        """
        Export news research findings to news.md file.
        
        Args:
            stock_symbol: Stock symbol analyzed
            news_result: News research results from the agent
        """
        try:
            with open("news.md", "w", encoding="utf-8") as f:
                f.write(f"# {stock_symbol} - Recent News & Market Intelligence\n\n")
                f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
                f.write(str(news_result))
                f.write("\n\n---\n\n")
                f.write("*This report was generated using CrewAI with Brave Search for market intelligence.*\n")
            
            print(f"✅ News report exported to news.md")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not export news report: {str(e)}")

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
