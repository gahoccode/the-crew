mport os
from pandasai import Agent
from pandasai.llm import OpenAI
from pandasai.responses.streamlit_response import StreamlitResponse


import pandas as pd
from vnstock import Vnstock
import warnings

warnings.filterwarnings("ignore")
# Instantiate
stock_symbol = "REE"
stock = Vnstock().stock(symbol=stock_symbol, source="VCI")
company = Vnstock().stock(symbol=stock_symbol, source="TCBS").company

# Load dataframes
CashFlow = stock.finance.cash_flow(period="year")
BalanceSheet = stock.finance.balance_sheet(period="year", lang="en", dropna=True)
IncomeStatement = stock.finance.income_statement(period="year", lang="en", dropna=True)
Ratio = stock.finance.ratio(period="year", lang="en", dropna=True)
dividend_schedule = company.dividends()