#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
fred = Fred(api_key='0c2c8c3b572a356851d65eaa399a554e')
import plotly.express as px
import bcb
from bcb import Expectativas
from bcb import sgs
import datetime
from datetime import date
from datetime import timedelta
import warnings
import numpy as np
#import nasdaqdatalink
import quandl
quandl.ApiConfig.api_key = 'xhzW3vmVVALs4xStA47P'
import requests


# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title='HS Market Monitor', layout='wide')

#@st.cache_data
def get_data(tickers):
    data = yf.download(tickers, period='5y')['Adj Close']
    return data


# Ações
tickers_acoes = ['^GSPC', '^IXIC', '^RUT', '^N225', '^FTSE', '^STOXX50E', '^GDAXI', '^BVSP', '^AXJO', '^MXX', '000001.SS', '^HSI', '^NSEI', 'M.BA', 'FTSEMIB.MI']
df_acoes = get_data(tickers_acoes).fillna(method='ffill', axis=0)
names_acoes = ['S&P 500', 'Nasdaq', 'Russel 2000', 'Nikkei', 'FTSE', 'Euro Stoxx', 'DAX', 'Ibovespa', 'S&P ASX', 'BMV', 'Shanghai', 'Hang Seng', 'NSE', 'Merval', 'FTSE MIB']
column_mapping = dict(zip(tickers_acoes, names_acoes))
df_acoes.rename(columns=column_mapping, inplace=True)

# Moedas
tickers_moedas = ['6E=F', '6J=F', '6S=F', '6B=F', '6C=F', '6A=F','6M=F', '6Z=F', '6N=F', '6L=F', 'DX=F']
df_moedas = get_data(tickers_moedas).fillna(method='ffill', axis=0)
names_moedas = ['Euro', 'Japanese Yen', 'Swiss Franc', 'British Pound', 'Canadian Dollar', 'Australian Dollar', 'Mexican Peso', 'South African Rand','New Zeland Dollar', 'Brazilian Real', 'Dollar Index']
column_mapping = dict(zip(tickers_moedas, names_moedas))
# Renomeie as colunas
df_moedas.rename(columns=column_mapping, inplace=True)

# Commodities
tickers_commodities = ['CL=F', 'GC=F', 'SI=F', 'U-UN.TO', 'HG=F', 'ZC=F', 'CC=F', 'KC=F', 'OJ=F', 'SB=F', 'CT=F', 'LE=F', 'ZS=F', 'ZL=F', 'PL=F', 'HO=F', 'BZ=F']
df_commodities = get_data(tickers_commodities).fillna(method='ffill', axis=0)
names_commodities = ['Crude Oil', 'Gold', 'Silver', 'Uranium', 'Copper', 'Corn', 'Cocoa', 'Coffe', 'Orange Juice', 'Sugar', 'Cotton', 'Live Cattle', 'Soybean', 'Soybean Oil', 'Platinum', 'Heating Oil', 'Brent Oil']
column_mapping = dict(zip(tickers_commodities, names_commodities))
# Renomeie as colunas
df_commodities.rename(columns=column_mapping, inplace=True)

# Renda Fixa
tickers_rf = ['BIL', 'SHY', 'IEI', 'IEF', 'TLT', 'TIP', 'STIP', 'LQD', 'HYG', 'EMB', 'BNDX', 'IAGG','HYEM','IRFM11.SA', 'IMAB11.SA']
df_rf = get_data(tickers_rf).fillna(method='ffill', axis=0)
names_rf = ['BIL', 'SHY', 'IEI', 'IEF', 'TLT', 'TIP', 'STIP', 'LQD', 'HYG', 'EMB', 'BNDX', 'IAGG','HYEM','IRFM', 'IMAB']
column_mapping = dict(zip(tickers_rf, names_rf))
# Renomeie as colunas
df_rf.rename(columns=column_mapping, inplace=True)

# Crypto
tickers_crypto = ['BTC-USD', 'ETH-USD', 'MATIC-USD', 'LINK-USD', 'SOL-USD', 'UNI-USD', 'STX-USD', 'LDO-USD']
df_crypto = get_data(tickers_crypto).fillna(method='ffill', axis=0)
names_crypto = ['BTCUSD', 'ETHUSD', 'MATICUSD', 'LINKUSD', 'SOLUSD', 'UNIUSD', 'STXUSD', 'LDOUSD']
column_mapping = dict(zip(tickers_crypto, names_crypto))
# Renomeie as colunas
df_crypto.rename(columns=column_mapping, inplace=True)

# Factors
tikckers_factors = ['VLUE', 'QUAL', 'MTUM',  'SMLF', 'USMV', 'IVLU', 'IQLT', 'IMTM', 'ISCF', 'ACWV', 'EEM']
df_factors = get_data(tikckers_factors).ffill(axis=0)
names_factors = ['US Value', 'US Quality', 'US Momentum', 'US Small-Cap', 'US Low Vol', 'Global Value', 'Global Quality', 'Global Momentum', 'Global Small-Cap', 'Global Low Vol', 'EM Equity']
column_mapping = dict(zip(tikckers_factors, names_factors))
# Renomeie as colunas
df_factors.rename(columns=column_mapping, inplace=True)

#Sectors
tickers_sectors = ['XLE', 'XLY', 'XLP', 'XLF', 'XLI', 'XLV', 'XLK', 'XLB', 'XHB', 'XTL', 'XLU', 'SMH']
df_sectors = get_data(tickers_sectors).ffill(axis=0)
names_sectors = ['Energy', 'Consumer Discritionary', 'Consumer Staples', 'Financials','Industrials', 'Health Care', 'Technology', 'Materials', 'Homebuilders', 'Telecomunication', 'Utilities', 'Semiconductor']
column_mapping = dict(zip(tickers_sectors, names_sectors))
# Renomeie as colunas
df_sectors.rename(columns=column_mapping, inplace=True)

# Todos os Ativos
all_assets = pd.concat([df_acoes, df_moedas, df_commodities, df_rf, df_crypto, df_factors, df_sectors], axis=1).ffill().dropna()

options = ['Returns Heatmap', 'Correlation Matrix',  'Market Directionality', 'Macro Indicators', 'Positioning',  'Technical Analysis', 'Risk & Volatility']
selected = st.sidebar.selectbox('Main Menu', options)


if selected == 'Returns Heatmap':
    st.title('Returns Heatmap')
    st.markdown('##')


    # # Matriz de Retornos
    
    def returns_heatmap(df, classe):
        janelas = ['1D', '3D', '1W', '2W', '1M', '3M', '6M', '1Y', '2Y']
        matriz = pd.DataFrame(columns=janelas, index=df.columns)

        df_2y = df.ffill().pct_change(520).iloc[-1]
        df_1y = df.ffill().pct_change(260).iloc[-1]
        df_6m = df.ffill().pct_change(130).iloc[-1]
        df_3m = df.ffill().pct_change(60).iloc[-1]
        df_1m = df.ffill().pct_change(20).iloc[-1]
        df_2w = df.ffill().pct_change(10).iloc[-1]
        df_1w = df.ffill().pct_change(5).iloc[-1]
        df_3d = df.ffill().pct_change(3).iloc[-1]
        df_1d = df.ffill().pct_change(1).iloc[-1]


        matriz['1D'] = df_1d
        matriz['3D'] = df_3d
        matriz['1W'] = df_1w
        matriz['2W'] = df_2w
        matriz['1M'] = df_1m
        matriz['3M'] = df_3m
        matriz['6M'] = df_6m
        matriz['1Y'] = df_1y
        matriz['2Y'] = df_2y
        
        annotations = []
        for y, row in enumerate(matriz.values):
            for x, val in enumerate(row):
                annotations.append({
                    "x": matriz.columns[x],
                    "y": matriz.index[y],
                    "font": {"color": "black"},
                    "text": f"{val:.2%}",
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False
                })
        
        fig = go.Figure(data=go.Heatmap(
                        z=matriz.values,
                        x=matriz.columns.tolist(),
                        y=matriz.index.tolist(),
                        colorscale='RdYlGn',
                        zmin=matriz.values.min(), zmax=matriz.values.max(),  # para garantir que o 0 seja neutro em termos de cor
                        hoverongaps = False,
            text=matriz.apply(lambda x: x.map(lambda y: f"{y:.2%}")),
            hoverinfo='y+x+text',
            showscale=True,
            colorbar_tickformat='.2%'
        ))
        
        
        fig.update_layout(title=classe, annotations=annotations, width=600,  # Largura do gráfico
    height=600  # Altura do gráfico
)
        st.plotly_chart(fig)


    col1, col2 = st.columns(2)

    with col1:
        returns_heatmap(df_acoes, "Stocks")
    with col2:
        returns_heatmap(df_moedas, "Currencies (Futures)")

    col3, col4 = st.columns(2)

    with col3:
        returns_heatmap(df_commodities, "Commodities")
    with col4:
        returns_heatmap(df_rf, "Fixed Income")

    col5, col6 = st.columns(2)

    with col5:
        returns_heatmap(df_factors, "Factors")
    with col6:
        returns_heatmap(df_sectors, "US Sectors")

    col7, col8 = st.columns(2)

    with col7:
        returns_heatmap(df_crypto, "Crypto")


# # Matriz de Correlação

if selected == 'Correlation Matrix':
    st.title('Correlation Matrix')
    st.markdown('##')
   
    def corr_matrix(df, janela, classe):
        matriz = df.pct_change()[-janela:].corr()
        
        if classe == str('Multi-Asset'):
            
            fig = go.Figure(data=go.Heatmap(
                        z=matriz.values,
                        x=matriz.columns.tolist(),
                        y=matriz.index.tolist(),
                        colorscale='RdYlGn',
                        zmin=matriz.values.min(), zmax=matriz.values.max(),
                        hoverongaps = False,
        text=matriz.apply(lambda x: x.map(lambda y: f"{y:.2f}")),
                hoverinfo='y+x+text',
                showscale=True,
                colorbar_tickformat='.2f'
            ))
        
            fig.update_layout(title=classe, height=800  # Altura do gráfico
)
            
            st.plotly_chart(fig, use_container_width=True, height=1600  # Altura do gráfico
)

        else:
            
            annotations = []
            for y, row in enumerate(matriz.values):
                for x, val in enumerate(row):
                    annotations.append({
                        "x": matriz.columns[x],
                        "y": matriz.index[y],
                        "font": {"color": "black"},
                        "text": f"{val:.2f}",
                        "xref": "x1",
                        "yref": "y1",
                        "showarrow": False
                    })

            fig = go.Figure(data=go.Heatmap(
                            z=matriz.values,
                            x=matriz.columns.tolist(),
                            y=matriz.index.tolist(),
                            colorscale='RdYlGn',
                            zmin=matriz.values.min(), zmax=matriz.values.max(),  # para garantir que o 0 seja neutro em termos de cor
                            hoverongaps = False,
                text=matriz.apply(lambda x: x.map(lambda y: f"{y:.2f}")),
                hoverinfo='y+x+text',
                showscale=True,
                colorbar_tickformat='.2f'
            ))


            fig.update_layout(title=classe, annotations=annotations,  width=600,  # Largura do gráfico
    height=600  # Altura do gráfico
)
            

            st.plotly_chart(fig)



           

    lookback = st.number_input(label="Choose the lookback period", value=20)

    col1, col2 = st.columns(2)

    with col1:
        
        corr_matrix(df_acoes, lookback, 'Stocks')

    with col2:

        corr_matrix(df_moedas, lookback, 'Currencies (Futures)')

    col3, col4 = st.columns(2)

    with col3:

        corr_matrix(df_commodities, lookback, 'Commodities')

    with col4:
    
        corr_matrix(df_rf, lookback, 'Fixed Income')

    col5, col6 = st.columns(2)

    with col5:
        corr_matrix(df_factors, lookback, "Factors")
    with col6:
        corr_matrix(df_sectors, lookback, "US Sectors")

    col7, col8 = st.columns(2)

    with col7:
        corr_matrix(df_crypto, lookback, "Crypto")

    


    corr_matrix(all_assets, lookback, 'Multi-Asset')

    all_assets_list = all_assets.columns.tolist()
    
    all_assets_list.remove('S&P 500')
    all_assets_list.insert(0, 'S&P 500')

    asset = st.selectbox(
        'Choose the asset:',
        (all_assets_list))

    d_5 = all_assets.pct_change()[-5:].corr()[asset]
    d_10 = all_assets.pct_change()[-10:].corr()[asset]
    d_21 = all_assets.pct_change()[-21:].corr()[asset]
    d_63 = all_assets.pct_change()[-63:].corr()[asset]
    d_126 = all_assets.pct_change()[-126:].corr()[asset]
    d_252 = all_assets.pct_change()[-252:].corr()[asset]
    d_504 = all_assets.pct_change()[-504:].corr()[asset]
    d_756 = all_assets.pct_change()[-756:].corr()[asset]
    d_1260 = all_assets.pct_change()[-1260:].corr()[asset]
    
    asset_matrix = pd.concat([d_5, d_10, d_21, d_63, d_126, d_252, d_504, d_756, d_1260], axis=1)
    
    asset_matrix.columns = ['5 D', '10 D', '21 D', '63 D', '126 D', '252 D', '504 D', '756 D', '1260 D']
    
    asset_matrix.drop(index = asset, inplace=True)

    asset_matrix = asset_matrix.iloc[::-1]

    annotations = []
    for y, row in enumerate(asset_matrix.values):
        for x, val in enumerate(row):
            annotations.append({
                "x": asset_matrix.columns[x],
                "y": asset_matrix.index[y],
                "font": {"color": "black"},
                "text": f"{val:.2f}",
                "xref": "x1",
                "yref": "y1",
                "showarrow": False
            })

    
    fig = go.Figure(data=go.Heatmap(
                    z=asset_matrix.values,
                    x=asset_matrix.columns.tolist(),
                    y=asset_matrix.index.tolist(),
                    colorscale='RdYlGn',
                    zmin=asset_matrix.values.min(), zmax=asset_matrix.values.max(),  # para garantir que o 0 seja neutro em termos de cor
                    hoverongaps = False,
        text=asset_matrix.applymap(lambda x: f"{x:.2f}"),
        hoverinfo='y+x+text',
        showscale=True,
        colorbar_tickformat='.2f'
    ))

    fig.update_layout(
    xaxis=dict(
        side='top'  # Move os rótulos das colunas para o topo
    ))
    
    fig.update_layout(title=asset + " Correlation (Multi-Timeframe)", height=1600, annotations=annotations)
                
    st.plotly_chart(fig,use_container_width=True)


# # Directional Indicator

if selected == 'Market Directionality':
    st.title('Market Directionality')
    st.markdown('##')
    
    def directional_indicator(df, window, classe):

      
    
        df = df.dropna()

        price_change = df.diff(window).abs()
        
        # Daily price variability
        daily_variability = df.diff().abs().rolling(window=window).sum()
        
        # Trend Ratio
        tr = price_change / daily_variability
        
        # Directional Indicator
        di = tr.mean(axis=1).dropna()
        
        # Convert to DataFrame for plotting
        di_df = pd.DataFrame({
            'Date': di.index,
            'Value': di.values
        })
        
        fig = px.line(di_df, x='Date', y='Value', title=classe)
        
        fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

        fig.update_yaxes(tickformat=".2f")

        if classe == 'Multi-Asset':
            
            st.plotly_chart(fig, use_container_width=True)
    
        else:
            fig.update_layout( width=600,  # Largura do gráfico
                height=500  # Altura do gráfico
            )

            st.plotly_chart(fig)


    lookback = st.number_input(label="Choose the lookback period", value=260)

    col1, col2 = st.columns(2)

    with col1:
    
        directional_indicator(df_acoes, lookback, 'Stocks')

    with col2:

        directional_indicator(df_moedas, lookback, 'Currencies (Futures)')

    col3, col4 = st.columns(2)

    with col3:

        directional_indicator(df_commodities, lookback, 'Commodities')

    with col4:

        directional_indicator(df_rf, lookback, 'Fixed Income')

    col5, col6 = st.columns(2)

    with col5:

        directional_indicator(df_crypto, lookback, 'Crypto')



    directional_indicator(all_assets, lookback, 'Multi-Asset')


# # Macro Indicators

if selected == 'Macro Indicators':
    st.title('Macro Indicators')
    st.markdown('##')

    economy = st.radio(
    "Choose the economy:",
    ["United States", "Brazil"])

    st.markdown('##')

    if economy == "United States":

        st.subheader('Credit')
        
        # Yield Curve
        
        col1, col2 = st.columns(2)

        with col1:

            T10Y3M = fred.get_series('T10Y3M').dropna()

            T10Y2Y = fred.get_series('T10Y2Y').dropna()


            # Convert to DataFrame for plotting
            df_yc = pd.concat([T10Y3M, T10Y2Y], axis=1).dropna()
            df_yc.columns = ['T10Y3M', 'T10Y2Y']
            df_yc['Date'] = df_yc.index


            # Plot both 'Value' and '12M MA' on the same figure
            fig_yc = px.line(df_yc, x='Date', y=['T10Y3M', 'T10Y2Y'], title='Yield Curve')
                
            fig_yc.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(count=3, label="3y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(count=20, label="20y", step="year", stepmode="todate"),
                        dict(step="all")
                    ])
                )
            )

            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            # Adicionar o símbolo de % ao eixo y
            fig_yc.update_yaxes(tickformat=".2f", ticksuffix="%")

            fig_yc.update_layout( width=600,  # Largura do gráfico
        height=500  # Altura do gráfico
    )

            fig_yc.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
            
            st.plotly_chart(fig_yc)



        # ## High Yield Spread

        with col2:

            us_hys = fred.get_series('BAMLH0A0HYM2').dropna()

            eu_hys = fred.get_series('BAMLHE00EHYIOAS').dropna()

            em_hys = fred.get_series('BAMLEMHYHYLCRPIUSOAS').dropna()


            hys = pd.concat([us_hys, eu_hys, em_hys], axis=1).dropna()
            hys.columns=['US', 'EU', 'EM']

            hys['Date'] = hys.index


            # Plot both 'Value' and '12M MA' on the same figure
            fig_hys = px.line(hys, x='Date', y=['US', 'EU', 'EM'], title='High Yield Spread')
                
            fig_hys.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(count=3, label="3y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(count=20, label="20y", step="year", stepmode="todate"),
                        dict(step="all")
                    ])
                )
            )


            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            # Adicionar o símbolo de % ao eixo y
            fig_hys.update_yaxes(tickformat=".2f", ticksuffix="%")

            fig_hys.update_layout( width=600,  # Largura do gráfico
        height=500  # Altura do gráfico
    )

            fig_hys.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
            st.plotly_chart(fig_hys)


        col3, col4 = st.columns(2)
    
        with col3:
            T01Y = fred.get_series('DGS1').dropna()

            # Convert to DataFrame for plotting
            df_t1 = pd.DataFrame({'Date':T01Y.index,
                                'T01Y':T01Y.values,
                                 '12 MA': T01Y.rolling(260).mean().values})
            
            
            # Plot both 'Value' and '12M MA' on the same figure
            fig_t1y = px.line(df_t1, x='Date', y=['T01Y', '12 MA'], title='Hike Indicator')
            
            fig_t1y.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(count=10, label="10y", step="year", stepmode="backward"),
                    dict(count=20, label="20y", step="year", stepmode="todate"),
                    dict(step="all")
                ])
            )
            )
            
            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            # Adicionar o símbolo de % ao eixo y
            fig_t1y.update_yaxes(tickformat=".2f", ticksuffix="%")

            fig_t1y.update_layout( width=600,  # Largura do gráfico
        height=500  # Altura do gráfico
    )

            fig_t1y.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
            
            st.plotly_chart(fig_t1y)


        st.subheader('Liquidity')

        col4, col5 = st.columns(2)

        with col4:

            fci = fred.get_series('NFCI')

            fci_risk = fred.get_series('NFCIRISK')

            fci_leverage = fred.get_series('NFCILEVERAGE')

            fci_credit = fred.get_series('NFCICREDIT')



            # Convert to DataFrame for plotting
            df_fci = pd.DataFrame({
                    'Date': fci.index,
                    'FCI': fci.values
                
                }).dropna()

            # Plot both 'Value' and '12M MA' on the same figure
            fig_fci = px.line(df_fci, x='Date', y='FCI', title=' Financial Conditions Index')
                
            fig_fci.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(count=20, label="20y", step="year", stepmode="todate"),            
                        dict(step="all")
                    ])
                )
            )

            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            fig_fci.update_yaxes(tickformat=".2f")

            fig_fci.update_layout( width=600,  # Largura do gráfico
        height=500  # Altura do gráfico
    )
                
            st.plotly_chart(fig_fci)


        with col5:
    
            df_fci_comp = pd.concat([fci_risk, fci_leverage, fci_credit], axis=1).dropna()
            df_fci_comp.columns= ['Risk', 'Leverage', 'Credit']

            df_fci_comp['Date'] = df_fci_comp.index

            fig_fci_comp = px.line(df_fci_comp, x='Date', y=['Risk', 'Leverage', 'Credit'], title=' FCI Components')

            fig_fci_comp.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(count=20, label="20y", step="year", stepmode="todate"),
                        dict(step="all")
                    ])
                )
            )


            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            # Adicionar o símbolo de % ao eixo y
            fig_fci_comp.update_yaxes(tickformat=".2f")
            
            fig_fci_comp.update_layout( width=600,  # Largura do gráfico
        height=500  # Altura do gráfico
    )

            fig_fci_comp.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
            
            st.plotly_chart(fig_fci_comp)


        col6, col7 = st.columns(2)
        
        with col6:
            
            #fed_bs = fred.get_series('QBPBSTAS')
            tga = fred.get_series('WTREGEN')
            rrp = fred.get_series('RRPONTSYD')


           # Create initial df_fed_liq DataFrame
            df_fed_liq = pd.concat([tga, rrp], axis=1).ffill().dropna()

            df_fed_liq.columns = ['TGA', 'RRP']
            
            df_fed_liq['Date'] = df_fed_liq.index
            

            # Plot both 'Value' and '12M MA' on the same figure
            fig_fed_liq = px.line(df_fed_liq, x='Date', y=['TGA', 'RRP'], title='Fed Liquidity')

            fig_fed_liq.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(count=20, label="20y", step="year", stepmode="todate"),
                        dict(step="all")
                    ])
                )
            )
                
            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            fig_fed_liq.update_yaxes(tickformat=".2f")

            fig_fed_liq.update_layout( width=600,  # Largura do gráfico
        height=500  # Altura do gráfico
    )
                
            st.plotly_chart(fig_fed_liq)


        with col7:

            # ## United States M2

            m2_us = fred.get_series('WM2NS').dropna()

            # Convert to DataFrame for plotting
            df_m2_us = pd.DataFrame({
                    'Date': np.array(m2_us.index.to_pydatetime()),
                    '12-month change': m2_us.pct_change(52).values,  
                }).dropna()

            # Plot both 'Value' and '12M MA' on the same figure
            fig_m2_us = px.line(df_m2_us, x='Date', y='12-month change', title='United States M2')
                
            fig_m2_us.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(count=20, label="20y", step="year", stepmode="todate"),
                        dict(count=50, label="50y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            # Adicionar o símbolo de % ao eixo y
            fig_m2_us.update_yaxes(tickformat=".2%")

            fig_m2_us.update_layout( width=600,  # Largura do gráfico
        height=500  # Altura do gráfico
    )
                
            st.plotly_chart(fig_m2_us)

        
        st.subheader('Economic Activity')
        
        col8, col9 = st.columns(2)

        with col8:
            
            nfpr = fred.get_series('ADPWNUSNERSA').dropna()

            # Convert to DataFrame for plotting
            nfpr = pd.DataFrame({'Date': nfpr.index,
                                'Nonfarm Payroll':nfpr.pct_change(52).values})
            
            
            # Plot both 'Value' and '12M MA' on the same figure
            fig_nfpr = px.line(nfpr, x='Date', y='Nonfarm Payroll', title='Nonfarm Payroll 12-month change')
            
            fig_nfpr.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(count=10, label="10y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
            )
            
            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            # Adicionar o símbolo de % ao eixo y
            fig_nfpr.update_yaxes(tickformat='.2%')

            fig_nfpr.update_layout( width=600,  # Largura do gráfico
        height=500  # Altura do gráfico
    )

            fig_nfpr.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
            
            st.plotly_chart(fig_nfpr)


        with col9:
            
            oh = fred.get_series('AWOTMAN').dropna()

            # Convert to DataFrame for plotting
            oh = pd.DataFrame({'Date': oh.index,
                                'Overtime Hours':oh.values})
            
            
            # Plot both 'Value' and '12M MA' on the same figure
            fig_oh = px.line(oh, x='Date', y='Overtime Hours', title='Average Weekly Overtime Hours')
            
            fig_oh.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(count=10, label="10y", step="year", stepmode="backward"),
                    dict(count=20, label="20y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
            )

            fig_oh.update_layout( width=600,  # Largura do gráfico
        height=500  # Altura do gráfico
    )

            fig_oh.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

            st.plotly_chart(fig_oh)

        col10, col11 = st.columns(2)
            
        with col10:
    
            gea = fred.get_series('IGREA').dropna()
    
            # Convert to DataFrame for plotting
            gea = pd.DataFrame({'Date': gea.index,
                                'Index of Global Real Economic Activity':gea.values})
            
            
            # Plot both 'Value' and '12M MA' on the same figure
            fig_gea = px.line(gea, x='Date', y='Index of Global Real Economic Activity', title='Index of Global Real Economic Activity')
            
            fig_gea.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(count=10, label="10y", step="year", stepmode="backward"),
                    dict(count=20, label="20y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
            )
    
            fig_gea.update_layout( width=600,  # Largura do gráfico
        height=500  # Altura do gráfico
    )
    
            fig_gea.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
    )
            st.plotly_chart(fig_gea)
    
    
        with col11:
    
            # Sample data fetch (you can use your own methods)
            tax = fred.get_series('W006RC1Q027SBEA').dropna()
            
            df = pd.concat([tax], axis=1).dropna()
            df.columns = ['Federal government current tax receipts']
    
            
            # Create a subplot with dual Y-axes
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Plot data
            fig.add_trace(go.Scatter(x=df.index, y=df['Federal government current tax receipts'], name="Tax Receipts"), secondary_y=False)
            
            # Titles and labels
            fig.update_layout(title_text="Federal government current tax receipts")
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Tax Receipts", secondary_y=False)
            
            # Add range slider
            fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    rangeselector=dict(
                        buttons=list([
                            dict(count=5, label="5y", step="year", stepmode="backward"),
                            dict(count=10, label="10y", step="year", stepmode="backward"),
                            dict(count=20, label="20y", step="year", stepmode="todate"),
                            dict(step="all")
                        ])
                    )
                )
            )
            
            # Set width and height
            fig.update_layout(width=600, height=500)

            fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
            
            st.plotly_chart(fig)
        
        

        st.subheader('Housing')
        
        col12, col13 = st.columns(2)

        with col12:

            nhs = fred.get_series('HSN1F').dropna()
            
            
            # Convert to DataFrame for plotting
            nhs = pd.DataFrame({'Date': nhs.index,
                                'New Home Sales':nhs.values})
            
            
            # Plot both 'Value' and '12M MA' on the same figure
            fig_nhs = px.line(nhs, x='Date', y='New Home Sales', title='New Home Sales')
            
            fig_nhs.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(count=10, label="10y", step="year", stepmode="backward"),
                    dict(count=20, label="20y", step="year", stepmode="todate"),
                    dict(step="all")
                ])
            )
            )


            fig_nhs.update_layout( width=600,  # Largura do gráfico
        height=500  # Altura do gráfico
    )
            
            st.plotly_chart(fig_nhs)

        with col13:

            hperm = fred.get_series('PERMIT').dropna()


            # Convert to DataFrame for plotting
            hperm = pd.DataFrame({'Date': hperm.index,
                                'Housing Permits and Starts':hperm.values})
            
            
            # Plot both 'Value' and '12M MA' on the same figure
            fig_hperm = px.line(hperm, x='Date', y='Housing Permits and Starts', title='Housing Permits and Starts')
            
            fig_hperm.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(count=10, label="10y", step="year", stepmode="backward"),
                    dict(count=20, label="20y", step="year", stepmode="todate"),
                    dict(step="all")
                ])
            )
            )

            fig_hperm.update_layout( width=600,  # Largura do gráfico
        height=500  # Altura do gráfico
    )
        
            st.plotly_chart(fig_hperm)
        


        st.subheader('Sentiment')

        col14, col15 = st.columns(2)
        
        with col14:

            non_fin_corp = fred.get_series('NCBEILQ027S')/1000
            fin_corp = fred.get_series('FBCELLQ027S')/1000
            smv = (fin_corp + non_fin_corp)

            fed_liab = fred.get_series('FGSDODNS')
            household_liab = fred.get_series('CMDEBT')
            non_corp_lib = fred.get_series('BCNSDODNS')
            rest_wrld_liab = fred.get_series('DODFFSWCMI')
            gov_liab = fred.get_series('SLGSDODNS')
            bmv = fed_liab + household_liab + non_corp_lib + rest_wrld_liab + gov_liab


            aiae = smv/(smv+bmv)


            # Convert to DataFrame for plotting
            df_aiae = pd.DataFrame({
                    'Date': aiae.index,
                    'Value': aiae.values
                }).dropna()
            fig_aiae = px.line(df_aiae, x='Date', y='Value', title='AIAE Indicator')
                
            fig_aiae.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(count=20, label="20y", step="year", stepmode="todate"),
                        dict(count=50, label="50y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            fig_aiae.update_yaxes(tickformat=".2%")

            fig_aiae.update_layout( width=600,  # Largura do gráfico
        height=500  # Altura do gráfico
    )
        
            st.plotly_chart(fig_aiae)

        
        with col15:

            cons = fred.get_series('UMCSENT').dropna()

            # Convert to DataFrame for plotting
            cons = pd.DataFrame({'Date': cons.index,
                                'Consumer Sentiment':cons.values})
            
            
            # Plot both 'Value' and '12M MA' on the same figure
            fig_cons = px.line(cons, x='Date', y='Consumer Sentiment', title='Consumer Sentiment')
            
            fig_cons.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(count=10, label="10y", step="year", stepmode="backward"),
                    dict(count=20, label="20y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
            )

            fig_cons.update_layout( width=600,  # Largura do gráfico
        height=500  # Altura do gráfico
    )
    
            st.plotly_chart(fig_cons)
    

    if economy == "Brazil":

        st.subheader('Credit')
        
        col1, col2 = st.columns(2)
        
        with col1:

            # Saldo de Créditos Livres

            scl = sgs.get({'Total': 20542,
            'Households':20570,
            'Non-financial corporations':20543
            })

            scl = scl.pct_change(12)

            scl['Date'] = scl.index

            scl.dropna(inplace=True)

            fig_scl = px.line(scl, x='Date', y=['Total', 'Households', 'Non-financial corporations'], title='Nonearmarked credit operations outstanding (12-month change)')

            fig_scl.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=3, label="3y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            fig_scl.update_layout( width=600,  # Largura do gráfico
        height=500  # Altura do gráfico
    )
        

            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            # Adicionar o símbolo de % ao eixo y
            fig_scl.update_yaxes(tickformat=".2%")

            fig_scl.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
                
            st.plotly_chart(fig_scl)

        with col2:

            # Spread ICC

            icc = sgs.get({'Total': 27446,
            'Households':27448,
            'Non-financial corporations': 27447
                    })

            icc['Date'] = icc.index

            icc.dropna(inplace=True)

            fig_icc = px.line(icc, x='Date', y=['Total', 'Households', 'Non-financial corporations'], title='CCI Spread')

            fig_icc.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=3, label="3y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )


            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            # Adicionar o símbolo de % ao eixo y
            fig_icc.update_yaxes(tickformat=".2f", ticksuffix="%")

            fig_icc.update_layout( width=600,  # Largura do gráfico
        height=500  # Altura do gráfico
    )
            fig_icc.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
            
                
            st.plotly_chart(fig_icc)

        col3, col4 = st.columns(2)
        
        with col3:

            # Prazo Médio dos Empréstimos
            
            pmc = sgs.get({'Total': 20927,
            'Households':20954,
            'Non-financial corporations':20928
            }).dropna()

            pmc['Date'] = pmc.index

            fig_pmc = px.line(pmc, x='Date', y=['Total', 'Households', 'Non-financial corporations'], title='Average remaining maturity (months)')
            
            fig_pmc.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=3, label="3y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            fig_pmc.update_layout( width=600,  # Largura do gráfico
            height=500  # Altura do gráfico
            )

            fig_pmc.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
        

            st.plotly_chart(fig_pmc)


        with col4:

            # Endividamento das Famílias
            
            ef = sgs.get({'Household debt to income': 29037
        })

            ef['Date'] = ef.index

            ef.dropna(inplace=True)

            fig_ef = px.line(ef, x='Date', y='Household debt to income', title='Household debt to income')

            fig_ef.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=3, label="3y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )


            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            # Adicionar o símbolo de % ao eixo y
            fig_ef.update_yaxes(tickformat=".2f", ticksuffix="%")

            fig_ef.update_layout( width=600,  # Largura do gráfico
            height=500  # Altura do gráfico
            )
    

            st.plotly_chart(fig_ef)

        col5, col6 = st.columns(2)
        
        with col5:

            # Comprometimento de Renda das Famílias

            crf = sgs.get({'Household debt service ratio': 29265
        })

            crf['Date'] = crf.index

            crf.dropna(inplace=True)

            fig_crf = px.line(crf, x='Date', y='Household debt service ratio', title='Household debt service ratio')

            fig_crf.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=3, label="3y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )


            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            # Adicionar o símbolo de % ao eixo y
            fig_crf.update_yaxes(tickformat=".2f", ticksuffix="%")

            fig_crf.update_layout( width=600,  # Largura do gráfico
            height=500  # Altura do gráfico
            )
         
                
            st.plotly_chart(fig_crf)


        with col6:

            inad = sgs.get({'Total': 21082,
            'Households':21084,
            'Non-financial corporations':21083
            }).dropna()

            inad['Date'] = inad.index
            
            fig_inad = px.line(inad, x='Date', y=['Total', 'Households', 'Non-financial corporations'], title='Delinquent loans')
            
            fig_inad.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=3, label="3y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            fig_inad.update_layout( width=600,  # Largura do gráfico
            height=500  # Altura do gráfico
            )
            
            
            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            # Adicionar o símbolo de % ao eixo y
            fig_inad.update_yaxes(ticksuffix="%")

            fig_inad.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

        
            st.plotly_chart(fig_inad)

        st.subheader('Economic Activity')
        
        col7, col8 = st.columns(2)

        with col7:
            
            # ## Brazil 12-month Inflation Forecast
            # Instanciar a classe Expectativas
            em = Expectativas()        
            # Obter o endpoint para ExpectativasMercadoInflacao12Meses
            ep = em.get_endpoint('ExpectativasMercadoInflacao12Meses')
            # Puxar os dados filtrando pelo indicador IPCA
            data = ep.query().filter(ep.Indicador == 'IPCA').collect()

            df_inflation = pd.DataFrame({'Date':data['Data'].values,
                                        'IPCA Forecast':data['Mediana'].rolling(20).mean().values,
                                        '12-month change':data['Mediana'].rolling(20).mean().diff(260).values})
            # Plot both 'Value' and '12M MA' on the same figure
            fig_inf = px.line(df_inflation, x='Date', y=['IPCA Forecast', '12-month change'], title='Brazil Inflation Forecast')
                
            fig_inf.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=3, label="3y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(count=10, label="10y", step="year", stepmode="backward"),           
                        dict(step="all")
                    ])
                )
            )
            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            fig_inf.update_yaxes(tickformat=".2f", ticksuffix="%")
    
            fig_inf.update_layout( width=600,  # Largura do gráfico
        height=500  # Altura do gráfico
    )

            fig_inf.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

            st.plotly_chart(fig_inf)

        
        with col8:

            ibc = sgs.get({'IBC-Br': 24363}).dropna()

            ibc['Date'] = ibc.index
        
            fig_ibc = px.line(ibc , x='Date', y='IBC-Br', title='IBC-Br')
            
            fig_ibc.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=3, label="3y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            fig_ibc.update_layout( width=600,  # Largura do gráfico
            height=500  # Altura do gráfico
            )
            
            st.plotly_chart(fig_ibc)

        col9, col10 = st.columns(2)
        
        with col9:

            pib = sgs.get({'Investment': 24363,
                          'Services':22107,
                          'Household Consumption':22100,
                          'Goverment Cosnumption':22101}).dropna()

            pib['Date'] = pib.index
            
            fig_pib = px.line(pib , x='Date', y=['Investment','Services', 'Household Consumption','Goverment Cosnumption']  , title='GDP')
            
            fig_pib.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=3, label="3y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            fig_pib.update_layout( width=600,  # Largura do gráfico
            height=500  # Altura do gráfico
            )
            
            st.plotly_chart(fig_pib)



if selected == 'Positioning':
    st.title('Positioning')
    st.markdown('Commitment of Traders')
    st.markdown('##')

    category = [
    "CRYPTO CURRENCIES", "CRYPTO CURRENCIES", "CRYPTO CURRENCIES",
    "CURRENCIES", "CURRENCIES", "CURRENCIES", "CURRENCIES", "CURRENCIES", "CURRENCIES",
    "CURRENCIES", "CURRENCIES", "CURRENCIES", "CURRENCIES", "CURRENCIES",
    "ENERGIES", "ENERGIES", "ENERGIES",
    "EQUITIES", "EQUITIES", "EQUITIES", "EQUITIES", "EQUITIES - OTHER",
    "EQUITIES - OTHER", "EQUITIES - OTHER", "EQUITIES - OTHER", "EQUITIES - OTHER",
    "EQUITIES - OTHER", "EQUITIES - OTHER", "EQUITIES - OTHER", "EQUITIES - OTHER",
    "EQUITIES - OTHER", "EQUITIES - OTHER",
    "FIXED INCOME", "FIXED INCOME", "FIXED INCOME", "FIXED INCOME",
    "FIXED INCOME - OTHER", "FIXED INCOME - OTHER", "FIXED INCOME - OTHER",
    "FIXED INCOME - OTHER", "FIXED INCOME - OTHER", "FIXED INCOME - OTHER",
    "GRAINS", "GRAINS", "GRAINS", "GRAINS", "GRAINS", "GRAINS",
    "METALS", "METALS", "METALS", "METALS", "METALS", "METALS", "METALS",
    "SOFTS", "SOFTS", "SOFTS", "SOFTS", "SOFTS", "SOFTS"
]

    symbols = [
        "BTC", "MBT", "ETH", "6E", "DX", "6C", "6B", "6S", "6J", "6A",
        "6N", "6L", "6M", "6Z", "CL", "RB", "NG", "ES", "YM", "NQ",
        "RTY", "NKD", "VX", "ES CON", "ES ENE", "ES FIN", "ES HCI", "ES UTI",
        "ES MAT", "ES IND", "ES TEC", "MME", "ZT", "ZF", "ZN",'ZB', "GE",
        "SR1", "SR3", "ZQ", "TN", "UB", "ZW", "ZO", "ZC", 
        "ZM", "ZS", "ZL", "GC", "SI", "PL", "PA", "HG", 
        "AUP", "HRC", "CC", "KC", "CT", "LBS", "OJ", "SB"
    ]
    
    
    market = [
        "BITCOIN", "MICRO BITCOIN", "ETHEREUM", "EURO FX", "DX USD INDEX",
        "CANADIAN DOLLAR", "BRITISH POUND", "SWISS FRANC", "JAPANESE YEN", "AUSTRALIAN DOLLAR",
        "NEW ZEALAND DOLLAR", "BRAZILIAN REAL", "MEXICAN PESO", "SOUTH AFRICAN RAND",
        "CRUDE OIL", "GASOLINE RBOB", "NATURAL GAS", "S&P 500", "DOW JONES", "NASDAQ",
        "RUSSELL 2000", "NIKKEI", "VIX", "EMINI S&P CONSU STAPLES INDEX", "EMINI S&P ENERGY INDEX",
        "EMINI S&P FINANCIAL INDEX", "EMINI S&P HEALTH CARE INDEX", "EMINI S&P UTILITIES INDEX",
        "E-MINI S&P MATERIALS INDEX", "E-MINI S&P INDUSTRIAL INDEX", "E-MINI S&P TECHNOLOGY INDEX",
        "MSCI EM INDEX", "2-YEAR NOTES", "5-YEAR NOTES", "10-YEAR NOTES", "30-YEAR BONDS", "GE EURODOLLARS",
        "SR1 SECURED OVERNIGHT FINANCING RATE (1-MONTH)", "SR3 SECURED OVERNIGHT FINANCING RATE (3-MONTH)",
        "ZQ FED FUNDS", "TN ULTRA 10-YEAR NOTES", "ULTRA 30-YEAR BONDS", "WHEAT",
        "OATS", "CORN", "SOYBEAN MEAL", "SOYBEANS", "SOYBEAN OIL", "GOLD", "SILVER", "PLATINUM",
        "PALLADIUM", "COPPER", "ALUMINIUM", "STEEL", "COCOA", "COFFEE",
        "COTTON", "LUMBER",  "ORANGE JUICE", "SUGAR"
    ]
    
    
    
    contract_code = [
        "133741", "133742", "146021", "099741", "098662",
        "090741", "096742", "092741", "097741", "232741", "112741",
        "102741", "095741", "122741", "06765A", "111659", "0233AX",
        "13874A", "124606", "209742", "239742", "240741", "1170E1",
        "138748", "138749", "13874C", "13874E", "13874J", "13874H",
        "13874F", "13874J", "244042", "042601", "044601", "043602", '020601',
        "132741", "134742", "134741", "045601", "043607", "020604",
        "001602", "004603", "002602", "026603",
        "005602", "007601", "088691", "084691", "076651", "075651",
        "085692", "191693", "192651", "073732", "083731",
         "033661", "058644", "040701", "080732"
    ]
    
    
    
    
    cot_data = pd.DataFrame({'Category':category,
                            'Market':market,
                             'Symbols':symbols,                         
                            'Contract_Code':contract_code})


    
    ativo = st.selectbox(
        'Choose the market:',
        (cot_data['Market'].unique().tolist()))

    contract_code = cot_data[cot_data['Market'] == ativo]['Contract_Code']

    posicao_ativo = cot_data['Market'].tolist().index(ativo)
    
    nome_ativo =  str(cot_data['Market'][posicao_ativo]+' - '+cot_data['Symbols'][posicao_ativo])

    df_cot = quandl.get_table('QDL/LFON',contract_code=contract_code)
    
    data = df_cot.loc[df_cot['type']=='FO_L_OLD']
    
    # Para Commercials
    net_commercials = data['commercial_longs'] - data['commercial_shorts']
    
    # Para Small Speculators
    net_small_specs = (data['non_reportable_longs']-data['non_reportable_shorts'])
    
    # Para Large Speculators
    net_large_specs = (data['non_commercial_longs']-data['non_commercial_shorts'])-net_small_specs
    
    
    df = pd.DataFrame({
            'Date': pd.to_datetime(data['date']),
            'Sum of Small Speculators': net_small_specs,
            'Sum of Large Speculators': net_large_specs,
            'Sum of Commercials': net_commercials
        })
    
    
    
    # Create the figure
    fig_cot = go.Figure()
    
    # Add traces for each series with desired colors
    fig_cot.add_trace(go.Bar(x=df['Date'], y=df['Sum of Small Speculators'], name='Sum of Small Speculators', marker_color='yellow'))
    fig_cot.add_trace(go.Bar(x=df['Date'], y=df['Sum of Large Speculators'], name='Sum of Large Speculators', marker_color='blue'))
    fig_cot.add_trace(go.Bar(x=df['Date'], y=df['Sum of Commercials'], name='Sum of Commercials', marker_color='red'))
    
    # Update the layout
    fig_cot.update_layout(
        barmode='group',  # Barras agrupadas
        title=nome_ativo,
        xaxis_title="Date",
        yaxis_title="Number of Contracts",
        xaxis=dict(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(count=7, label="7y", step="year", stepmode="backward"),
                    dict(count=10, label="10y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        ),
        template="plotly_white",  # White background template
    )
    
    st.plotly_chart(fig_cot, use_container_width = True, height = 1600) 

    
def rank(series,n):
    # Calcula o percentil do último valor
    series = series[:n]
    percentile = (np.searchsorted(np.sort(series), series.iloc[0]) + 1) / len(series) * 100
    return round(percentile)

col1, col2, col3 = st.columns(3)

with col1:

    # Large Specs
    try:
        st.metric(label = 'Large Speculators Index', value = str(rank(df['Sum of Large Speculators'], 52)))
        st.text('1 Year')
    except:
        pass

with col2:

    # Small Specs
    try:
        st.metric(label = 'Small Speculators Index', value = str(rank(df['Sum of Small Speculators'], 52)))
        st.text('1 Year')
    except:
        pass

with col3:

    # Commercials
    try:
        st.metric(label = 'Commercials Index', value = str(rank(df['Sum of Commercials'], 52)))
        st.text('1 Year')
    except:
        pass

st.markdown('##')

col1, col2, col3 = st.columns(3)

with col1:

    # Large Specs
    try:
        st.metric(label = 'Large Speculators Index', value = str(rank(df['Sum of Large Speculators'], 156)))
        st.text('3 Years')
    except:
        pass

with col2:

    # Small Specs
    try:
        st.metric(label = 'Small Speculators Index', value = str(rank(df['Sum of Small Speculators'], 156)))
        st.text('3 Years')
    except:
        pass

with col3:

    # Commercials
    try:
        st.metric(label = 'Commercials Index', value = str(rank(df['Sum of Commercials'], 156)))
        st.text('3 Years')
    except:
        pass

st.markdown('##')

col1, col2, col3 = st.columns(3)

with col1:

    # Large Specs
    try:
        st.metric(label = 'Large Speculators Index', value = str(rank(df['Sum of Large Speculators'], 260)))
        st.text('5 Years')
    except:
        pass

with col2:

    # Small Specs
    try:
        st.metric(label = 'Small Speculators Index', value = str(rank(df['Sum of Small Speculators'], 260)))
        st.text('5 Years')
    except:
        pass

with col3:

    # Commercials
    try:
        st.metric(label = 'Commercials Index', value = str(rank(df['Sum of Commercials'], 260)))
        st.text('5 Years')
    except:
        pass


if selected == 'Technical Analysis':
    st.title('Technical Analysis')
    st.markdown('##')

    tickers_ibov = [
    "RRRP3.SA", "RADL3.SA", "SOMA3.SA", "BBAS3.SA", "CMIN3.SA",
    "JBSS3.SA", "IGTI11.SA", "TOTS3.SA", "EQTL3.SA", "SBSP3.SA",
    "ELET3.SA", "RENT3.SA", "ELET6.SA", "BRFS3.SA", "ITUB4.SA",
    "PETR3.SA", "PETR4.SA", "VALE3.SA", "MULT3.SA", "AZUL4.SA",
    "TAEE11.SA", "SANB11.SA", "MRFG3.SA", "UGPA3.SA", "BBDC3.SA",
    "ALPA4.SA", "EGIE3.SA", "USIM5.SA", "SMTO3.SA", "MRVE3.SA",
    "RAIZ4.SA", "MGLU3.SA", "CSNA3.SA", "VIVT3.SA", "EZTC3.SA",
    "BBDC4.SA", "CRFB3.SA", "BBSE3.SA", "VAMO3.SA", "BRKM5.SA",
    "YDUQ3.SA", "ENEV3.SA", "CMIG4.SA", "CPLE6.SA", "ABEV3.SA",
    "TRPL4.SA", "B3SA3.SA", "DXCO3.SA", "PCAR3.SA", "GGBR4.SA",
    "CYRE3.SA", "SUZB3.SA", "EMBR3.SA", "COGN3.SA", "ITSA4.SA",
    "ALOS3.SA", "LREN3.SA", "ARZZ3.SA", "CIEL3.SA", "PETZ3.SA",
    "HYPE3.SA", "GOAU4.SA", "TIMS3.SA", "RAIL3.SA", "LWSA3.SA",
    "BEEF3.SA", "PRIO3.SA", "BRAP4.SA", "KLBN11.SA", "HAPV3.SA",
    "CCRO3.SA", "FLRY3.SA", "NTCO3.SA", "ASAI3.SA", "ENGI11.SA",
    "CPFE3.SA", "BHIA3.SA", "IRBR3.SA", "BPAC11.SA", "WEGE3.SA",
    "VBBR3.SA", "RDOR3.SA", "SLCE3.SA", "GOLL4.SA", "RECV3.SA",
    "CSAN3.SA", "CVCB3.SA"
]

    tickers_bmv = [
    "MEGACPO.MX", "AC.MX", "ASURB.MX", "FEMSAUBD.MX", "CUERVO.MX", "KIMBERA.MX",
    "GRUMAB.MX", "GMEXICOB.MX", "GCC.MX", "BOLSAA.MX", "OMAB.MX", "GFNORTEO.MX",
    "BIMBOA.MX", "ALSEA.MX", "LABB.MX", "GAPB.MX", "GENTERA.MX", "TLEVISACPO.MX",
    "GCARSOA1.MX", "BBAJIOO.MX", "CEMEXCPO.MX", "AMXB.MX", "PINFRA.MX",
    "WALMEX.MX", "ORBIA.MX", "Q.MX", "RA.MX", "PE&OLES.MX", "CHDRAUIB.MX", "ELEKTRA.MX", "KOFUBL.MX",
    "ALFAA.MX", "GFINBURO.MX", "VESTA.MX", "VOLARA.MX"
]
  
    tickers_ftse = ['ANTO.L', 'STJ.L', 'FCIT.L', 'BEZ.L', 'SSE.L', 'WEIR.L', 'RIO.L', 'AUTO.L', 'AAL.L',
                    'BDEV.L', 'BKG.L', 'ITRK.L', 'KGF.L', 'MNG.L', 'MKS.L', 'NXT.L', 'PSON.L', 'PSN.L', 
                    'SN.L', 'REL.L', 'TW.L', 'LSEG.L', 'JD.L', 'WTB.L', 'AAF.L', 'EXPN.L', 'WPP.L', 'ENT.L', 
                    'DCC.L', 'SPX.L', 'NG.L', 'SHEL.L', 'FRAS.L', 'ULVR.L', 'GSK.L', 'ADM.L', 'INF.L', 'EDV.L', 
                    'IHG.L', 'HIK.L', 'OCDO.L', 'BRBY.L', 'CPG.L', 'DPLM.L', 'LAND.L', 'PSH.L', 'AZN.L', 'AHT.L', 
                    'ABF.L', 'RKT.L', 'BME.L', 'RR.L', 'BNZL.L', 'RMV.L', 'CRDA.L', 'RS1.L', 'CNA.L', 'DGE.L', 'HLMA.L', 
                    'HSBA.L', 'IMI.L', 'HLN.L', 'PHNX.L', 'HWDN.L', 'UU.L', 'IMB.L', 'VOD.L', 'RTO.L', 'NWG.L', 'BARC.L', 
                    'SBRY.L', 'SDR.L', 'BATS.L', 'SGRO.L', 'BA.L', 'SMIN.L', 'STAN.L', 'TSCO.L', 'SGE.L', 'CCH.L', 'LLOY.L', 
                    'SMT.L', 'MNDI.L', 'FLTR.L', 'CTEC.L', 'SMDS.L', 'ICP.L', 'III.L', 'MRO.L', 'IAG.L', 'SVT.L', 'BP.L', 
                    'BT-A.L', 'SKG.L', 'GLEN.L', 'FRES.L', 'UTG.L', 'AV.L', 'LGEN.L', 'PRU.L']
    
    tickers_nasdaq = [
    "MELI", "CDW", "MDLZ", "AMZN", "CPRT", "DDOG", "IDXX", "GOOG", "CSGP", 
    "ABNB", "CHTR", "TEAM", "CSCO", "INTC", "PANW", "MSFT", "NVDA", "CTSH", 
    "ISRG", "MRVL", "BKR", "BKNG", "ILMN", "TXN", "KHC", "GOOGL", "DASH", 
    "NFLX", "KDP", "ADP", "XEL", "CRWD", "WBA", "PDD", "AMD", "ADBE", "ODFL", 
    "AMGN", "AAPL", "CSX", "ADSK", "CTAS", "CMCSA", "CEG", "KLAC", "AEP", 
    "MDB", "AVGO", "CDNS", "PCAR", "COST", "REGN", "AMAT", "ZS", "LULU", 
    "CCEP", "GEHC", "SPLK", "SNPS", "EA", "FAST", "ANSS", "GILD", "META", 
    "EXC", "BIIB", "LRCX", "TTWO", "GFS", "VRTX", "ON", "ADI", "PAYX", 
    "PYPL", "QCOM", "MAR", "ROST", "SBUX", "PEP", "INTU", "MCHP", "MNST", 
    "WDAY", "ORLY", "ASML", "TSLA", "ROP", "NXPI", "HON", "AZN", "MRNA", 
    "SIRI", "WBD", "MU", "TMUS", "VRSK", "DLTR", "FANG", "DXCM", "FTNT", 
    "TTD"]
    
    tickers_sp500 = ['LYB', 'AXP', 'VZ', 'AVGO', 'BA', 'CAT', 'JPM', 'CVX', 'KO', 'ABBV', 'DIS', 'FLT', 'EXR', 
                     'XOM', 'PSX', 'GE', 'HPQ', 'HD', 'MPWR', 'IBM', 'JNJ', 'LULU', 'MCD', 'MRK', 'MMM', 'AWK', 
                     'BAC', 'PFE', 'PG', 'T', 'TRV', 'RTX', 'ADI', 'WMT', 'CSCO', 'INTC', 'GM', 'MSFT', 'DG', 'CI', 
                     'KMI', 'C', 'AIG', 'MO', 'HCA', 'IP', 'HPE', 'ABT', 'AFL', 'APD', 'RCL', 'HES', 'ADM', 'ADP', 'VRSK', 
                     'AZO', 'LIN', 'AVY', 'ENPH', 'MSCI', 'BALL', 'AXON', 'CDAY', 'CARR', 'BK', 'OTIS', 'BAX', 'BDX', 'BRK-B', 
                     'BBY', 'BSX', 'BMY', 'BF-B', 'CTRA', 'CPB', 'HLT', 'CCL', 'QRVO', 'BLDR', 'UDR', 'CLX', 'PAYC', 'CMS', 'CL', 
                     'EPAM', 'CMA', 'CAG', 'ABNB', 'ED', 'GLW', 'CMI', 'CZR', 'DHR', 'TGT', 'DE', 'D', 'DOV', 'LNT', 'STLD', 'DUK', 
                     'REG', 'ETN', 'ECL', 'RVTY', 'EMR', 'EOG', 'AON', 'ETR', 'EFX', 'EQT', 'IQV', 'IT', 'FDX', 'FMC', 'BRO', 'F', 'NEE', 
                     'BEN', 'GRMN', 'FCX', 'DXCM', 'GD', 'GIS', 'GPC', 'ATO', 'GWW', 'HAL', 'LHX', 'PEAK', 'PODD', 'CTLT', 'FTV', 'HSY', 'SYF', 
                     'HRL', 'AJG', 'MDLZ', 'CNP', 'HUM', 'WTW', 'ITW', 'CDW', 'TT', 'IPG', 'IFF', 'GNRC', 'NXPI', 'K', 'BR', 'KMB', 'KIM', 'ORCL', 
                     'KR', 'LEN', 'LLY', 'BBWI', 'CHTR', 'L', 'LOW', 'HUBB', 'IEX', 'MMC', 'MAS', 'SPGI', 'MDT', 'VTRS', 'CVS', 'DD', 'MU', 'MSI', 
                     'CBOE', 'LH', 'NEM', 'NKE', 'NI', 'NSC', 'PFG', 'ES', 'NOC', 'WFC', 'NUE', 'OXY', 'OMC', 'OKE', 'RJF', 'PCG', 'PH', 'ROL', 
                     'PPL', 'COP', 'PHM', 'PNW', 'PNC', 'PPG', 'PGR', 'VLTO', 'PEG', 'RHI', 'COO', 'EIX', 'SLB', 'SCHW', 'SHW', 'WST', 'SJM', 
                     'SNA', 'AME', 'UBER', 'SO', 'TFC', 'LUV', 'WRB', 'SWK', 'PSA', 'ANET', 'SYY', 'CTVA', 'TXN', 'TXT', 'TMO', 'TJX', 'GL', 'JCI', 
                     'ULTA', 'UNP', 'KEYS', 'UNH', 'BX', 'MRO', 'BIO', 'VTR', 'VFC', 'VMC', 'WY', 'WHR', 'WMB', 'CEG', 'WEC', 'ADBE', 'AES', 'EXPD', 
                     'AMGN', 'AAPL', 'ADSK', 'CTAS', 'CMCSA', 'TAP', 'KLAC', 'MAR', 'FI', 'MKC', 'PCAR', 'COST', 'SYK', 'TSN', 'LW', 'AMAT', 'AAL', 
                     'CAH', 'CINF', 'PARA', 'DHI', 'EA', 'FICO', 'FAST', 'MTB', 'XEL', 'FITB', 'GILD', 'HAS', 'HBAN', 'WELL', 'BIIB', 'NTRS', 'PKG', 
                     'PAYX', 'QCOM', 'ROST', 'IDXX', 'SBUX', 'KEY', 'FOXA', 'FOX', 'STT', 'NCLH', 'USB', 'AOS', 'GEN', 'TROW', 'WM', 'STZ', 'XRAY', 'ZION', 
                     'IVZ', 'INTU', 'MS', 'MCHP', 'CB', 'HOLX', 'CFG', 'JBL', 'ORLY', 'ALL', 'EQR', 'BWA', 'KDP', 'HST', 'INCY', 'SPG', 'EMN', 'AVB', 'PRU', 
                     'UPS', 'WBA', 'STE', 'MCK', 'LMT', 'COR', 'COF', 'WAT', 'NDSN', 'DLTR', 'DRI', 'EVRG', 'MTCH', 'DPZ', 'NVR', 'NTAP', 'ODFL', 'DVA', 'HIG', 
                     'IRM', 'EL', 'CDNS', 'TYL', 'UHS', 'SWKS', 'DGX', 'ROK', 'KHC', 'AMT', 'REGN', 'AMZN', 'JKHY', 'RL', 'BXP', 'APH', 'HWM', 'PXD', 'VLO', 'SNPS', 
                     'ETSY', 'CHRW', 'ACN', 'TDG', 'YUM', 'PLD', 'FE', 'VRSN', 'PWR', 'HSIC', 'AEE', 'ANSS', 'FDS', 'NVDA', 'CTSH', 'ISRG', 'TTWO', 'RSG', 'EBAY', 'GS', 
                     'SBAC', 'SRE', 'MCO', 'ON', 'BKNG', 'FFIV', 'AKAM', 'CRL', 'MKTX', 'DVN', 'TECH', 'GOOGL', 'TFX', 'NFLX', 'ALLE', 'A', 'WBD', 'ELV', 'TRMB', 'CME', 
                     'JNPR', 'BLK', 'DTE', 'NDAQ', 'CE', 'PM', 'CRM', 'IR', 'HII', 'ROP', 'MET', 'TPR', 'CSX', 'EW', 'AMP', 'ZBRA', 'ZBH', 'CBRE', 'CPT', 'MA', 'KMX', 'ICE', 
                     'FIS', 'CMG', 'WYNN', 'LYV', 'AIZ', 'NRG', 'RF', 'MNST', 'MOS', 'BKR', 'EXPE', 'CF', 'LDOS', 'APA', 'GOOG', 'FSLR', 'TEL', 'DFS', 'V', 'MAA', 'XYL', 'MPC', 
                     'AMD', 'TSCO', 'RMD', 'MTD', 'J', 'CPRT', 'VICI', 'FTNT', 'ALB', 'MRNA', 'ESS', 'CSGP', 'O', 'WRK', 'WAB', 'POOL', 'WDC', 'PEP', 'FANG', 'PANW', 'NOW', 'CHD', 
                     'FRT', 'MGM', 'AEP', 'INVH', 'PTC', 'JBHT', 'LRCX', 'MHK', 'PNR', 'GEHC', 'VRTX', 'AMCR', 'META', 'TMUS', 'URI', 'HON', 'ARE', 'DAL', 'STX', 'UAL', 'NWS', 'CNC', 
                     'MLM', 'TER', 'PYPL', 'TSLA', 'ACGL', 'DOW', 'EG', 'TDY', 'NWSA', 'EXC', 'GPN', 'CCI', 'APTV', 'ALGN', 'ILMN', 'KVUE', 'TRGP', 'BG', 'LKQ', 'ZTS', 'DLR', 'EQIX', 'LVS', 'MOH']
    
    tickers_dow = ['AXP', 'VZ', 'BA', 'CAT', 'JPM', 'CVX', 'KO', 'DIS', 'HD', 'IBM', 'JNJ', 
                   'MCD', 'MRK', 'MMM', 'PG', 'TRV', 'WMT', 'HON', 'INTC', 'MSFT', 'GS', 
                   'AMGN', 'AAPL', 'CSCO', 'UNH', 'CRM', 'DOW', 'V', 'WBA', 'NKE']

    tickers_euro_stoxx = [
    "OR.PA", "DG.PA", "BBVA.MC", "SAN.MC", "ASML.AS", "TTE.PA",
    "AI.PA", "CS.PA", "BNP.PA", "BN.PA", "SGO.PA", "EL.PA",
    "MC.PA", "KER.PA", "RACE.MI", "SAF.PA", "WKL.AS", "AD.AS",
    "UCG.MI", "IBE.MC", "RMS.PA", "INGA.AS", "STLAM.MI", "PRX.AS",
    "ITX.MC", "ISP.MI", "ENI.MI", "ABI.BR", "NDA-FI.HE", "ADYEN.AS",
    "SAN.PA", "ENEL.MI", "NOKIA.HE", "SU.PA", "ALV.DE", "AIR.PA",
    "BAYN.DE", "BMW.DE", "BAS.DE", "SIE.DE", "VOW3.DE", "MUV2.DE",
    "SAP.DE", "RI.PA", "ADS.DE", "DTE.DE", "DHL.DE", "MBG.DE",
    "IFX.DE", "DB1.DE"
]

    tickers_asx = [
    "CQE.AX", "QBE.AX", "ALX.AX", "BKW.AX", "WEB.AX", "PNI.AX", "BOQ.AX", 
    "ORA.AX", "GNC.AX", "WTC.AX", "WES.AX", "RRL.AX", "EDV.AX", "BPT.AX", 
    "NEC.AX", "EVT.AX", "FLT.AX", "RMD.AX", "MPL.AX", "LIC.AX", "ALU.AX", 
    "A2M.AX", "NSR.AX", "KLS.AX", "LTM.AX", "CPU.AX", "REA.AX", "SOL.AX", 
    "EMR.AX", "ANN.AX", "WDS.AX", "JBH.AX", "TNE.AX", "GOR.AX", "EVN.AX", 
    "IFL.AX", "DEG.AX", "RMS.AX", "SUL.AX", "BGA.AX", "SIQ.AX", "GPT.AX", 
    "DTL.AX", "CLW.AX", "CRN.AX", "JHX.AX", "CGC.AX", "SFR.AX", "FPH.AX", 
    "HLS.AX", "CNI.AX", "PNV.AX", "HMC.AX", "PMV.AX", "KAR.AX", "TLX.AX", 
    "PRU.AX", "NAN.AX", "NXT.AX", "CXO.AX", "PXA.AX", "CAR.AX", "DMP.AX", 
    "CWY.AX", "SEK.AX", "NST.AX", "CHC.AX", "XRO.AX", "COL.AX", "NHF.AX", 
    "MIN.AX", "SCG.AX", "VEA.AX", "SYA.AX", "QUB.AX", "TLC.AX", "BHP.AX", 
    "NAB.AX", "AWC.AX", "WBC.AX", "NWH.AX", "ORG.AX", "ALD.AX", "AUB.AX", 
    "REH.AX", "SLR.AX", "CBA.AX", "RIO.AX", "HVN.AX", "ORI.AX", "SDF.AX", 
    "LLC.AX", "NWS.AX", "SUN.AX", "STO.AX", "CCP.AX", "IRE.AX", "SGM.AX", 
    "ILU.AX", "WOW.AX", "PDN.AX", "SHL.AX", "RHC.AX", "CSL.AX", "LYC.AX", 
    "GMD.AX", "MTS.AX", "SVW.AX", "BEN.AX", "CQR.AX", "TAH.AX", "IAG.AX", 
    "FMG.AX", "COH.AX", "MQG.AX", "ASX.AX", "BWP.AX", "AMP.AX", "HLI.AX", 
    "MGR.AX", "RGN.AX", "WAF.AX", "BLD.AX", "WHC.AX", "WOR.AX", "IPL.AX", 
    "CMM.AX", "GMG.AX", "AGL.AX", "SGR.AX", "BOE.AX", "AIA.AX", "BSL.AX", 
    "PME.AX", "ALL.AX", "IEL.AX", "S32.AX", "NUF.AX", "INA.AX", "LNW.AX", 
    "ELD.AX", "CTD.AX", "PPT.AX", "CGF.AX", "SGP.AX", "JLG.AX", "VCX.AX", 
    "RWC.AX", "NEU.AX", "IPH.AX", "CNU.AX", "NWL.AX", "VUK.AX", "AMC.AX", 
    "LOV.AX", "BGL.AX", "LTR.AX", "VNT.AX", "APA.AX", "ALQ.AX", "APE.AX", 
    "MP1.AX", "MFG.AX", "ARB.AX", "HDN.AX", "GUD.AX", "TPG.AX", "MND.AX", 
    "QAN.AX", "CIA.AX", "PLS.AX", "HUB.AX", "BRG.AX", "TCL.AX", "TLS.AX", 
    "NHC.AX", "SQ2.AX", "NIC.AX", "BXB.AX", "CKF.AX", "AZJ.AX", "360.AX", 
    "CHN.AX", "IGO.AX", "CSR.AX", "DXS.AX", "DRR.AX", "ARF.AX", "CIP.AX", 
    "BAP.AX", "NEM.AX", "ING.AX", "FBU.AX", "DHG.AX", "ANZ.AX", "DOW.AX", 
    "TWE.AX", "SPK.AX", "WBT.AX"
]

    tickers_dax = [
    "ENR.DE", "SY1.DE", "PAH3.DE", "MTX.DE", "RHM.DE", "DTG.DE", "SHL.DE",
    "ZAL.DE", "QIA.DE", "SRT3.DE", "BNR.DE", "AIR.DE", "ALV.DE", "1COV.DE",
    "RWE.DE", "BAYN.DE", "BMW.DE", "CBK.DE", "DBK.DE", "BAS.DE", "HEN3.DE",
    "SIE.DE", "VOW3.DE", "EOAN.DE", "BEI.DE", "HEI.DE", "MUV2.DE", "FRE.DE",
    "SAP.DE", "MRK.DE", "ADS.DE", "DTE.DE", "DHL.DE", "MBG.DE", "IFX.DE",
    "DB1.DE", "VNA.DE", "P911.DE", "HNR1.DE", "CON.DE"
]
    
    tickers_nse = [
    "APOLLOHOSP.NS", "ULTRACEMCO.NS", "KOTAKBANK.NS", "NESTLEIND.NS", "HEROMOTOCO.NS",
    "BAJFINANCE.NS", "BAJAJ-AUTO.NS", "HDFCLIFE.NS", "BAJAJFINSV.NS", "BHARTIARTL.NS",
    "TATACONSUM.NS", "ONGC.NS", "COALINDIA.NS", "CIPLA.NS", "HINDALCO.NS", "LT.NS",
    "BRITANNIA.NS", "MARUTI.NS", "ITC.NS", "ADANIENT.NS", "TCS.NS", "LTIM.NS", "TECHM.NS",
    "INDUSINDBK.NS", "RELIANCE.NS", "TITAN.NS", "TATASTEEL.NS", "MM.NS", "NTPC.NS",
    "WIPRO.NS", "EICHERMOT.NS", "AXISBANK.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS",
    "HDFCBANK.NS", "BPCL.NS", "ASIANPAINT.NS", "JSWSTEEL.NS", "DRREDDY.NS", "HINDUNILVR.NS",
    "INFY.NS", "HCLTECH.NS", "POWERGRID.NS", "ICICIBANK.NS", "ADANIPORTS.NS", "TATAMOTORS.NS",
    "DIVISLAB.NS", "UPL.NS", "GRASIM.NS"]

    tickers_nikkei = [
    "8766.T", "8309.T", "5411.T", "8316.T", "8411.T", "2768.T", "1721.T",
    "8795.T", "6674.T", "3382.T", "4568.T", "4188.T", "3436.T", "1605.T",
    "8354.T", "4751.T", "3086.T", "8725.T", "3099.T", "2269.T", "8630.T",
    "5020.T", "9843.T", "6988.T", "3289.T", "1808.T", "6723.T", "6861.T",
    "8697.T", "2413.T", "6098.T", "3863.T", "4631.T", "6501.T", "7267.T",
    "6971.T", "6752.T", "6758.T", "6762.T", "6857.T", "2802.T", "9202.T",
    "6770.T", "6113.T", "2502.T", "3407.T", "5201.T", "7741.T", "5108.T",
    "7751.T", "6952.T", "8331.T", "9502.T", "4519.T", "7762.T", "8253.T",
    "7912.T", "6367.T", "4506.T", "7735.T", "1925.T", "8601.T", "4061.T",
    "5714.T", "6361.T", "4523.T", "6954.T", "6504.T", "7270.T", "4901.T",
    "6981.T", "5803.T", "6702.T", "5801.T", "6594.T", "9201.T", "7205.T",
    "6305.T", "7004.T", "7013.T", "7202.T", "8001.T", "5631.T", "1963.T",
    "8267.T", "1812.T", "9503.T", "4452.T", "6645.T", "7012.T", "9107.T",
    "8591.T", "9434.T", "9008.T", "9009.T", "2801.T", "7974.T", "2503.T",
    "5406.T", "6301.T", "9766.T", "4902.T", "6473.T", "6326.T", "3405.T",
    "4151.T", "8002.T", "8252.T", "7261.T", "6479.T", "9301.T", "8058.T",
    "6503.T", "8802.T", "7011.T", "4043.T", "5711.T", "7211.T", "8031.T",
    "8801.T", "5706.T", "9104.T", "7272.T", "4183.T", "6701.T", "5333.T",
    "2871.T", "7731.T", "8304.T", "5214.T", "2282.T", "7832.T", "6471.T",
    "5831.T", "5401.T", "1332.T", "9432.T", "9101.T", "6902.T", "4021.T",
    "6273.T", "7201.T", "2002.T", "4385.T", "8604.T", "6472.T", "1802.T",
    "9007.T", "3861.T", "7186.T", "4661.T", "6103.T", "7733.T", "5233.T",
    "9532.T", "5541.T", "7752.T", "6178.T", "2501.T", "9735.T", "1928.T",
    "6753.T", "1803.T", "4063.T", "4507.T", "4911.T", "5019.T", "4004.T",
    "5232.T", "4005.T", "8053.T", "5802.T", "6302.T", "4755.T", "5713.T",
    "8830.T", "7269.T", "1801.T", "6976.T", "2531.T", "8233.T", "4502.T",
    "3401.T", "4543.T", "6724.T", "6920.T", "9001.T", "9602.T", "5301.T",
    "9501.T", "2432.T", "8035.T", "9531.T", "8804.T", "9005.T", "3659.T",
    "7911.T", "3402.T", "4042.T", "5332.T", "4578.T", "7203.T", "8015.T",
    "4208.T", "7951.T", "4503.T", "9147.T", "9064.T", "6506.T", "6841.T",
    "5101.T", "9020.T", "9433.T", "4324.T", "9984.T", "9983.T", "2914.T",
    "9613.T", "9021.T", "9022.T", "8750.T", "4689.T", "4704.T", "8306.T",
    "8308.T"
]


    market = st.selectbox(
        'Choose the market index:',
        (['Nasdaq', 'Dow Jones','S&P 500', 'Nikkei', 'FTSE', 'Euro Stoxx', 'DAX', 'Ibovespa', 'S&P/BMV IPC', 'S&P/ASX', 'NSE']))

    if market == 'S&P 500':
        list_of_stocks = tickers_sp500

    if market == 'Nasdaq':
        list_of_stocks = tickers_nasdaq
        
    if market == 'Dow Jones':
        list_of_stocks = tickers_dow
    
    if market == 'FTSE':
        list_of_stocks = tickers_ftse

    if market == 'Euro Stoxx':
        list_of_stocks = tickers_euro_stoxx

    if market == 'DAX':
        list_of_stocks = tickers_dax
        
    if market == 'Ibovespa':
        list_of_stocks = tickers_ibov
        
    if market == 'S&P/BMV IPC':
        list_of_stocks = tickers_bmv

    if market == 'S&P/ASX':
        list_of_stocks = tickers_asx

    if market == 'NSE':
        list_of_stocks = tickers_nse

    if market == 'Nikkei':
        list_of_stocks = tickers_nikkei

    df = yf.download(tickers=list_of_stocks, period = '4y').ffill(axis=0)

    df = df.stack()


    if market == 'S&P 500':
        ticker = '^GSPC'
    
    if market == 'Nasdaq':
        ticker = '^IXIC'
        
    if market == 'Dow Jones':
        ticker = '^DJI'
        
    if market == 'FTSE':
        ticker = '^FTSE'

    if market == 'Euro Stoxx':
        ticker = '^STOXX50E'

    if market == 'DAX':
        ticker = '^GDAXI'
        
    if market == 'Ibovespa':
        ticker = '^BVSP'
    
    if market == 'S&P/BMV IPC':
        ticker = '^MXX'

    if market == 'S&P/ASX':
        ticker = '^AXJO'

    if market == 'NSE':
        ticker = '^NSEI'

    if market == 'Nikkei':
        ticker = '^N225'
        
        
    df1 = yf.download(ticker, period='10y')
    df1['Returns'] = df1['Adj Close'].pct_change()
    df1['High'] = df1['High'] - (df1['Close']-df1['Adj Close'])
    df1['Low'] = df1['Low'] - (df1['Close']-df1['Adj Close'])
    df1['Open'] = df1['Open'] - (df1['Close']-df1['Adj Close'])
    df1['Dates'] = df1.index

    vol_pl = 20
    df1['Vol'] = (np.round(df1['Returns'].rolling(window=vol_pl).std()*np.sqrt(252), 4))/np.sqrt(12)
    
    df1['Month_End'] = df1['Adj Close'].asfreq('BM').ffill()  # 'BM' para o último dia útil de cada mês
    df1['Month_End_Vol'] = df1['Vol'].asfreq('BM').ffill()

    df1 = df1.ffill()

    df1['Prev_Month_Close'] = df1['Month_End'].shift(1)
    df1['Prev_Month_Vol'] = df1['Month_End_Vol'].shift(1)
    
    # Calcular as bandas mensais usando o fechamento do mês anterior.
    df1['Upper_Band_1sd'] = df1['Prev_Month_Vol'] * df1['Prev_Month_Close'] + df1['Prev_Month_Close']
    df1['Lower_Band_1sd'] = df1['Prev_Month_Close'] - df1['Prev_Month_Vol'] * df1['Prev_Month_Close']
    
    df1['Upper_Band_2sd'] = (2*df1['Prev_Month_Vol']) * df1['Prev_Month_Close'] + df1['Prev_Month_Close']
    df1['Lower_Band_2sd'] = df1['Prev_Month_Close'] - (2*df1['Prev_Month_Vol']) * df1['Prev_Month_Close']
    
    
    # Preencher os valores faltantes.
    df1 = df1.ffill()
    

    # Converter a coluna 'Date' para datetime se ainda não for
    df1['Date'] = df1['Dates']

    #df1 = df1.fillna(method='ffill', axis=0)
    #df1.dropna(inplace=True)
    
    # Criar o texto de hover com o formato correto
    hovertext = []
    for i in range(len(df1)):
        hovertext.append('Date: {}<br>Open: {:.2f}<br>High: {:.2f}<br>Low: {:.2f}<br>Close: {:.2f}'.format(
            df1.iloc[i]['Date'].strftime('%Y-%m-%d'), df1.iloc[i]['Open'], df1.iloc[i]['High'], 
            df1.iloc[i]['Low'], df1.iloc[i]['Adj Close']))

    # Criar a figura com subplots
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    
    # Adicionar o gráfico OHLC ao subplot
    fig.add_trace(go.Candlestick(x=df1['Date'],
                          open=df1['Open'],
                          high=df1['High'],
                          low=df1['Low'],
                          close=df1['Adj Close'],
                          hovertext=hovertext,
                          hoverinfo='text',
                          name=market,
                          increasing_line_color='blue', decreasing_line_color='pink'), # Cores distintas
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=df1['Date'], y=df1['Upper_Band_1sd'], mode='lines', 
                             name='Supply Band 1sd', line=dict(color='green')))
    
    fig.add_trace(go.Scatter(x=df1['Date'], y=df1['Lower_Band_1sd'], mode='lines', 
                             name='Demand Band 1sd', line=dict(color='red')))
    
    # Adicionar as bandas com linha pontilhada ao gráfico
    fig.add_trace(go.Scatter(x=df1['Date'], y=df1['Upper_Band_2sd'], mode='lines', 
                             name='Supply Band 2sd', line=dict(dash='dash', color='green')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df1['Date'], y=df1['Lower_Band_2sd'], mode='lines', 
                             name='Demand Band 2sd', line=dict(dash='dash', color='red')), row=1, col=1)
    
    
    # Configurações adicionais do layout do gráfico
    fig.update_layout(
        title='Ticker',
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            # Ajustar o intervalo do eixo x para terminar um pouco depois da série
            range=[df1['Date'].min(), df1['Date'].max() + timedelta(days=30)]
        ),
        yaxis=dict(
            tickformat='.2f',
            title='Price'
        ),
        font=dict(size=12),
        legend=dict(
            y=1.02,
            x=1
        )
    )
    
    fig.update_xaxes(
                    rangeslider_visible=False,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=3, label="3y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )

    
    fig.update_yaxes(tickformat='.2f')
    
    fig.update_layout(title='OM Monthly S&D Volatility Zones')
    
    st.plotly_chart(fig, use_container_width=True, height=5000)

    ################## Stocks abova MA (50 and 200) #################
    
    df['50_sma'] = df.groupby(level=1)['Adj Close']\
                  .rolling(window=50, min_periods=1).mean()\
                  .reset_index(level=0, drop=True)

    df['200_sma'] = df.groupby(level=1)['Adj Close']\
                  .rolling(window=200, min_periods=1).mean()\
                  .reset_index(level=0, drop=True)

    df['above_50_sma'] = df.apply(lambda x: 1 if (x['Adj Close'] > x['50_sma'])
                                          else 0, axis=1)

    df['above_200_sma'] = df.apply(lambda x: 1 if (x['Adj Close'] > x['200_sma'])
                                           else 0, axis=1)

    above_50 = round(((df.groupby(level=0)['above_50_sma'].sum()/len(list_of_stocks))*100),2)

    above_50 = above_50[above_50>0]
    
    above_200 = round(((df.groupby(level=0)['above_200_sma'].sum()/len(list_of_stocks))*100),2)
    
    above_200 = above_200[above_200>0]

    ################## Range High_Low #################

    # Definir a janela de rolagem
    window = 200
    
    # Identificar novas máximas e mínimas com base em uma janela móvel de 200 dias
    df['Rolling High'] = df.groupby(level=1)['High'].transform(lambda x: x.rolling(window, min_periods=1).max())
    
    df['Rolling Low'] = df.groupby(level=1)['Low'].transform(lambda x: x.rolling(window, min_periods=1).min())
    
    dist_low = (df['Close'] - df['Rolling Low'])
    
    dist_low[dist_low < 0] = 0
    
    df['RHL'] = dist_low/(df['Rolling High'] - df['Rolling Low'])
    
    rhl = df['RHL'].groupby(level='Date').mean() * 100

    ################# New-High New-Low #################

    # Definir a janela de rolagem
    window = 20
    
    # Identificar novas máximas e mínimas com base em uma janela móvel de 260 dias
    df['Rolling High'] = df.groupby(level=1)['High'].transform(lambda x: x.rolling(window, min_periods=1).max())
    df['New High'] = df['High'] == df['Rolling High']
    
    df['Rolling Low'] = df.groupby(level=1)['Low'].transform(lambda x: x.rolling(window, min_periods=1).min())
    df['New Low'] = df['Low'] == df['Rolling Low']
    
    
    # Calcular o número de novas máximas e mínimas por data
    daily_highs_lows = df.groupby(level='Date').agg({'New High': 'sum', 'New Low': 'sum'})
    
    # Calcular o Record High Percent
    daily_highs_lows['Record High Percent'] = daily_highs_lows['New High']/len(list_of_stocks)
    
    daily_highs_lows['Record Low Percent'] = daily_highs_lows['New Low']/len(list_of_stocks)
    
    
    # Calcular a média móvel simples de 10 dias do Record High Percent
    daily_highs_lows['High-Low Index'] = (daily_highs_lows['Record High Percent']*100)-(daily_highs_lows['Record Low Percent']*100)

    daily_highs_lows['High-Low Index'] = daily_highs_lows['High-Low Index'].ffill(axis=0)
    daily_highs_lows['Date']= daily_highs_lows.index

    ################# S&D Volume #######################
    
    df['Close Change'] = df.groupby(level=1)['Adj Close'].pct_change(10)
        
    # Identifique as emissões avançadas e em declínio
    df['Advancing'] = df['Close Change'] > 0.015
    df['Declining'] = df['Close Change'] < -0.015
    
    # Calcule o volume diário para emissões avançadas e em declínio
    df['Advancing Volume'] = df['Volume'] * df['Advancing']
    df['Declining Volume'] = df['Volume'] * df['Declining']
    
    # Faça a soma de 10 dias dos volumes
    rolling_adv_vol = df.groupby(level=0)['Advancing Volume'].rolling(window=10).sum().reset_index(level=0, drop=True)
    rolling_dec_vol = df.groupby(level=0)['Declining Volume'].rolling(window=10).sum().reset_index(level=0, drop=True)
    
    # Agora, divida a soma de 10 dias do volume avançado pelo volume em declínio para obter o indicador
    v_r = rolling_adv_vol.groupby(level=0).sum() / (rolling_dec_vol.groupby(level=0).sum()+rolling_adv_vol.groupby(level=0).sum())

    v_r = v_r*100

    v_r = v_r.ffill(axis=0)
    
    ################# McClellan Oscillator #######################
    
    # Calcule a mudança diária no preço de fechamento
    df['Close Change'] = df.groupby(level=1)['Adj Close'].pct_change()
    
    # Identifique as emissões avançadas e em declínio
    df['Advancing'] = df['Close Change'] > 0
    df['Declining'] = df['Close Change'] < 0
    
    # Substituindo True e False por 1 e 0, respectivamente
    df['Advancing'] = df['Advancing'].astype(int)
    df['Declining'] = df['Declining'].astype(int)
    
    # Calcular a diferença diária entre avanços e declínios
    # Primeiro, somamos as emissões avançadas e em declínio para cada dia
    daily_advances = df.groupby(level=0)['Advancing'].sum()
    daily_declines = df.groupby(level=0)['Declining'].sum()
    
    # Depois calculamos a diferença para cada dia
    net_advances = daily_advances - daily_declines
    
    # Calcular a EMA de 19 dias para Net Advances
    mco_19ema = net_advances.ewm(span=19, adjust=False).mean()
    
    # Calcular a EMA de 39 dias para Net Advances
    mco_39ema = net_advances.ewm(span=39, adjust=False).mean()
    
    # Calcular o McClellan Oscillator
    mco = mco_19ema - mco_39ema
    
    
    ###############################################################################
       
    def last_20_80(value):
        if value <= 20:
            reading = 'Strong Bull'
        
        if 20 <= value <= 40:
            reading = 'Bull'
        
        if 40 < value <= 60:
            reading = 'Neutral'
        
        if 60 <= value < 80:
            reading = 'Bear'
        
        if value >= 80:
            reading = 'Strong Bear'
            
        return reading

    def last_30_70(value):
        if value <= 30:
            reading = 'Strong Bull'
        
        if 30 <= value <= 40:
            reading = 'Bull'
        
        if 40 < value <= 60:
            reading = 'Neutral'
        
        if 60 <= value < 70:
            reading = 'Bear'
        
        if value >= 70:
            reading = 'Strong Bear'
            
        return reading
    

    def last_nhnl(value):
        if value <= -40:
            reading = 'Strong Bull'

        if -40 < value <= -20:
            reading = 'Bull'

        if -20 < value <= 20:
            reading = 'Neutral'

        if 20 <= value < 40:
            reading = 'Bear'
        
        if value >= 40:
            reading = 'Strong Bear'
        
        return reading


    def last_vr(value):
        if value < 50:
            reading = 'Bear'
        
        if value > 50:
            reading = 'Bull'
        
        return reading


    def last_mco(value):
        if np.percentile(mco, 10) > value:
            reading = 'Bull'
        
        if np.percentile(mco, 10) < value < 0:
            reading = 'Bear'
        
        if 0 < value < np.percentile(mco, 90):
            reading = 'Bull'

        if value > np.percentile(mco, 90):
            reading = 'Bear'
        
        return reading

    
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
    
        last_value_ma50 = "{:.2f}%".format(above_50.iloc[-1])
        
        last_signal = last_20_80(above_50.iloc[-1])
        
        st.metric("Stocks above 50 SMA", last_value_ma50)
        st.text(last_signal)

    with col2:

        last_value_ma200 = "{:.2f}%".format(above_200.iloc[-1])
        
        last_signal = last_20_80(above_200.iloc[-1])
        
        st.metric("Stocks above 200 SMA", last_value_ma200)
        st.text(last_signal)

    with col3:

        last_value_rhl = "{:.2f}%".format(rhl.iloc[-1])
        
        last_signal = last_20_80(rhl.iloc[-1])
        
        st.metric("Range High-Low", last_value_rhl)
        st.text(last_signal)

    with col4:

        lat_value_nhnl =  "{:.2f}%".format(daily_highs_lows['High-Low Index'].iloc[-1])
        
        last_signal = last_nhnl(daily_highs_lows['High-Low Index'].iloc[-1])
        
        st.metric("New Highs - New Lows Index", lat_value_nhnl)
        st.text(last_signal)

    with col5:

        last_value_vr = "{:.2f}%".format(v_r.iloc[-1])
        
        last_signal = last_vr(v_r.iloc[-1])
        
        st.metric("S&D Volume", last_value_vr)
        st.text(last_signal)

    with col6:

        last_value_mco = "{:.2f}".format(mco.iloc[-1])        
        
        last_signal = last_mco(mco.iloc[-1])
        
        st.metric("McClellan Oscillator", last_value_mco)
        st.text(last_signal)
    

    
    #################################################################################
    col1, col2 = st.columns(2)

    with col1:

        mkt_50 = pd.DataFrame({'Value':above_50,
                              'Date':above_50.index})
        
        fig = px.line(mkt_50, x='Date', y='Value', title='Stocks Above 50-Day SMA')
        
        fig.update_xaxes(
                        rangeslider_visible=True,
                        rangeselector=dict(
                            buttons=list([
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(count=3, label="3y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        )
                    )
        
        fig.update_yaxes(ticksuffix="%")
    
        fig.update_layout( width=600,  # Largura do gráfico
            height=500  # Altura do gráfico
        )

        # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=20, line_dash="dash", line_color="gray")

        # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=40, line_dash="dash", line_color="gray")

        # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=60, line_dash="dash", line_color="gray")
        
        # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=80, line_dash="dash", line_color="gray")
    
        st.plotly_chart(fig)


    with col2:

        mkt_200 = pd.DataFrame({'Value':above_200,
                      'Date':above_200.index})

        fig = px.line(mkt_200, x='Date', y='Value', title='Stocks Above 200-Day SMA')
        
        fig.update_xaxes(
                        rangeslider_visible=True,
                        rangeselector=dict(
                            buttons=list([
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(count=3, label="3y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        )
                    )
        
        fig.update_yaxes(ticksuffix="%")

        fig.update_layout( width=600,  # Largura do gráfico
            height=500  # Altura do gráfico
        )

            # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=20, line_dash="dash", line_color="gray")

        # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=40, line_dash="dash", line_color="gray")

        # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=60, line_dash="dash", line_color="gray")
        
        # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=80, line_dash="dash", line_color="gray")
    
        st.plotly_chart(fig)


    col3, col4 = st.columns(2)

    with col3:       
        
        df_rhl = pd.DataFrame({'Range High-Low':rhl,
                       'Date': rhl.index})
    
        fig = px.line(df_rhl, x='Date', y='Range High-Low', title='20-week Range High-Low')
        
        fig.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=3, label="3y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )
        
        fig.update_yaxes(tickformat=".2f", ticksuffix="%")
        
        # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=30, line_dash="dash", line_color="gray")

        # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=40, line_dash="dash", line_color="gray")

        # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=60, line_dash="dash", line_color="gray")
        
        # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=70, line_dash="dash", line_color="gray")
    
    
        fig.update_layout( width=600,  # Largura do gráfico
                height=500  # Altura do gráfico
            )
        
        st.plotly_chart(fig)
    
    with col4:        
        
        fig = px.line(daily_highs_lows, x='Date', y='High-Low Index', title='20-day New Highs - New Lows')
        
        fig.update_xaxes(
                        rangeslider_visible=True,
                        rangeselector=dict(
                            buttons=list([
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(count=3, label="3y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        )
                    )
        
        fig.update_yaxes(tickformat=".2f", ticksuffix="%")
        
        # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=40, line_dash="dash", line_color="gray")

        # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=20, line_dash="dash", line_color="gray")

        # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=-20, line_dash="dash", line_color="gray")
        
        # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=-40, line_dash="dash", line_color="gray")
    
        fig.update_layout( width=600,  # Largura do gráfico
                height=500  # Altura do gráfico
            )
        
        st.plotly_chart(fig)

    col5, col6 = st.columns(2)
    
    with col5:
    
        v_r_10 = pd.DataFrame({'Value':v_r,
                          'Date':v_r.index})
    
        fig = px.line(v_r_10, x='Date', y='Value', title='10-day Supply and Demand Volume')
        
        fig.update_xaxes(
                        rangeslider_visible=True,
                        rangeselector=dict(
                            buttons=list([
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(count=3, label="3y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        )
                    )
        
        fig.update_yaxes(tickformat=".2f", ticksuffix="%")
        
        # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=50, line_dash="dash", line_color="gray")
    
        fig.update_layout( width=600,  # Largura do gráfico
                height=500  # Altura do gráfico
            )
        
        st.plotly_chart(fig)
    
    with col6:

        mco_plot = pd.DataFrame({'Value':mco,
                      'Date':mco.index})

        fig = px.line(mco_plot, x='Date', y='Value', title='McClellan Ocslillator')
        
        fig.update_xaxes(
                        rangeslider_visible=True,
                        rangeselector=dict(
                            buttons=list([
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(count=3, label="3y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        )
                    )
        
        fig.update_yaxes(tickformat=".2f")
        
        # Adicionando a linha pontilhada cinza no y=0
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.add_hline(y=np.percentile(mco, 90), line_dash="dash", line_color="gray")

        fig.add_hline(y=np.percentile(mco, 10), line_dash="dash", line_color="gray")

        fig.update_layout( width=600,  # Largura do gráfico
            height=500  # Altura do gráfico
        )
    
        st.plotly_chart(fig)
        
if selected == 'Risk & Volatility':
    st.title('Volatillity Momentum')
    st.markdown('##')

    
    def vol_heatmap(df, classe):
        janelas = ['1W', '2W', '1M', '3M', '6M', 'YTD', '1Y', '2Y']
        matriz = pd.DataFrame(columns=janelas, index=df.columns)
         
        vol_pl = 20
        hist_vol = (np.round(df.ffill().pct_change(1).rolling(window=vol_pl).std()*np.sqrt(252), 4))
    
        
        df_2y = hist_vol.ffill().diff(520).iloc[-1]
        df_1y = hist_vol.ffill().diff(260).iloc[-1]
        start_of_year = hist_vol.index[df.index.year == hist_vol.index[-1].year][0]
        df_ytd = (hist_vol.ffill().loc[hist_vol.index[-1]] - hist_vol.ffill().loc[start_of_year])
        df_6m = hist_vol.ffill().diff(130).iloc[-1]
        df_3m = hist_vol.ffill().diff(60).iloc[-1]
        df_1m = hist_vol.ffill().diff(20).iloc[-1]
        df_2w = hist_vol.ffill().diff(10).iloc[-1]
        df_1w = hist_vol.ffill().diff(5).iloc[-1]
    
        matriz['1W'] = df_1w
        matriz['2W'] = df_2w
        matriz['1M'] = df_1m
        matriz['3M'] = df_3m
        matriz['6M'] = df_6m
        matriz['YTD'] = df_ytd
        matriz['1Y'] = df_1y
        matriz['2Y'] = df_2y
        
        annotations = []
        for y, row in enumerate(matriz.values):
            for x, val in enumerate(row):
                annotations.append({
                    "x": matriz.columns[x],
                    "y": matriz.index[y],
                    "font": {"color": "black"},
                    "text": f"{val:.2%}",
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False
                })
        
        fig = go.Figure(data=go.Heatmap(
                        z=matriz.values,
                        x=matriz.columns.tolist(),
                        y=matriz.index.tolist(),
                        colorscale='RdYlGn',
                        zmin=matriz.values.min(), zmax=matriz.values.max(),  # para garantir que o 0 seja neutro em termos de cor
                        hoverongaps = False,
            text=matriz.apply(lambda x: x.map(lambda y: f"{y:.2%}")),
            hoverinfo='y+x+text',
            showscale=True,
            colorbar_tickformat='.2%'
        ))
        
        
        fig.update_layout(title=classe, annotations=annotations, width=600,  # Largura do gráfico
height=600  # Altura do gráfico
)
        st.plotly_chart(fig)


    col1, col2 = st.columns(2)

    with col1:
        vol_heatmap(df_acoes, "Stocks")
    with col2:
        vol_heatmap(df_moedas, "Currencies (Futures)")

    col3, col4 = st.columns(2)

    with col3:
        vol_heatmap(df_commodities, "Commodities")
    with col4:
        vol_heatmap(df_rf, "Fixed Income")

    
    col5, col6 = st.columns(2)

    with col5:
        vol_heatmap(df_factors, "Factors")
    with col6:
        vol_heatmap(df_sectors, "US Sectors")

    col7, col8 = st.columns(2)

    with col7:
        vol_heatmap(df_crypto, "Crypto")

    
    all_assets_list = all_assets.columns.tolist()
    
    all_assets_list.remove('S&P 500')
    all_assets_list.insert(0, 'S&P 500')

    asset = st.selectbox(
        'Choose the asset:',
        (all_assets_list))

    original_names = tickers_acoes + tickers_moedas + tickers_commodities + tickers_rf + tickers_crypto + tikckers_factors + tickers_sectors
    trasformed_names = names_acoes + names_moedas + names_commodities + names_rf + names_crypto + names_factors + names_sectors
    
    # Crie um dicionário de correspondência
    correspondencia = dict(zip(trasformed_names, original_names))

    # Nome que você deseja encontrar a correspondência
    nome_procurado = asset
    
    # Verificar a correspondência
    correspondencia_original = correspondencia.get(nome_procurado)

    df_asset = yf.download(correspondencia_original, period='25y')['Adj Close']
    
    hist_vol_asset_20d = (np.round(df_asset.ffill().pct_change(1).rolling(window=20).std()*np.sqrt(252), 4))
    
    hist_vol_asset_60d = (np.round(df_asset.ffill().pct_change(1).rolling(window=60).std()*np.sqrt(252), 4))
        
    hist_vol_asset_260d = (np.round(df_asset.ffill().pct_change(1).rolling(window=260).std()*np.sqrt(252), 4))


    df_hist_vol_asset = pd.concat([hist_vol_asset_20d, hist_vol_asset_60d, hist_vol_asset_260d], axis=1).dropna()
    df_hist_vol_asset.columns = ['1 month', '3 month', '1 year']
    df_hist_vol_asset['Date'] = df_hist_vol_asset.index
    

    fig = px.line(df_hist_vol_asset, x='Date', y=['1 month', '3 month', '1 year'], title='Historical Volatility')
    
    fig.update_xaxes(
    rangeslider_visible=False,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

    fig.update_yaxes(tickformat=".2%")
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('##')

    col1, col2 = st.columns(2)

    with col1:

        all_assets_list_1 = all_assets.columns.tolist()
    
        all_assets_list_1.remove('Ibovespa')
        all_assets_list_1.insert(0, 'Ibovespa')
    
        asset_1 = st.selectbox(
            'Choose the first asset:',
            (all_assets_list_1))

    with col2:

        all_assets_list_2 = all_assets.columns.tolist()
    
        all_assets_list_2.remove('EM Equity')
        all_assets_list_2.insert(0, 'EM Equity')
    
        asset_2 = st.selectbox(
            'Choose the second asset:',
            (all_assets_list_2))

    
    original_names = tickers_acoes + tickers_moedas + tickers_commodities + tickers_rf + tickers_crypto + tikckers_factors + tickers_sectors
    trasformed_names = names_acoes + names_moedas + names_commodities + names_rf + names_crypto + names_factors + names_sectors
    
    # Crie um dicionário de correspondência
    correspondencia = dict(zip(trasformed_names, original_names))

    # Nome que você deseja encontrar a correspondência
    nome_procurado_1 = asset_1

    nome_procurado_2 = asset_2
    
    # Verificar a correspondência
    correspondencia_original_1 = correspondencia.get(nome_procurado_1)

    correspondencia_original_2 = correspondencia.get(nome_procurado_2)

    df_asset_1 = yf.download(correspondencia_original_1, period='25y')['Adj Close']

    df_asset_2 = yf.download(correspondencia_original_2, period='25y')['Adj Close']
    

    # Volatilidade de 20, 60 e 260 dias

    hist_vol_1_20d = (np.round(df_asset_1.ffill().pct_change(1).rolling(window=20).std()*np.sqrt(252), 4))
    
    hist_vol_2_20d = (np.round(df_asset_2.ffill().pct_change(1).rolling(window=20).std()*np.sqrt(252), 4))

    
    hist_vol_1_60d = (np.round(df_asset_1.ffill().pct_change(1).rolling(window=60).std()*np.sqrt(252), 4))
    
    hist_vol_2_60d = (np.round(df_asset_2.ffill().pct_change(1).rolling(window=60).std()*np.sqrt(252), 4))

    
    hist_vol_1_260d = (np.round(df_asset_1.ffill().pct_change(1).rolling(window=260).std()*np.sqrt(252), 4))
    
    hist_vol_2_260d = (np.round(df_asset_2.ffill().pct_change(1).rolling(window=260).std()*np.sqrt(252), 4))

    

    #vol_pl = 20
    #hist_vol_1 = (np.round(all_assets[asset_1].ffill().pct_change(1).rolling(window=vol_pl).std()*np.sqrt(252), 4))

    #hist_vol_2 = (np.round(all_assets[asset_2].ffill().pct_change(1).rolling(window=vol_pl).std()*np.sqrt(252), 4))

    #spread = (hist_vol_1 -  hist_vol_2)

    spread_20d = pd.concat([hist_vol_1_20d, hist_vol_2_20d], axis=1).dropna()
    spread_20d = spread_20d.iloc[:, 0] - spread_20d.iloc[:, 1]

    spread_60d = pd.concat([hist_vol_1_60d, hist_vol_2_60d], axis=1).dropna()
    spread_60d = spread_60d.iloc[:, 0] - spread_60d.iloc[:, 1]

    spread_260d = pd.concat([hist_vol_1_260d, hist_vol_2_260d], axis=1).dropna()
    spread_260d = spread_260d.iloc[:, 0] - spread_260d.iloc[:, 1]

    df_spread = pd.concat([spread_20d, spread_60d, spread_260d], axis=1).dropna()
    df_spread.columns = ['1 month', '3 month', '1 year']
    df_spread['Date'] = df_spread.index
    

    #df_spread = pd.DataFrame({'1 month':spread, 'Date':spread.index})
    

    fig = px.line(df_spread, x='Date', y=['1 month', '3 month', '1 year'], title='Volatility Spread - '+asset_1+str(' x ')+asset_2)
    
    fig.update_xaxes(
    rangeslider_visible=False,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

    fig.update_yaxes(tickformat=".2%")
    
    st.plotly_chart(fig, use_container_width=True)



    # Rolling Beta

    def rolling_beta(a1, a2, window):

        ret_a1 = a1.pct_change(1)    
        ret_a2 = a2.pct_change(1)

        df = pd.concat([ret_a1, ret_a2], axis=1).dropna()
        
        # Calcule o rolling beta usando a função rolling do pandas
        rolling_cov = df.iloc[:, 0].rolling(window).cov(df.iloc[:, 1])
        rolling_var = df.iloc[:, 1].rolling(window).var()
        
        # Calcule o beta dividindo a covariância pelo valor
        rolling_beta = rolling_cov / rolling_var

        return round(rolling_beta,2)
        
        
    beta_20d = rolling_beta(df_asset_1, df_asset_2, 20)

    beta_60d = rolling_beta(df_asset_1, df_asset_2, 60)

    beta_260d = rolling_beta(df_asset_1, df_asset_2, 260)

    df_beta = pd.concat([beta_20d, beta_60d, beta_260d], axis=1).dropna()
    df_beta.columns = ['1 month', '3 month', '1 year']
    df_beta['Date'] = df_beta.index
    

    fig = px.line(df_beta, x='Date', y=['1 month', '3 month', '1 year'], title='Rolling Beta - '+asset_1+str(' x ')+asset_2)
    
    fig.update_xaxes(
    rangeslider_visible=False,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

    fig.update_yaxes(tickformat=".2f")
    
    st.plotly_chart(fig, use_container_width=True)


    # Rolling Correlation

    def rolling_corr(a1, a2, window):

        ret_a1 = a1.pct_change(1)    
        ret_a2 = a2.pct_change(1)

        df = pd.concat([ret_a1, ret_a2], axis=1).dropna()

        rolling_corr = df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1])

        return round(rolling_corr, 2)


    corr_20d = rolling_corr(df_asset_1, df_asset_2, 20)

    corr_60d = rolling_corr(df_asset_1, df_asset_2, 60)

    corr_260d = rolling_corr(df_asset_1, df_asset_2, 260)

    df_corr = pd.concat([corr_20d, corr_60d, corr_260d], axis=1).dropna()
    df_corr.columns = ['1 month', '3 month', '1 year']
    df_corr['Date'] = df_corr.index
    

    fig = px.line(df_corr, x='Date', y=['1 month', '3 month', '1 year'], title='Rolling Correlation - '+asset_1+str(' x ')+asset_2)
    
    fig.update_xaxes(
    rangeslider_visible=False,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

    fig.update_yaxes(tickformat=".2f")
    
    st.plotly_chart(fig, use_container_width=True)

    # Relative Performance

    r_p = (df_asset_1/df_asset_2).dropna()

    df_rp = pd.DataFrame({'Value':r_p,
                          'Date':r_p.index})

    fig = px.line(df_rp, x='Date', y='Value', title='Relative Performance - '+asset_1+str(' x ')+asset_2)
    
    fig.update_xaxes(
    rangeslider_visible=False,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

    fig.update_yaxes(tickformat=".2f")
    
    st.plotly_chart(fig, use_container_width=True)

    

    
                


    

