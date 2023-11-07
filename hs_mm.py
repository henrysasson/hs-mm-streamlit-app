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
    data = yf.download(tickers, period='4y')['Adj Close']
    return data


# Ações
tickers_acoes = ['^GSPC', '^IXIC', '^RUT', '^N225', '^FTSE', '^STOXX50E', '^GDAXI', '^BVSP', '^AXJO', '^MXX', '000001.SS', '^HSI', '^NSEI']
df_acoes = get_data(tickers_acoes).fillna(method='ffill', axis=0)
names_acoes = ['SPX', 'Nasdaq', 'Russel 2000', 'Nikkei', 'FTSE', 'Euro Stoxx', 'DAX', 'IBOV', 'S&P ASX', 'BMV', 'Shanghai', 'Hang Seng', 'NSE']
column_mapping = dict(zip(tickers_acoes, names_acoes))
df_acoes.rename(columns=column_mapping, inplace=True)

# Moedas
tickers_moedas = ['EURUSD=X', 'JPY=X', 'GBPUSD=X', 'BRL=X', 'AUDUSD=X', 'MXN=X']
df_moedas = get_data(tickers_moedas).fillna(method='ffill', axis=0)
names_moedas = ['EURUSD', 'USDJPY', 'GBPUSD', 'USDBRL', 'AUDUSD', 'MXNUSD']
column_mapping = dict(zip(tickers_moedas, names_moedas))
# Renomeie as colunas
df_moedas.rename(columns=column_mapping, inplace=True)

# Commodities
tickers_commodities = ['DBC', 'GSG', 'USO', 'GLD', 'SLV', 'DBA', 'U-UN.TO', 'BDRY']
df_commodities = get_data(tickers_commodities).fillna(method='ffill', axis=0)
names_commodities = ['DBC', 'GSG', 'USO', 'GLD', 'SLV', 'DBA', 'U.UN', 'BDRY']
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
tickers_crypto = ['BTC-USD', 'ETH-USD']
df_crypto = get_data(tickers_crypto).fillna(method='ffill', axis=0)
names_crypto = ['BTCUSD', 'ETHUSD']
column_mapping = dict(zip(tickers_crypto, names_crypto))
# Renomeie as colunas
df_crypto.rename(columns=column_mapping, inplace=True)

# Todos os Ativos
all_assets = pd.concat([df_acoes, df_moedas, df_commodities, df_rf, df_crypto], axis=1).dropna()

options = ['Returns Heatmap', 'Correlation Matrix',  'Market Directionality', 'Macro Indicators', 'Positioning',  'Technical Analysis']
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
        returns_heatmap(df_moedas, "Currencies")

    col3, col4 = st.columns(2)

    with col3:
        returns_heatmap(df_commodities, "Commodities")
    with col4:
        returns_heatmap(df_rf, "Fixed Income")


# # Matriz de Correlação

if selected == 'Correlation Matrix':
    st.title('Correlation Matrix')
    st.markdown('##')
   
    def corr_matrix(df, janela, classe):
        matriz = df.pct_change()[-janela:].corr()
        
        if classe == str('Multi Asset'):
            
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
            
            st.plotly_chart(fig, use_container_width=True, height=800  # Altura do gráfico
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

        corr_matrix(df_moedas, lookback, 'Currencies')

    col3, col4 = st.columns(2)

    with col3:

        corr_matrix(df_commodities, lookback, 'Commodities')

    with col4:
    
        corr_matrix(df_rf, lookback, 'Fixed Income')


    corr_matrix(all_assets, lookback, 'Multi Asset')

    all_assets_list = all_assets.columns.tolist()
    
    all_assets_list.remove('SPX')
    all_assets_list.insert(0, 'SPX')

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

        if classe == 'Multi Asset':
            
            st.plotly_chart(fig, use_container_width=True)
    
        else:
            fig.update_layout( width=800,  # Largura do gráfico
    height=600  # Altura do gráfico
)

            st.plotly_chart(fig)


    lookback = st.number_input(label="Choose the lookback period", value=260)

    col1, col2 = st.columns(2)

    with col1:
    
        directional_indicator(df_acoes, lookback, 'Stocks')

    with col2:

        directional_indicator(df_moedas, lookback, 'Currencies')

    col3, col4 = st.columns(2)

    with col3:

        directional_indicator(df_commodities, lookback, 'Commodities')

    with col4:

        directional_indicator(df_rf, lookback, 'Fixed Income')



    directional_indicator(all_assets, lookback, 'Multi Asset')


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
        
        col1, col2, col3 = st.columns(3)

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
                rangeslider_visible=True,
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

            fig_yc.update_layout( width=500,  # Largura do gráfico
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
                rangeslider_visible=True,
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

            fig_hys.update_layout( width=500,  # Largura do gráfico
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



        with col3:
            T01Y = fred.get_series('DGS1').dropna()

            # Convert to DataFrame for plotting
            df_t1 = pd.DataFrame({'Date':T01Y.index,
                                'T01Y':T01Y.values,
                                 '12 MA': T01Y.rolling(260).mean().values})
            
            
            # Plot both 'Value' and '12M MA' on the same figure
            fig_t1y = px.line(df_t1, x='Date', y=['T01Y', '12 MA'], title='Hike Indicator')
            
            fig_t1y.update_xaxes(
            rangeslider_visible=True,
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

            fig_t1y.update_layout( width=500,  # Largura do gráfico
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
            
            fed_bs = fred.get_series('QBPBSTAS')
            tga = fred.get_series('WTREGEN')
            rrp = fred.get_series('RRPONTSYD')


            df_fed_liq = pd.DataFrame({
                        'TGA':tga,
                        'RRP':rrp})

            df_fed_liq.dropna(inplace=True)


            # Convert to DataFrame for plotting
            fed_liq = pd.DataFrame({
                    'Date': df_fed_liq.index,
                    'Fed Liquidity': df_fed_liq.sum(axis=1).values
                
                }).dropna()

            # Plot both 'Value' and '12M MA' on the same figure
            fig_fed_liq = px.line(fed_liq, x='Date', y='Fed Liquidity', title=' Fed Liquidity (TGA + RRP)')

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

            fig_nfpr.update_layout(height=600  # Altura do gráfico
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
            wilshire = fred.get_series('WILL5000PR').dropna()
            
            df = pd.concat([tax, wilshire], axis=1).dropna()
            df.columns = ['Federal government current tax receipts', 'Wilshire 5000']
    
            
            # Create a subplot with dual Y-axes
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Plot data
            fig.add_trace(go.Scatter(x=df.index, y=df['Federal government current tax receipts'], name="Tax Receipts"), secondary_y=False)
            fig.add_trace(go.Scatter(x=df.index, y=df['Wilshire 5000'], name="Wilshire 5000"), secondary_y=True)
            
            # Titles and labels
            fig.update_layout(title_text="Federal government current tax receipts")
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Tax Receipts", secondary_y=False)
            fig.update_yaxes(title_text="Wilshire 5000 Index", secondary_y=True)
            
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
        
        col1, col2, col3 = st.columns(3)
        
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

            fig_scl.update_layout( width=500,  # Largura do gráfico
        height=500  # Altura do gráfico
    )
        

            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            # Adicionar o símbolo de % ao eixo y
            fig_scl.update_yaxes(tickformat=".2%")
                
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

            fig_icc.update_layout( width=500,  # Largura do gráfico
        height=500  # Altura do gráfico
    )
            
                
            st.plotly_chart(fig_icc)

        
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
            
            fig_pmc.update_layout( width=500,  # Largura do gráfico
            height=500  # Altura do gráfico
            )
        

            st.plotly_chart(fig_pmc)

        
        col4, col5, col6 = st.columns(3)

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

            fig_ef.update_layout( width=500,  # Largura do gráfico
            height=500  # Altura do gráfico
            )
    

            st.plotly_chart(fig_ef)

        
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

            fig_crf.update_layout( width=500,  # Largura do gráfico
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
            
            fig_inad.update_layout( width=500,  # Largura do gráfico
            height=500  # Altura do gráfico
            )
            
            
            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            # Adicionar o símbolo de % ao eixo y
            fig_inad.update_yaxes(ticksuffix="%")

        
            st.plotly_chart(fig_inad)

        st.subheader('Economic Activity')
        
        col7, col8, col9 = st.columns(3)

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
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(count=7, label="7y", step="year", stepmode="backward"),
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(count=15, label="15y", step="year", stepmode="backward"),
                        dict(count=20, label="20y", step="year", stepmode="todate"),            
                        dict(step="all")
                    ])
                )
            )
            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            fig_inf.update_yaxes(tickformat=".2f", ticksuffix="%")
            fig_inf.update_layout( width=600,  # Largura do gráfico
        height=600,
        # Altura do gráfico
    )
    
            fig_inf.update_layout( width=500,  # Largura do gráfico
        height=500  # Altura do gráfico
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
            
            fig_ibc.update_layout( width=500,  # Largura do gráfico
            height=500  # Altura do gráfico
            )
            
            st.plotly_chart(fig_ibc)


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
            
            fig_pib.update_layout( width=500,  # Largura do gráfico
            height=500  # Altura do gráfico
            )
            
            st.plotly_chart(fig_pib)



if selected == 'Positioning':
    st.title('Positioning')
    st.markdown('Commitment of Traders')
    st.markdown('##')

    category = [
    "CRYPTO CURRENCIES", "CRYPTO CURRENCIES", "CRYPTO CURRENCIES", "CRYPTO CURRENCIES",
    "CURRENCIES", "CURRENCIES", "CURRENCIES", "CURRENCIES", "CURRENCIES", "CURRENCIES",
    "CURRENCIES", "CURRENCIES", "CURRENCIES", "CURRENCIES", "CURRENCIES",
    "ENERGIES", "ENERGIES", "ENERGIES", "ENERGIES",
    "EQUITIES", "EQUITIES", "EQUITIES", "EQUITIES", "EQUITIES - OTHER",
    "EQUITIES - OTHER", "EQUITIES - OTHER", "EQUITIES - OTHER", "EQUITIES - OTHER",
    "EQUITIES - OTHER", "EQUITIES - OTHER", "EQUITIES - OTHER", "EQUITIES - OTHER",
    "EQUITIES - OTHER", "EQUITIES - OTHER",
    "FIXED INCOME", "FIXED INCOME", "FIXED INCOME",
    "FIXED INCOME - OTHER", "FIXED INCOME - OTHER", "FIXED INCOME - OTHER",
    "FIXED INCOME - OTHER", "FIXED INCOME - OTHER", "FIXED INCOME - OTHER",
    "GRAINS", "GRAINS", "GRAINS", "GRAINS", "GRAINS", "GRAINS",
    "METALS", "METALS", "METALS", "METALS", "METALS", "METALS", "METALS",
    "SOFTS", "SOFTS", "SOFTS", "SOFTS", "SOFTS", "SOFTS"
]

    symbols = [
        "BTC", "MBT", "ETH", "MET", "6E", "DX", "6C", "6B", "6S", "6J", "6A",
        "6N", "6L", "6M", "6Z", "BZ", "CL", "RB", "NG", "ES", "YM", "NQ",
        "RTY", "NKD", "VX", "ES CON", "ES ENE", "ES FIN", "ES HCI", "ES UTI",
        "ES MAT", "ES IND", "ES TEC", "MME", "ZT", "ZF", "ZN", "GE",
        "SR1", "SR3", "ZQ", "TN", "UB", "ZW", "ZO", "ZC", 
        "ZM", "ZS", "ZL", "GC", "SI", "PL", "PA", "HG", 
        "AUP", "HRC", "CC", "KC", "CT", "LBS", "OJ", "SB"
    ]
    
    
    market = [
        "BITCOIN", "MICRO BITCOIN", "ETHEREUM", "MICRO ETHER", "EURO FX", "DX USD INDEX",
        "CANADIAN DOLLAR", "BRITISH POUND", "SWISS FRANC", "JAPANESE YEN", "AUSTRALIAN DOLLAR",
        "NEW ZEALAND DOLLAR", "BRAZILIAN REAL", "MEXICAN PESO", "SOUTH AFRICAN RAND",
        "BRENT CRUDE OIL", "CRUDE OIL", "GASOLINE RBOB", "NATURAL GAS", "S&P 500", "DOW JONES", "NASDAQ",
        "RUSSELL 2000", "NIKKEI", "VIX", "EMINI S&P CONSU STAPLES INDEX", "EMINI S&P ENERGY INDEX",
        "EMINI S&P FINANCIAL INDEX", "EMINI S&P HEALTH CARE INDEX", "EMINI S&P UTILITIES INDEX",
        "E-MINI S&P MATERIALS INDEX", "E-MINI S&P INDUSTRIAL INDEX", "E-MINI S&P TECHNOLOGY INDEX",
        "MSCI EM INDEX", "2-YEAR NOTES", "5-YEAR NOTES", "10-YEAR NOTES", "GE EURODOLLARS",
        "SR1 SECURED OVERNIGHT FINANCING RATE (1-MONTH)", "SR3 SECURED OVERNIGHT FINANCING RATE (3-MONTH)",
        "ZQ FED FUNDS", "TN ULTRA 10-YEAR NOTES", "UB ULTRA 30-YEAR BONDS", "WHEAT",
        "OATS", "CORN", "SOYBEAN MEAL", "SOYBEANS", "SOYBEAN OIL", "GOLD", "SILVER", "PLATINUM",
        "PALLADIUM", "COPPER", "ALUMINIUM", "STEEL", "SUGAR", "COCOA", "COFFEE", "ORANGE JUICE",
        "COTTON", "LUMBER"
    ]
    
    
    
    contract_code = [
        "133741", "133742", "146021", "146022", "099741", "098662",
        "090741", "096742", "092741", "097741", "232741", "112741",
        "102741", "095741", "122741", "06765T", "06765A", "111659", "0233AX",
        "13874A", "124606", "209742", "239742", "240741", "1170E1",
        "138748", "138749", "13874C", "13874E", "13874J", "13874H",
        "13874F", "13874J", "244042", "042601", "044601", "043602",
        "132741", "134742", "134741", "045601", "043607", "020604",
        "001602", "004603", "002602", "026603",
        "005602", "007601", "088691", "084691", "076651", "075651",
        "085692", "191693", "192651", "080732", "073732", "083731",
        "040701", "033661", "058644"
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
    
    st.plotly_chart(fig_cot, width_wide = True, height = 1600) 

    
def rank(series):
    # Calcula o percentil do último valor
    percentile = (np.searchsorted(np.sort(series), series.iloc[-1]) + 1) / len(series) * 100
    return round(percentile)

col1, col2, col3 = st.columns(3)

with col1:

    # Large Specs
    try:
        st.metric(label = 'Large Speculators Index', value = str(rank(df['Sum of Large Speculators'])))
    except:
        pass

with col2:

    # Small Specs
    try:
        st.metric(label = 'Small Speculators Index', value = str(rank(df['Sum of Small Speculators'])))
    except:
        pass

with col3:

    # Commercials
    try:
        st.metric(label = 'Commercials Index', value = str(rank(df['Sum of Commercials'])))
    except:
        pass


if selected == 'Technical Analysis':
    st.title('Technical Analysis')
    st.markdown('##')

    tickers_ibov = [
    "RRRP3.SA", "ALOS3.SA", "ALPA4.SA", "ABEV3.SA", "ARZZ3.SA", "ASAI3.SA", "AZUL4.SA",
    "B3SA3.SA", "BBSE3.SA", "BBDC3.SA", "BBDC4.SA", "BRAP4.SA", "BBAS3.SA", "BRKM5.SA",
    "BRFS3.SA", "BPAC11.SA", "CRFB3.SA", "BHIA3.SA", "CCRO3.SA", "CMIG4.SA", "CIEL3.SA",
    "COGN3.SA", "CPLE6.SA", "CSAN3.SA", "CPFE3.SA", "CMIN3.SA", "CVCB3.SA", "CYRE3.SA",
    "DXCO3.SA", "ELET3.SA", "ELET6.SA", "EMBR3.SA", "ENGI11.SA", "ENEV3.SA", "EGIE3.SA",
    "EQTL3.SA", "EZTC3.SA", "FLRY3.SA", "GGBR4.SA", "GOAU4.SA", "GOLL4.SA", "NTCO3.SA",
    "SOMA3.SA", "HAPV3.SA", "HYPE3.SA", "IGTI11.SA", "IRBR3.SA", "ITSA4.SA", "ITUB4.SA",
    "JBSS3.SA", "KLBN11.SA", "RENT3.SA", "LWSA3.SA", "LREN3.SA", "MGLU3.SA", "MRFG3.SA",
    "BEEF3.SA", "MRVE3.SA", "MULT3.SA", "PCAR3.SA", "PETR3.SA", "PETR4.SA", "RECV3.SA",
    "PRIO3.SA", "PETZ3.SA", "RADL3.SA", "RAIZ4.SA", "RDOR3.SA", "RAIL3.SA", "SBSP3.SA",
    "SANB11.SA", "SMTO3.SA", "CSNA3.SA", "SLCE3.SA", "SUZB3.SA", "TAEE11.SA", "VIVT3.SA",
    "TIMS3.SA", "TOTS3.SA", "UGPA3.SA", "USIM5.SA", "VALE3.SA", "VAMO3.SA", "VBBR3.SA",
    "WEGE3.SA", "YDUQ3.SA"
]

    tickers_bmv = [
    "MEGACPO.MX", "AC.MX", "ASURB.MX", "FEMSAUBD.MX", "CUERVO.MX", "KIMBERA.MX",
    "GRUMAB.MX", "GMEXICOB.MX", "GCC.MX", "BOLSAA.MX", "OMAB.MX", "GFNORTEO.MX",
    "BIMBOA.MX", "ALSEA.MX", "LABB.MX", "GAPB.MX", "GENTERA.MX", "TLEVISACPO.MX",
    "GCARSOA1.MX", "ALPEKA.MX", "LIVEPOLC-1.MX",
    "BBAJIOO.MX", "CEMEXCPO.MX", "AMXB.MX", "PINFRA.MX", "SITES1A-1.MX", "WALMEX.MX", "ORBIA.MX",
    "Q.MX", "RA.MX", "PE&OLES.MX", "CHDRAUIB.MX", "ELEKTRA.MX", "KOFUBL.MX"
]
  
    tickers_ftse = [
    "WTB.L", "WPP.L", "WEIR.L", "VOD.L", "UU.L", "UTG.L", "ULVR.L", "TW.L", "TSCO.L",
    "SVT.L", "STJ.L", "STAN.L", "SSE.L", "SPX.L", "SN.L", "SMT.L", "SMIN.L", "SMDS.L",
    "SKG.L", "SHEL.L", "SGRO.L", "SGE.L", "SDR.L", "SBRY.L", "RTO.L", "RS1.L", "RR.L",
    "RMV.L", "RKT.L", "RIO.L", "REL.L", "PSON.L", "PSH.L", "PRU.L", "PHNX.L", "OCDO.L",
    "NXT.L", "NWG.L", "NG.L", "MRO.L", "MNG.L", "MNDI.L", "MKS.L", "LSEG.L", "LLOY.L",
    "LGEN.L", "LAND.L", "KGF.L", "JD.L", "ITRK.L", "INF.L", "IMI.L", "IMB.L", "III.L",
    "IHG.L", "IAG.L", "HWDN.L", "HSBA.L", "HLN.L", "HLMA.L", "HL.L", "HIK.L", "GSK.L",
    "GLEN.L", "FRES.L", "FRAS.L", "FLTR.L", "FCIT.L", "EXPN.L", "ENT.L", "EDV.L",
    "DPLM.L", "DPH.L", "DGE.L", "DCC.L", "CTEC.L", "CRDA.L", "CPG.L", "CNA.L", "CCH.L",
    "BT-A.L", "BRBY.L", "BP.L", "BNZL.L", "BME.L", "BKG.L", "BEZ.L", "BDEV.L", "BATS.L",
    "BARC.L", "BA.L", "AZN.L", "AV.L", "AUTO.L", "ANTO.L", "AHT.L", "ADM.L", "ABF.L",
    "AAL.L", "AAF.L"
] 
    tickers_nasdaq = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "AVGO", "GOOGL", "GOOG", "TSLA", "COST",
    "ADBE", "PEP", "CSCO", "NFLX", "CMCSA", "TMUS", "AMD", "INTC", "INTU", "AMGN",
    "TXN", "HON", "QCOM", "AMAT", "SBUX", "BKNG", "GILD", "VRTX", "ISRG", "MDLZ",
    "ADP", "REGN", "ADI", "LRCX", "PANW", "MU", "SNPS", "PDD", "CDNS", "KLAC",
    "MELI", "CHTR", "CSX", "PYPL", "MAR", "ORLY", "MNST", "ASML", "CTAS", "ABNB",
    "LULU", "FTNT", "NXPI", "WDAY", "PCAR", "KDP", "ADSK", "CPRT", "ODFL", "MRVL",
    "PAYX", "CRWD", "SGEN", "ROST", "AEP", "MCHP", "EXC", "KHC", "AZN", "CEG",
    "BKR", "DXCM", "BIIB", "EA", "FAST", "IDXX", "VRSK", "XEL", "CTSH", "TTD",
    "GEHC", "CSGP", "MRNA", "FANG", "TEAM", "GFS", "ON", "DLTR", "DDOG", "WBD",
    "ANSS", "ZS", "EBAY", "WBA", "ILMN", "SIRI", "ZM", "ALGN", "JD", "ENPH", "LCID"
]
    tickers_sp500 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "BRK.B", "TSLA", "UNH", "LLY",
    "XOM", "JPM", "V", "JNJ", "AVGO", "PG", "MA", "HD", "CVX", "MRK", "ADBE", "ABBV", "COST",
    "WMT", "PEP", "KO", "CSCO", "CRM", "MCD", "ACN", "BAC", "LIN", "NFLX", "CMCSA", "AMD",
    "TMO", "PFE", "ORCL", "ABT", "INTC", "DIS", "VZ", "WFC", "COP", "AMGN", "PM", "INTU", "IBM",
    "TXN", "QCOM", "UNP", "NKE", "DHR", "HON", "SPGI", "NOW", "CAT", "NEE", "RTX", "GE", "AMAT",
    "SBUX", "LOW", "T", "BA", "BMY", "ELV", "GS", "TJX", "UPS", "LMT", "DE", "BKNG", "GILD",
    "ISRG", "VRTX", "PLD", "MDT", "MMC", "CI", "SYK", "MS", "PGR", "MDLZ", "BLK", "ADP", "CB",
    "CVS", "AXP", "REGN", "ETN", "AMT", "LRCX", "ADI", "SCHW", "SLB", "C", "MU", "BSX", "PANW",
    "CME", "SO", "TMUS", "EOG", "BDX", "ZTS", "SNPS", "MO", "FI", "EQIX", "BX", "DUK", "NOC",
    "KLAC", "CDNS", "AON", "APD", "ITW", "MPC", "CL", "WM", "CSX", "ICE", "PYPL", "MCK", "HUM",
    "SHW", "PXD", "ORLY", "FDX", "CMG", "GD", "USB", "ANET", "PSX", "ROP", "EMR", "AJG", "PH",
    "MCO", "TGT", "MMM", "FCX", "APH", "ABNB", "TT", "TDG", "PNC", "MSI", "MAR", "LULU", "AZO",
    "HCA", "AIG", "NXPI", "WELL", "VLO", "SRE", "CTAS", "AFL", "PCAR", "NSC", "WMB", "ECL",
    "ADSK", "CCI", "CHTR", "OXY", "CARR", "KMB", "HES", "AEP", "EXC", "ROST", "MCHP", "F",
    "HLT", "COF", "EW", "TFC", "GM", "DLR", "PSA", "CPRT", "MNST", "TEL", "ADM", "OKE", "GIS",
    "TRV", "MSCI", "STZ", "SPG", "MET", "CEG", "NUE", "FTNT", "CNC", "HAL", "DXCM", "PAYX",
    "O", "BKR", "CTVA", "PCG", "BIIB", "IQV", "ODFL", "IDXX", "YUM", "DHI", "LHX", "JCI", "DOW",
    "BK", "ALL", "D", "FAST", "AMP", "GWW", "XEL", "VRSK", "PRU", "KVUE", "SYY", "AME", "KMI",
    "OTIS", "CTSH", "COR", "ACGL", "PEG", "EA", "DD", "ED", "RSG", "KDP", "A", "CMI", "FIS",
    "DVN", "NEM", "CSGP", "ROK", "KR", "URI", "PPG", "LEN", "FANG", "GPN", "ON", "VICI", "HSY",
    "CDW", "VMC", "MLM", "GEHC", "IT", "IR", "KHC", "EL", "WEC", "DG", "PWR", "WTW", "WBD",
    "EIX", "WST", "AWK", "CAH", "DLTR", "ANSS", "MRNA", "SBAC", "AVB", "LYB", "HPQ", "ZBH",
    "FTV", "MPWR", "HIG", "XYL", "CHD", "FICO", "EXR", "CBRE", "WY", "RMD", "KEYS", "APTV",
    "EFX", "TTWO", "MTD", "CTRA", "DFS", "STT", "TSCO", "TROW", "BR", "STE", "ETR", "GLW",
    "RCL", "EBAY", "DAL", "DTE", "AEE", "TRGP", "HPE", "MTB", "WAB", "MOH", "ES", "ULTA", "HWM",
    "FE", "DOV", "RJF", "NVR", "EQR", "PPL", "GPC", "LH", "VRSN", "INVH", "DRI", "IFF", "ILMN",
    "VTR", "GRMN", "PHM", "TDY", "FLT", "PTC", "STLD", "CNP", "BAX", "IRM", "CBOE", "FITB",
    "MRO", "FDS", "TYL", "J", "NDAQ", "BRO", "EXPD", "ATO", "EG", "HOLX", "MKC", "CMS", "COO",
    "EQT", "AKAM", "LVS", "BG", "FSLR", "NTAP", "PFG", "CINF", "CF", "WBA", "VLTO", "TXT",
    "BALL", "ARE", "CLX", "OMC", "HUBB", "AXON", "HBAN", "ALB", "DGX", "IEX", "NTRS", "WAT",
    "RF", "SWKS", "AVY", "JBHT", "SNA", "LDOS", "PKG", "MAA", "STX", "WRB", "LUV", "ALGN", "K",
    "TSN", "ESS", "WDC", "LW", "SWK", "EPAM", "TER", "CAG", "EXPE", "AMCR", "BBY", "POOL", "LNT",
    "APA", "DPZ", "SYF", "MAS", "L", "CCL", "IP", "CFG", "EVRG", "UAL", "NDSN", "HST", "LYV",
    "SJM", "LKQ", "CE", "KIM", "IPG", "MOS", "TAP", "ENPH", "VTRS", "ZBRA", "ROL", "BF.B", "RVTY",
    "NI", "TRMB", "AES", "JKHY", "NRG", "KEY", "CDAY", "REG", "MGM", "GL", "KMX", "INCY", "UDR",
    "PNR", "TFX", "PODD", "GEN", "HRL", "CHRW", "WRK", "CPT", "HII", "PEAK", "FFIV", "EMN", "CRL",
    "AOS", "ALLE", "TECH", "AIZ", "WYNN", "JNPR", "CZR", "PNW", "QRVO", "MKTX", "MTCH", "CPB",
    "RHI", "NWSA", "HSIC", "PAYC", "FOXA", "UHS", "BXP", "BWA", "ETSY", "AAL", "BBWI", "FMC",
    "FRT", "BEN", "TPR", "GNRC", "IVZ", "XRAY", "CTLT", "HAS", "BIO", "WHR", "PARA", "CMA",
    "NCLH", "ZION", "VFC", "SEE", "RL", "MHK", "DVA", "SEDG", "ALK", "FOX", "NWS"
]
    tickers_dow = [
    "MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DIS", "DWDP", 
    "XOM", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "MCD", "MEK", "MSFT", 
    "NKE", "PFE", "PG", "TRV", "UTX", "UNH", "VZ", "V", "WMT", "WBA"
]

    tickers_euro_stoxx = [
    "ADS.DE", "ADYEN.AS", "AIR.PA", "AI.PA", "ALV.DE", "ABI.BR", "ASML.AS",
    "CS.PA", "BBVA.MC", "SAN.MC", "BAS.DE", "BAYN.DE", "BMW.DE", "BNP.PA",
    "CRH", "BN.PA", "DBK.DE", "DB1.DE", "DHL.DE", "DTE.DE", "ENEL.MI", "ENI.MI",
    "EL.PA", "FLTR.IR", "RMS.PA", "IBE.MC", "ITX.MC", "IFX.DE", "INGA.AS",
    "ISP.MI", "KER.PA", "OR.PA", "MC.PA", "MBG.DE", "MUV2.DE", "NOKIA.HE",
    "NDA-FI.HE", "RI.PA", "PRX.AS", "SAF.PA", "SAN.PA", "SAP.DE", "SU.PA",
    "SIE.DE", "GLE.PA", "STLA", "TTE.PA", "UNA.AS", "DG.PA", "VOW.DE", "VNA.DE"
]

    tickers_asx = [
    "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "ANZ.AX", "WBC.AX", "WDS.AX", "WES.AX",
    "MQG.AX", "RIO.AX", "TLS.AX", "WOW.AX", "FMG.AX", "TCL.AX", "GMG.AX", "ALL.AX",
    "STO.AX", "QBE.AX", "NEM.AX", "COL.AX", "BXB.AX", "JHX.AX", "SUN.AX", "COH.AX",
    "XRO.AX", "S32.AX", "ORG.AX", "CPU.AX", "IAG.AX", "SHL.AX", "SCG.AX", "NST.AX",
    "ASX.AX", "WTC.AX", "CAR.AX", "MIN.AX", "APA.AX", "PLS.AX", "TLC.AX", "MPL.AX",
    "RHC.AX", "RMD.AX", "QAN.AX", "SGP.AX", "BSL.AX", "SOL.AX", "AMC.AX", "REA.AX",
    "TWE.AX", "ALD.AX", "SEK.AX", "MGR.AX", "GPT.AX", "DXS.AX", "VCX.AX", "EDV.AX",
    "ORI.AX", "LYC.AX", "IGO.AX", "AZJ.AX", "IEL.AX", "EVN.AX", "NXT.AX", "AGL.AX",
    "ALX.AX", "WOR.AX", "WHC.AX", "SDF.AX", "AKE.AX", "IPL.AX", "CWY.AX", "ALQ.AX",
    "BEN.AX", "QUB.AX", "JBH.AX", "ALU.AX", "CHC.AX", "SVW.AX", "TNE.AX", "LLC.AX",
    "PME.AX", "REH.AX", "MTS.AX", "BOQ.AX", "VUK.AX", "NHF.AX", "LTR.AX",
    "DMP.AX", "FLT.AX", "ORA.AX", "FPH.AX", "ILU.AX", "VEA.AX", "CIA.AX", "PDN.AX",
    "CNU.AX", "RWC.AX", "AMP.AX", "A2M.AX", "SFR.AX", "NHC.AX", "ANN.AX", "CSR.AX",
    "AUB.AX", "NEC.AX", "CGF.AX", "HUB.AX", "NSR.AX", "DOW.AX", "TLX.AX", "BPT.AX",
    "ARB.AX", "WEB.AX", "PMV.AX", "RGN.AX", "HVN.AX", "SQ2.AX", "PRU.AX", "PPT.AX",
    "APE.AX", "TPG.AX", "BRG.AX", "CTD.AX", "BKW.AX", "CLW.AX", "SUL.AX", "VNT.AX",
    "NIC.AX", "DRR.AX", "HDN.AX", "SGM.AX", "GOR.AX", "BAP.AX", "AWC.AX", "TAH.AX",
    "RMS.AX", "DEG.AX", "NUF.AX", "AIA.AX", "BWP.AX", "CMM.AX", "CQR.AX", "LIC.AX",
    "GNC.AX", "MP1.AX", "NEU.AX", "NWL.AX", "IPH.AX", "360.AX", "GUD.AX", "CIP.AX",
    "INA.AX", "BGL.AX", "WPR.AX", "PXA.AX", "CRN.AX", "IFL.AX", "KAR.AX", "JLG.AX",
    "KLS.AX", "BLD.AX", "GMD.AX", "IVC.AX", "ING.AX", "MND.AX", "LNW.AX", "PNI.AX",
    "RRL.AX", "EMR.AX", "CGC.AX", "ARF.AX", "SGR.AX", "LOV.AX", "EVT.AX", "NWH.AX",
    "MFG.AX", "DTL.AX", "HMC.AX", "NAN.AX", "HLS.AX", "AVZ.AX", "CKF.AX", "NWS.AX",
    "SPK.AX", "CNI.AX", "FBU.AX", "SLR.AX", "ELD.AX", "IRE.AX", "DHG.AX", "PNV.AX",
    "CCP.AX", "CQE.AX", "WAF.AX", "SYA.AX", "BGA.AX", "WBT.AX", "CXO.AX", "CHN.AX",
    "LNK.AX", "GOZ.AX", "CMW.AX", "UMG.AX", "NCM.AX", "PBH.AX"
]

    tickers_dax = [
    "SAP.DE", "SIE.DE", "ALV.DE", "AIR.PA", "DTE.DE", "MUV2.DE", "MBG.DE", "BAYN.DE",
    "BAS.DE", "IFX.DE", "DHL.DE", "DB1.DE", "ADS.DE", "BMW.DE", "RWE.DE", "EOAN.DE",
    "VOW3.DE", "DBK.DE", "MRK.DE", "VNA.DE", "DTG.DE", "SHL.DE", "HNR1.DE", "BEI.DE",
    "SY1.DE", "RHM.DE", "HEN3.DE", "CBK.DE", "FRE.DE", "P911.DE", "MTX.DE", "BNR.DE",
    "HEI.DE", "1COV.DE", "QIA.DE", "PAH3.DE", "CON.DE", "SRT3.DE", "ZAL.DE", "ENR.DE"
]



    market = st.selectbox(
        'Choose the market index:',
        (['Nasdaq', 'Dow Jones','S&P 500', 'FTSE', 'Euro Stoxx', 'DAX', 'Ibovespa', 'S&P/BMV IPC', 'S&P/ASX']))

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
        
        
    df_index =  yf.download(ticker, period='4y').ffill(axis=0)
    
    df_index['50-day SMA'] = df_index['Adj Close'].rolling(50, min_periods=1).mean()
    df_index['200-day SMA'] = df_index['Adj Close'].rolling(200, min_periods=1).mean()
    
    df_index['Date'] = df_index.index
    
    hovertext = []
    for i in range(len(df_index)):
        hovertext.append('Date: {}<br>Open: {:.2f}<br>High: {:.2f}<br>Low: {:.2f}<br>Close: {:.2f}'.format(
            df_index['Date'][i].strftime('%Y-%m-%d'), df_index['Open'][i], df_index['High'][i], df_index['Low'][i], df_index['Adj Close'][i]))
    
    fig = go.Figure(data=[go.Ohlc(x=df_index['Date'],
                                 open=df_index['Open'],
                                 high=df_index['High'],
                                 low=df_index['Low'],
                                 close=df_index['Adj Close'],
                                 hovertext=hovertext,
                                 hoverinfo='text',
                                 name=market)])
    
    
    # Adicionar a média móvel de 200 dias como uma linha trace ao gráfico
    fig.add_trace(go.Scatter(x=df_index['Date'], y=df_index['200-day SMA'], mode='lines', name='200-day SMA'))
    
    # Adicionar a média móvel de 50 dias como uma linha trace ao gráfico
    fig.add_trace(go.Scatter(x=df_index['Date'], y=df_index['50-day SMA'], mode='lines', name='50-day SMA'))
    
    fig.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=3, label="3y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )
    
    fig.update_yaxes(tickformat='.2f')
    
    fig.update_layout(title=market)
    
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
    

    
    ################################################################################
    col1, col2 = st.columns(2)

    with col1:

        mkt_50 = pd.DataFrame({'Percentile':above_50,
                              'Date':above_50.index})
        
        fig = px.line(mkt_50, x='Date', y='Percentile', title='Stocks Above 50-Day SMA')
        
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
        fig.add_hline(y=80, line_dash="dash", line_color="gray")
    
        st.plotly_chart(fig)


    with col2:

        mkt_200 = pd.DataFrame({'Percentile':above_200,
                      'Date':above_200.index})

        fig = px.line(mkt_200, x='Date', y='Percentile', title='Stocks Above 200-Day SMA')
        
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
        fig.add_hline(y=-40, line_dash="dash", line_color="gray")
    
        fig.update_layout( width=600,  # Largura do gráfico
                height=500  # Altura do gráfico
            )
        
        st.plotly_chart(fig)

    col5, col6 = st.columns(2)
    
    with col5:
    
        v_r_10 = pd.DataFrame({'Percentile':v_r,
                          'Date':v_r.index})
    
        fig = px.line(v_r_10, x='Date', y='Percentile', title='10-day Supply and Demand Volume')
        
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
        
    
    

