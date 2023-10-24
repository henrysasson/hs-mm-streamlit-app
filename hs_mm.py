#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from fredapi import Fred
fred = Fred(api_key='0c2c8c3b572a356851d65eaa399a554e')
import plotly.express as px
import bcb
from bcb import Expectativas
from bcb import sgs
import datetime
import warnings
import numpy as np
import nasdaqdatalink
import requests


# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title='HS Market Monitor', layout='wide')

@st.cache_data
def get_data(tickers):
    data = yf.download(tickers, period='3y')['Close']
    return data

# Ações
tickers_acoes = ['^GSPC', '^IXIC', '^RUT', '^N225', '^FTSE', '^STOXX50E', '^GDAXI', '^BVSP', '^AXJO', '^MXX', '000001.SS', '^HSI', '^NSEI']
df_acoes = get_data(tickers_acoes).dropna()
names_acoes = ['SPX', 'Nasdaq', 'Russel 2000', 'Nikkei', 'FTSE', 'Euro Stoxx', 'DAX', 'IBOV', 'S&P ASX', 'BMV', 'Shanghai', 'Hang Seng', 'NSE']
column_mapping = dict(zip(tickers_acoes, names_acoes))
df_acoes.rename(columns=column_mapping, inplace=True)

# Moedas
tickers_moedas = ['EURUSD=X', 'JPY=X', 'GBPUSD=X', 'BRL=X', 'AUDUSD=X', 'MXN=X']
df_moedas = get_data(tickers_moedas).dropna()
names_moedas = ['EURUSD', 'USDJPY', 'GBPUSD', 'USDBRL', 'AUDUSD', 'MXNUSD']
column_mapping = dict(zip(tickers_moedas, names_moedas))
# Renomeie as colunas
df_moedas.rename(columns=column_mapping, inplace=True)

# Commodities
tickers_commodities = ['DBC', 'GSG', 'USO', 'GLD', 'SLV', 'DBA', 'BDRY']
df_commodities = get_data(tickers_commodities).dropna()

# Renda Fixa
tickers_rf = ['BILL', 'SHY', 'IEI', 'IEF', 'TLT', 'TIP', 'STIP', 'LQD', 'HYG', 'EMB', 'BNDX', 'IAGG','HYEM','IRFM11.SA', 'IMAB11.SA']
df_rf = get_data(tickers_rf).dropna()
names_rf = ['BILL', 'SHY', 'IEI', 'IEF', 'TLT', 'TIP', 'STIP', 'LQD', 'HYG', 'EMB', 'BNDX', 'IAGG','HYEM','IRFM', 'IMAB']
column_mapping = dict(zip(tickers_rf, names_rf))
# Renomeie as colunas
df_rf.rename(columns=column_mapping, inplace=True)

# Crypto
tickers_crypto = ['BTC-USD', 'ETH-USD']
df_crypto = get_data(tickers_crypto).dropna()
names_crypto = ['BTCUSD', 'ETHUSD']
column_mapping = dict(zip(tickers_crypto, names_crypto))
# Renomeie as colunas
df_crypto.rename(columns=column_mapping, inplace=True)

# Todos os Ativos
all_assets = pd.concat([df_acoes, df_moedas, df_commodities, df_rf, df_crypto], axis=1).dropna()



options = ['Returns Heatmap', 'Correlation Matrix',  'Market Directionality', 'Macro Indicators', 'Positioning']
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
        matriz = df.ffill().pct_change()[-janela:].corr()
        
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



           

    lookback = st.number_input(label="Choose the lookback period", value=30)

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

    

        if classe == 'Multi Asset':
            
            st.plotly_chart(fig, use_container_width=True)
    
        else:
            fig.update_layout( width=800,  # Largura do gráfico
    height=600  # Altura do gráfico
)

            st.plotly_chart(fig)


    lookback = st.number_input(label="Choose the lookback period", value=30)

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

        col4, col5, col6, col7 = st.columns(4)

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
                        dict(count=15, label="15y", step="year", stepmode="backward"),
                        dict(count=20, label="20y", step="year", stepmode="todate"),            
                        dict(step="all")
                    ])
                )
            )

            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            fig_fci.update_yaxes(tickformat=".2f")
                
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
                        dict(count=15, label="15y", step="year", stepmode="backward"),
                        dict(count=20, label="20y", step="year", stepmode="todate"),
                        dict(step="all")
                    ])
                )
            )


            # Formatar os números do eixo y até a segunda casa decimal e adicionar o símbolo de %
            # Adicionar o símbolo de % ao eixo y
            fig_fci_comp.update_yaxes(tickformat=".2f")
                
            st.plotly_chart(fig_fci_comp)


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
                
            st.plotly_chart(fig_m2_us)

        
        st.subheader('Economic Activity')
        
        col8, col9, col10 = st.columns(3)

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
            
            st.plotly_chart(fig_nfpr)


        with col9:
            
            oh = fred.get_series('AWOTMAN').dropna()

            # Convert to DataFrame for plotting
            oh = pd.DataFrame({'Date': oh.index,
                                'Overtime Hours':oh.values})
            
            
            # Plot both 'Value' and '12M MA' on the same figure
            fig_oh = px.line(oh, x='Date', y='Overtime Hours', title='Average Weekly Overtime Hours of Production and Nonsupervisory Employees, Manufacturing')
            
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
        

            st.plotly_chart(fig_oh)


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
        
        
            st.plotly_chart(fig_gea)
        
        

        st.subheader('Housing')
        
        col11, col12 = st.columns(2)

        with col11:

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

            st.plotly_chart(fig_nhs)

        with col12:

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
        
            st.plotly_chart(fig_hperm)
        


        st.subheader('Sentiment')

        col13, col14 = st.columns(2)
        
        with col13:

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
        
            st.plotly_chart(fig_aiae)

        
        with col14:

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
            
    
            st.plotly_chart(fig_cons)
    

    if economy == "Brazil":
        
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

            fig_scl = px.line(scl, x='Date', y=['Total', 'Households', 'Non-financial corporations'], title='Nonearmarked credit operations outstanding - Total (12-month change)')

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
        height=600  # Altura do gráfico
    )
            fig_scl.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
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

            fig_icc = px.line(icc, x='Date', y=['Total', 'Households', 'Non-financial corporations'], title='ICC Spread')

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
        height=600  # Altura do gráfico
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
        height=600  # Altura do gráfico
    )
            fig_ef.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)


            st.plotly_chart(fig_ef)

        
        with col4:

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
        height=600  # Altura do gráfico
    )
            fig_crf.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
                
            st.plotly_chart(fig_crf)




        col5, col6 = st.columns(2)

        with col5:
            
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
            fig_inf.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
            fig_inf.update_layout( width=600,  # Largura do gráfico
        height=600  # Altura do gráfico
    )

            st.plotly_chart(fig_inf)

        
        with col6:

            ## Brazil M2
            m2_br = sgs.get({'M2':27810}, start='2001-12-01')
            # Convert to DataFrame for plotting
            df_m2_br = pd.DataFrame({
                    'Date': m2_br.index,
                    '12-month change': m2_br['M2'].pct_change(12).values,  
                }).dropna()
            # Plot both 'Value' and '12M MA' on the same figure
            fig_m2_br = px.line(df_m2_br, x='Date', y='12-month change', title='Brazil M2')
                
            fig_m2_br.update_xaxes(
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
            fig_m2_br.update_yaxes(tickformat=".2%")
            fig_m2_br.update_layout( width=600,  # Largura do gráfico
        height=600  # Altura do gráfico
    )
            st.plotly_chart(fig_m2_br)





if selected == 'Positioning':
    st.title('Positioning')
    st.markdown('Commitment of Traders')
    st.markdown('##')

    def fetch_data(contract_code):
        base_url = "https://data.nasdaq.com/api/v3/datasets/CFTC/{}.json"
        params = {"api_key": "xhzW3vmVVALs4xStA47P"}  # Substitua pela sua chave
    
        response = requests.get(base_url.format(contract_code), params=params)
        if response.status_code == 200:
            data_json = response.json()
            df = pd.DataFrame(data_json['dataset']['data'], columns=data_json['dataset']['column_names'])
            return df
        else:
            st.error(f"Erro ao buscar os dados: {response.status_code}")
            return pd.DataFrame()

    category = [
        "CRYPTO CURRENCIES", "CRYPTO CURRENCIES", "CRYPTO CURRENCIES", "CRYPTO CURRENCIES",
        "CURRENCIES", "CURRENCIES", "CURRENCIES", "CURRENCIES", "CURRENCIES", "CURRENCIES",
        "CURRENCIES", "CURRENCIES", "CURRENCIES", "CURRENCIES", "CURRENCIES",
        "ENERGIES", "ENERGIES", "ENERGIES", "ENERGIES", "ENERGIES", "ENERGIES", "ENERGIES",
        "ENERGIES", "ENERGIES",
        "EQUITIES", "EQUITIES", "EQUITIES", "EQUITIES", "EQUITIES",
        "EQUITIES - OTHER", "EQUITIES - OTHER", "EQUITIES - OTHER", "EQUITIES - OTHER",
        "EQUITIES - OTHER", "EQUITIES - OTHER", "EQUITIES - OTHER", "EQUITIES - OTHER",
        "EQUITIES - OTHER", "EQUITIES - OTHER",
        "FIXED INCOME", "FIXED INCOME", "FIXED INCOME",
        "FIXED INCOME - OTHER", "FIXED INCOME - OTHER", "FIXED INCOME - OTHER",
        "FIXED INCOME - OTHER", "FIXED INCOME - OTHER", "FIXED INCOME - OTHER",
        "GRAINS", "GRAINS", "GRAINS", "GRAINS", "GRAINS", "GRAINS", "GRAINS", "GRAINS",
        "METALS", "METALS", "METALS", "METALS", "METALS", "METALS", "METALS",
        "SOFTS", "SOFTS", "SOFTS", "SOFTS", "SOFTS", "SOFTS"
    ]


    market = [
        "BITCOIN", "MICRO BITCOIN", "ETHEREUM", "MICRO ETHER", "EURO FX", "DX USD INDEX",
        "CANADIAN DOLLAR", "BRITISH POUND", "SWISS FRANC", "JAPANESE YEN", "AUSTRALIAN DOLLAR",
        "NEW ZEALAND DOLLAR", "BRAZILIAN REAL", "MEXICAN PESO", "SOUTH AFRICAN RAND",
        "BRENT CRUDE OIL", "CRUDE OIL", "CRUDE OIL", "CRUDE OIL", "CRUDE OIL", "CRUDE OIL",
        "CRUDE OIL", "GASOLINE RBOB", "NATURAL GAS", "S&P 500", "DOW JONES", "NASDAQ",
        "RUSSELL 2000", "NIKKEI", "VIX", "EMINI S&P CONSU STAPLES INDEX", "EMINI S&P ENERGY INDEX",
        "EMINI S&P FINANCIAL INDEX", "EMINI S&P HEALTH CARE INDEX", "EMINI S&P UTILITIES INDEX",
        "E-MINI S&P MATERIALS INDEX", "E-MINI S&P INDUSTRIAL INDEX", "E-MINI S&P TECHNOLOGY INDEX",
        "MSCI EM INDEX", "2-YEAR NOTES", "5-YEAR NOTES", "10-YEAR NOTES", "GE EURODOLLARS",
        "SR1 SECURED OVERNIGHT FINANCING RATE (1-MONTH)", "SR3 SECURED OVERNIGHT FINANCING RATE (3-MONTH)",
        "ZQ FED FUNDS", "TN ULTRA 10-YEAR NOTES", "UB ULTRA 30-YEAR BONDS", "WHEAT", "WHEAT", "WHEAT",
        "OATS", "CORN", "SOYBEAN MEAL", "SOYBEANS", "SOYBEAN OIL", "GOLD", "SILVER", "PLATINUM",
        "PALLADIUM", "COPPER", "ALUMINIUM", "STEEL", "SUGAR", "COCOA", "COFFEE", "ORANGE JUICE",
        "COTTON", "LUMBER"
    ]



    contract_code = [
        "133741_F_ALL", "133742_F_ALL", "146021_F_ALL", "146022_F_ALL", "099741_F_ALL", "098662_F_ALL",
        "090741_F_ALL", "096742_F_ALL", "092741_F_ALL", "097741_F_ALL", "232741_F_ALL", "112741_F_ALL",
        "102741_F_ALL", "095741_F_ALL", "122741_F_ALL", "06765T_F_ALL", "037021_F_ALL", "06739C_F_ALL",
        "067411_F_ALL", "06742G_F_ALL", "06742T_F_ALL", "06765A_F_ALL", "111659_F_ALL", "0233AX_F_ALL",
        "13874A_F_ALL", "124606_F_ALL", "209742_F_ALL", "239742_F_ALL", "240741_F_ALL", "1170E1_F_ALL",
        "138748_F_ALL", "138749_F_ALL", "13874C_F_ALL", "13874E_F_ALL", "13874J_F_ALL", "13874H_F_ALL",
        "13874F_F_ALL", "13874J_F_ALL", "244042_F_ALL", "042601_F_ALL", "044601_F_ALL", "043602_F_ALL",
        "132741_F_ALL", "134742_F_ALL", "134741_F_ALL", "045601_F_ALL", "043607_F_ALL", "020604_F_ALL",
        "001602_F_ALL", "001612_F_ALL", "001626_F_ALL", "004603_F_ALL", "002602_F_ALL", "026603_F_ALL",
        "005602_F_ALL", "007601_F_ALL", "088691_F_ALL", "084691_F_ALL", "076651_F_ALL", "075651_F_ALL",
        "085692_F_ALL", "191693_F_ALL", "192651_F_ALL", "080732_F_ALL", "073732_F_ALL", "083731_F_ALL",
        "040701_F_ALL", "033661_F_ALL", "058644_F_ALL"
    ]




    cot_data = pd.DataFrame({'Category':category, 
                            'Market':market, 
                            'Contract_Code':contract_code})


    
    ativo = st.selectbox(
        'Choose the market:',
        (cot_data['Market'].unique().tolist()))

    # Etapa 1: Filtrar todos os códigos de contrato associados ao mercado selecionado
    contract_codes_for_selected_market = cot_data[cot_data['Market'] == ativo]['Contract_Code'].tolist()

    all_dataframes = []

    # Etapa 2: Para cada código de contrato, faça a chamada à API e obtenha os dados
    for contract_code in contract_codes_for_selected_market:
        temp_df = fetch_data(contract_code)
        all_dataframes.append(temp_df)

    # Asegurando que a coluna 'Dates' é uma data
    for df in all_dataframes:
        df['Date'] = pd.to_datetime(df['Date'])

    # Etapa 3: Concatenar todos esses DataFrames individuais em um único DataFrame
    data = pd.concat(all_dataframes, ignore_index=True).set_index('Date').groupby(level=0).sum()

    dates = data.index

    posicao_ativo = cot_data['Market'].tolist().index(ativo)

    codigo_ativo = str('CFTC/'+str(cot_data['Contract_Code'][posicao_ativo]))

    nome_ativo =  cot_data['Market'][posicao_ativo]

    categoria_ativo = cot_data['Category'][posicao_ativo]


    if categoria_ativo in ['CRYPTO CURRENCIES', 'EQUITIES', 'CURRENCIES', 'EQUITIES - OTHER', 'FIXED INCOME', 'FIXED INCOME - OTHER']:

        # Para Commercials
        commercials_long = data['Dealer Longs']
        commercials_short = data['Dealer Shorts']
        net_commercials = commercials_long - commercials_short

        # Para Large Speculators
        large_specs_long = data['Asset Manager Longs'] + data['Leveraged Funds Longs']
        large_specs_short = data['Asset Manager Shorts'] + data['Leveraged Funds Shorts']
        net_large_specs = large_specs_long - large_specs_short

        # Para Small Speculators
        # Primeiro, deduzimos os totais reportáveis do interesse aberto para obter o valor bruto
        gross_small_specs_long = data['Open Interest'] - data['Total Reportable Longs']
        gross_small_specs_short = data['Open Interest'] - data['Total Reportable Shorts']

        # Em seguida, calculamos a posição líquida dos Small Speculators
        net_small_specs = gross_small_specs_long - gross_small_specs_short
        
    else:
        # Para Commercials
        commercials_long = data['Producer/Merchant/Processor/User Longs']
        commercials_short = data['Producer/Merchant/Processor/User Shorts']
        net_commercials = commercials_long - commercials_short

        # Para Large Speculators
        large_specs_long = data['Money Manager Longs'] + data['Swap Dealer Longs']
        large_specs_short = data['Money Manager Shorts'] + data['Swap Dealer Shorts']
        net_large_specs = large_specs_long - large_specs_short
        
        # Ajustando o sinal de net_large_specs para ser oposto ao de net_commercials
        if net_commercials.sum() > 0:
            net_large_specs = -abs(net_large_specs)
        else:
            net_large_specs = abs(net_large_specs)

        # Para Small Speculators
        gross_small_specs_long = data['Open Interest'] - data['Total Reportable Longs']
        gross_small_specs_short = data['Open Interest'] - data['Total Reportable Shorts']
        net_small_specs = gross_small_specs_long - gross_small_specs_short




    df = pd.DataFrame({
        'Date': dates,
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

    st.plotly_chart(fig_cot, use_container_width=True, height=5000)

    
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




