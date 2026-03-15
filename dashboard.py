import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Market Sentiment x Trader Analysis", layout="wide", page_icon="chart_with_upwards_trend")

st.title("Market Sentiment x Trader Performance Dashboard")

@st.cache_data
def load_data():
    daily_seg = pd.read_csv('daily_segmented.csv', parse_dates=['date'])
    daily_agg = pd.read_csv('daily_aggregate_metrics.csv', parse_dates=['date'])
    trader_stats = pd.read_csv('trader_segments.csv')
    merged = pd.read_csv('merged_data.csv', parse_dates=['date'])
    return daily_seg, daily_agg, trader_stats, merged

daily_seg, daily_agg, trader_stats, merged = load_data()

smap = {'Extreme Fear': 'Fear', 'Fear': 'Fear', 'Neutral': 'Neutral', 'Greed': 'Greed', 'Extreme Greed': 'Greed'}
daily_seg['sentiment_binary'] = daily_seg['sentiment'].map(smap)
daily_agg['sentiment_binary'] = daily_agg['sentiment'].map(smap)

st.sidebar.header("Filters")
sentiment_filter = st.sidebar.multiselect('Sentiment', daily_seg['sentiment'].unique(), default=list(daily_seg['sentiment'].unique()))
date_range = st.sidebar.date_input('Date Range', [daily_seg['date'].min(), daily_seg['date'].max()])

filtered = daily_seg[(daily_seg['sentiment'].isin(sentiment_filter)) & (daily_seg['date'] >= pd.Timestamp(date_range[0])) & (daily_seg['date'] <= pd.Timestamp(date_range[1]))]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trades", f"{filtered['total_trades'].sum():,}")
col2.metric("Avg Daily PnL", f"${filtered['net_pnl'].mean():,.2f}")
col3.metric("Avg Win Rate", f"{filtered['win_rate'].mean():.1%}")
col4.metric("Active Traders", f"{filtered['Account'].nunique()}")

st.divider()
tab1, tab2, tab3 = st.tabs(["Sentiment & Performance", "Trader Segments", "Insights"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(filtered, x='sentiment', y='net_pnl', color='sentiment', title='PnL Distribution by Sentiment', color_discrete_sequence=px.colors.qualitative.Set2, category_orders={'sentiment': ['Extreme Fear','Fear','Neutral','Greed','Extreme Greed']})
        fig.update_layout(template='plotly_dark', height=450)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        wr_data = filtered.groupby('sentiment')['win_rate'].mean().reindex(['Extreme Fear','Fear','Neutral','Greed','Extreme Greed'])
        fig2 = px.bar(x=wr_data.index, y=wr_data.values, title='Avg Win Rate by Sentiment', color=wr_data.values, color_continuous_scale='RdYlGn')
        fig2.update_layout(template='plotly_dark', height=450, xaxis_title='', yaxis_title='Win Rate')
        st.plotly_chart(fig2, use_container_width=True)

    ts_data = filtered.groupby('date').agg(pnl=('net_pnl','sum'), fgi=('fgi_value','first')).reset_index()
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Scatter(x=ts_data['date'], y=ts_data['pnl'].rolling(7).mean(), name='PnL (7d MA)', line=dict(color='#4ECDC4', width=2)), secondary_y=False)
    fig3.add_trace(go.Scatter(x=ts_data['date'], y=ts_data['fgi'], name='FGI', line=dict(color='#FF6B35', width=1.5), opacity=0.7), secondary_y=True)
    fig3.update_layout(template='plotly_dark', title='Daily PnL vs Fear & Greed Index', height=400)
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    seg_choice = st.selectbox('Segment By', ['leverage_segment', 'frequency_segment', 'performance_segment'])
    c1, c2 = st.columns(2)
    with c1:
        fig4 = px.box(filtered, x=seg_choice, y='net_pnl', color='sentiment_binary', title='PnL by Segment x Sentiment', color_discrete_sequence=['#FF6B6B','#7f7f7f','#4ECDC4'])
        fig4.update_layout(template='plotly_dark', height=450)
        st.plotly_chart(fig4, use_container_width=True)
    with c2:
        fig5 = px.scatter(trader_stats, x='avg_win_rate', y='total_pnl', color='performance_segment', size='total_volume', hover_data=['Account'], title='Trader Map: Win Rate vs PnL', color_discrete_sequence=['#2ca02c','#ff7f0e','#d62728'])
        fig5.update_layout(template='plotly_dark', height=450)
        st.plotly_chart(fig5, use_container_width=True)

with tab3:
    st.subheader("Key Findings & Strategy Rules")
    st.markdown("**Strategy 1:** During Fear days, reduce position sizes by ~30percent for high-leverage traders. During Greed days, tighten stop-losses.")
    st.markdown("**Strategy 2:** Infrequent traders should increase activity during Extreme Fear. Frequent traders should reduce activity during Extreme Greed.")
    st.dataframe(filtered.groupby(['sentiment', seg_choice])['net_pnl'].agg(['mean','median','std','count']).round(2), use_container_width=True)