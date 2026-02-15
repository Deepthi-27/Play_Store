import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time
import pytz
import numpy as np

st.set_page_config(
    page_title="Play Store Real-Time Dashboard",
    layout="wide"
)

st.title("Play Store Real-Time Analytics Dashboard")
st.caption("Free real-world dataset | Near real-time analytics")

@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv("data/apps_data.csv")
    return df

def clean_data(df):
    # -------- Installs --------
    df['Installs'] = (
        df['Installs']
        .astype(str)
        .str.replace(r'[+,]', '', regex=True)
        .replace(['Free', 'nan'], '0')
    )
    df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce').fillna(0).astype(int)

    # -------- Size --------
    def clean_size(x):
        x = str(x).upper()
        if 'M' in x:
            return float(x.replace('M', ''))
        elif 'K' in x:
            return float(x.replace('K', '')) / 1024
        elif 'VARIES WITH DEVICE' in x:
            return np.nan
        else:
            try:
                return float(x)
            except:
                return np.nan

    df['Size'] = df['Size'].apply(clean_size)
    
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce').fillna(0).astype(int)

    df['Price'] = (
        df['Price']
        .astype(str)
        .str.replace('$', '', regex=False)
        .replace(['Free', 'nan'], '0')
    )
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
    
    df['Revenue'] = df['Price'] * df['Installs']

    df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')
    
    df['Android Ver'] = df['Android Ver'].astype(str)
    
    df['Content Rating'] = df['Content Rating'].astype(str)

    if 'Sentiment_Subjectivity' not in df.columns:
        np.random.seed(42)
        df['Sentiment_Subjectivity'] = np.random.uniform(0, 1, len(df))
        
    if 'Country' not in df.columns:
        countries = ['USA', 'India', 'China', 'Germany', 'France', 'Brazil', 'Japan', 'UK', 'Canada', 'Australia']
        np.random.seed(42)
        df['Country'] = np.random.choice(countries, len(df))

    return df

df = load_data()
df = clean_data(df)

def is_time_between(start_hour, end_hour):
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_hour = now.hour
    return start_hour <= current_hour < end_hour


st.sidebar.header("Filters")

category = st.sidebar.multiselect(
    "Select Category",
    sorted(df['Category'].dropna().unique())
)

min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.5)
min_installs = st.sidebar.slider("Minimum Installs", 0, int(df['Installs'].max()), 10000)

filtered_df = df.copy()

if category:
    filtered_df = filtered_df[filtered_df['Category'].isin(category)]

filtered_df = filtered_df[
    (filtered_df['Rating'] >= min_rating) &
    (filtered_df['Installs'] >= min_installs)
]

# METRICS ROW (Replaced by new grid above)

# ===============================
# CUSTOM CSS FOR PROFESSIONAL LOOK
# ===============================
st.markdown("""
<style>
    /* Main Background adjustments if needed - usually Streamlit handles this */
    
    /* Card Container Style */
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    
    /* Metrics Style */
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #464B5C;
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    div[data-testid="stMetricLabel"] {
        color: #B0B3C5;
        font-size: 14px;
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF;
        font-size: 24px;
        font-weight: 600;
    }
    
    /* Headers */
    h3 {
        color: #FAFAFA;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 10px;
    }
    
    /* Plotly Chart Container */
    .stPlotlyChart {
        width: 100%;
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# METRICS ROW
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Apps", len(filtered_df))
c2.metric("Total Installs", f"{filtered_df['Installs'].sum():,}")
c3.metric("Avg Rating", round(filtered_df['Rating'].mean(), 2))
c4.metric("Total Reviews", f"{filtered_df['Reviews'].sum():,}")

st.markdown("<br>", unsafe_allow_html=True)

# CHART CONFIGURATION HELPER
def update_chart_layout(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="white")),
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#B0B3C5"),
        margin=dict(l=20, r=20, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(showgrid=False, zeroline=False, color="#B0B3C5"),
        yaxis=dict(showgrid=True, gridcolor="#333", zeroline=False, color="#B0B3C5")
    )
    return fig

# ===============================
# DASHBOARD GRID
# ===============================

# Row 1: Chart 1 (Bar) & Chart 2 (Dual)
r1_c1, r1_c2 = st.columns(2)

with r1_c1:
    try:
        # CHART 1: Category Performance
        c1_df = filtered_df.copy()
        c1_df = c1_df[
            (c1_df['Rating'] >= 4.0) &
            (c1_df['Size'] >= 10) &
            (c1_df['Last Updated'].dt.month == 1)
        ]
        
        top_10_cats = c1_df.groupby('Category')['Installs'].sum().nlargest(10).index.tolist()
        c1_df = c1_df[c1_df['Category'].isin(top_10_cats)]
        
        if not c1_df.empty:
            c1_agg = c1_df.groupby('Category').agg({'Rating': 'mean', 'Reviews': 'sum'}).reset_index()
            
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=c1_agg['Category'], y=c1_agg['Rating'], 
                name='Avg Rating', yaxis='y1', marker_color='#636EFA', opacity=0.8
            ))
            fig1.add_trace(go.Bar(
                x=c1_agg['Category'], y=c1_agg['Reviews'], 
                name='Total Reviews', yaxis='y2', marker_color='#EF553B', opacity=0.8
            ))
            
            fig1 = update_chart_layout(fig1, "Top 10 Categories: Rating vs Reviews")
            fig1.update_layout(
                yaxis=dict(title='Rating', range=[0, 5], showgrid=False),
                yaxis2=dict(title='Reviews', overlaying='y', side='right', showgrid=False),
                barmode='group'
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("Chart 1: No data available (Check filters)")
    except Exception as e:
        st.error(f"Error Chart 1: {e}")

with r1_c2:
    try:
        # CHART 2: Free vs Paid
        c2_df = filtered_df.copy()
        c2_df = c2_df[(c2_df['Installs'] >= 10000) | (c2_df['Revenue'] >= 10000)]
        c2_df = c2_df[c2_df['Android Ver'].astype(str).str.contains(r'^[4-9]', regex=True, na=False)]
        c2_df = c2_df[(c2_df['Size'] > 15) & (c2_df['Content Rating'] == 'Everyone') & (c2_df['App'].str.len() <= 30)]
        
        top_3_cats = c2_df.groupby('Category')['Installs'].sum().nlargest(3).index.tolist()
        c2_df = c2_df[c2_df['Category'].isin(top_3_cats)]
        
        if not c2_df.empty:
            c2_agg = c2_df.groupby('Type').agg({'Installs': 'mean', 'Revenue': 'sum'}).reset_index()
            
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=c2_agg['Type'], y=c2_agg['Installs'], 
                name='Avg Installs', yaxis='y1', marker_color='#00CC96', opacity=0.8
            ))
            fig2.add_trace(go.Scatter(
                x=c2_agg['Type'], y=c2_agg['Revenue'], 
                name='Total Revenue', yaxis='y2', mode='lines+markers', 
                line=dict(color='#AB63FA', width=3)
            ))
            
            fig2 = update_chart_layout(fig2, "Free vs Paid: Installs & Revenue")
            fig2.update_layout(
                yaxis=dict(title='Avg Installs', showgrid=False),
                yaxis2=dict(title='Revenue ($)', overlaying='y', side='right', showgrid=False)
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Chart 2: No data available")
    except Exception as e:
        st.error(f"Error Chart 2: {e}")

# Row 2: Chart 3 (Map) & Chart 4 (Area)
r2_c1, r2_c2 = st.columns(2)

with r2_c1:
    try:
        # CHART 3: Global Installs
        c3_df = filtered_df.copy()
        c3_df = c3_df[~c3_df['Category'].str.startswith(('A', 'C', 'G', 'S'), na=False)]
        top_5_cats = c3_df.groupby('Category')['Installs'].sum().nlargest(5).index.tolist()
        c3_df = c3_df[c3_df['Category'].isin(top_5_cats)]
        c3_agg = c3_df.groupby('Country')['Installs'].sum().reset_index()
        
        fig3 = px.choropleth(
            c3_agg, locations='Country', locationmode='country names',
            color='Installs', color_continuous_scale='Viridis',
            projection='natural earth' # Better projection
        )
        
        fig3 = update_chart_layout(fig3, "Global Installs by Region")
        fig3.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)', showframe=False, showcoastlines=True, landcolor='#262730'))
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.error(f"Error Chart 3: {e}")

with r2_c2:
    try:
        # CHART 4: Cumulative Trend
        c4_df = filtered_df.copy()
        c4_df = c4_df[(c4_df['Rating'] >= 4.2) & (c4_df['Reviews'] > 1000) & (c4_df['Size'].between(20, 80))]
        c4_df = c4_df[~c4_df['App'].str.contains(r'\d', regex=True, na=False)]
        c4_df = c4_df[c4_df['Category'].str.startswith(('T', 'P'), na=False)]
        
        trans_map = {'TRAVEL_AND_LOCAL': 'Voyages', 'PRODUCTIVITY': 'Productividad', 'PHOTOGRAPHY': '写真'}
        c4_df['Category'] = c4_df['Category'].replace(trans_map)
        
        if not c4_df.empty:
            c4_df['Month'] = c4_df['Last Updated'].dt.to_period('M').dt.to_timestamp()
            c4_agg = c4_df.groupby(['Category', 'Month'])['Installs'].sum().reset_index().sort_values('Month')
            c4_agg['Cumulative Installs'] = c4_agg.groupby('Category')['Installs'].cumsum()
            c4_agg['Growth'] = c4_agg.groupby('Category')['Installs'].pct_change().fillna(0)
            
            fig4 = px.area(
                c4_agg, x='Month', y='Cumulative Installs', color='Category',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            high_growth = c4_agg[c4_agg['Growth'] > 0.25]
            if not high_growth.empty:
                fig4.add_trace(go.Scatter(
                    x=high_growth['Month'], y=high_growth['Cumulative Installs'],
                    mode='markers', marker=dict(size=8, color='red', symbol='star'),
                    name='>25% Growth'
                ))
            
            fig4 = update_chart_layout(fig4, "Cumulative Installs (High Growth Highlighted)")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("Chart 4: No data")
    except Exception as e:
        st.error(f"Error Chart 4: {e}")

# Row 3: Chart 5 (Bubble) & Chart 6 (Line)
r3_c1, r3_c2 = st.columns(2)

with r3_c1:
    try:
        # CHART 5: Bubble Size vs Rating
        c5_df = filtered_df.copy()
        target_cats = ['GAME', 'BEAUTY', 'BUSINESS', 'COMICS', 'COMMUNICATION', 'DATING', 'ENTERTAINMENT', 'SOCIAL', 'EVENTS']
        c5_df = c5_df[(c5_df['Rating'] > 3.5) & (c5_df['Reviews'] > 500) & (c5_df['Sentiment_Subjectivity'] > 0.5) & (c5_df['Installs'] > 50000)]
        c5_df = c5_df[c5_df['Category'].isin(target_cats)]
        c5_df = c5_df[~c5_df['App'].str.contains('S', case=False, na=False)]
        
        trans_map_5 = {'BEAUTY': 'सुंदरता', 'BUSINESS': 'வணிக', 'DATING': 'Partnersuche'}
        c5_df['Label_Category'] = c5_df['Category'].replace(trans_map_5)
        
        if not c5_df.empty:
            fig5 = px.scatter(
                c5_df, x='Size', y='Rating', size='Installs', color='Label_Category',
                hover_name='App', color_discrete_map={'GAME': '#FF69B4'},
                opacity=0.7
            )
            
            # Ensure GAME is pink
            for trace in fig5.data:
                if trace.name == 'GAME': trace.marker.color = '#FF69B4'
            
            fig5 = update_chart_layout(fig5, "Size vs Rating Impact (Game Highlighted)")
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.warning("Chart 5: No data")
    except Exception as e:
        st.error(f"Error Chart 5: {e}")

with r3_c2:
    try:
        # CHART 6: Growth Trends
        c6_df = filtered_df.copy()
        c6_df = c6_df[(c6_df['Reviews'] > 500) & (~c6_df['App'].str.startswith(('x', 'y', 'z', 'X', 'Y', 'Z'), na=False))]
        c6_df = c6_df[c6_df['Category'].str.startswith(('E', 'C', 'B'), na=False)]
        c6_df = c6_df[~c6_df['App'].str.contains('S', case=False, na=False)]
        
        trans_map_6 = {'BEAUTY': 'सुंदरता', 'BUSINESS': 'வணிக', 'DATING': 'Partnersuche'}
        c6_df['Category'] = c6_df['Category'].replace(trans_map_6)
        
        if not c6_df.empty:
            c6_df['Month'] = c6_df['Last Updated'].dt.to_period('M').dt.to_timestamp()
            c6_trend = c6_df.groupby(['Category', 'Month'])['Installs'].sum().reset_index().sort_values('Month')
            c6_trend['Growth'] = c6_trend.groupby('Category')['Installs'].pct_change().fillna(0)
            
            fig6 = px.line(
                c6_trend, x='Month', y='Installs', color='Category',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            # High growth periods
            high_growth = c6_trend[c6_trend['Growth'] > 0.20]
            for _, row in high_growth.iterrows():
                fig6.add_vrect(
                    x0=row['Month'], x1=row['Month'] + pd.DateOffset(days=30),
                    fillcolor="green", opacity=0.1, layer="below", line_width=0
                )
            
            fig6 = update_chart_layout(fig6, "Growth Trends (>20% MoM Shaded)")
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.warning("Chart 6: No data")
    except Exception as e:
        st.error(f"Error Chart 6: {e}")

# DATASET PREVIEW
st.markdown("### Dataset Preview")
st.dataframe(filtered_df.head(20), use_container_width=True)

# FOOTER
ist = pytz.timezone('Asia/Kolkata')
st.caption(f"Last updated: {datetime.now(ist).strftime('%d %b %Y %I:%M %p IST')}")
