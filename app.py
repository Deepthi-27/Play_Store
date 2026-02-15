import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import pytz
import os

app = Flask(__name__)

# Global dataframe cache
DATA_CACHE = None
LAST_LOAD_TIME = None
CACHE_TTL = 300  # 5 minutes

def load_data():
    global DATA_CACHE, LAST_LOAD_TIME
    
    # Simple cache mechanism
    now = datetime.now()
    if DATA_CACHE is not None and LAST_LOAD_TIME is not None:
        if (now - LAST_LOAD_TIME).total_seconds() < CACHE_TTL:
            return DATA_CACHE
            
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'apps_data.csv')
        df = pd.read_csv(data_path)
        DATA_CACHE = clean_data(df)
        LAST_LOAD_TIME = now
        return DATA_CACHE
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return empty dataframe structure if load fails to prevent crash
        return pd.DataFrame(columns=['App', 'Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver', 'Android Ver', 'Revenue', 'Sentiment_Subjectivity', 'Country'])

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

    # Synthetic Data Injection (if missing)
    if 'Sentiment_Subjectivity' not in df.columns:
        np.random.seed(42)
        df['Sentiment_Subjectivity'] = np.random.uniform(0, 1, len(df))
        
    if 'Country' not in df.columns:
        countries = ['USA', 'India', 'China', 'Germany', 'France', 'Brazil', 'Japan', 'UK', 'Canada', 'Australia']
        np.random.seed(42)
        df['Country'] = np.random.choice(countries, len(df))

    return df

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

@app.route('/')
def index():
    df = load_data()
    categories = sorted(df['Category'].dropna().unique())
    max_installs = int(df['Installs'].max()) if not df.empty else 1000000000
    return render_template('index.html', categories=categories, max_installs=max_installs)

@app.route('/api/data')
def get_data():
    df = load_data()
    filtered_df = df.copy()
    
    # 1. Categories Filter
    # request.args.getlist('categories[]') for multiple values depending on how JS sends it
    # We'll assume comma separated or multiple keys
    selected_categories = request.args.get('categories')
    if selected_categories:
        cat_list = selected_categories.split(',')
        if cat_list and cat_list[0] != '':
            filtered_df = filtered_df[filtered_df['Category'].isin(cat_list)]

    # 2. Rating Filter
    min_rating = float(request.args.get('min_rating', 0))
    filtered_df = filtered_df[filtered_df['Rating'] >= min_rating]

    # 3. Installs Filter
    min_installs = int(request.args.get('min_installs', 0))
    filtered_df = filtered_df[filtered_df['Installs'] >= min_installs]

    # --- METRICS ---
    metrics = {
        'total_apps': len(filtered_df),
        'total_installs': f"{filtered_df['Installs'].sum():,}",
        'avg_rating': round(filtered_df['Rating'].mean(), 2) if not filtered_df.empty else 0,
        'total_reviews': f"{filtered_df['Reviews'].sum():,}"
    }

    charts = {}

    # --- CHART 1: Category Performance (Horizontal Bar) ---
    try:
        c1_df = filtered_df.copy()
        c1_df = c1_df[
            (c1_df['Rating'] >= 4.0) &
            (c1_df['Size'] >= 10) &
            (c1_df['Last Updated'].dt.month == 1)
        ]
        if not c1_df.empty:
            top_10_cats = c1_df.groupby('Category')['Installs'].sum().nlargest(10).index.tolist()
            c1_df = c1_df[c1_df['Category'].isin(top_10_cats)]
            c1_agg = c1_df.groupby('Category').agg({'Rating': 'mean', 'Reviews': 'sum'}).reset_index().sort_values('Rating')
            
            fig1 = go.Figure()
            # Rating Bar
            fig1.add_trace(go.Bar(
                y=c1_agg['Category'], x=c1_agg['Rating'], 
                name='Avg Rating', orientation='h', marker_color='#636EFA', opacity=0.8,
                hovertemplate='%{y}: %{x:.2f} Stars<extra></extra>'
            ))
            # Reviews Bar (Secondary Axis Logic Simulated via separate trace or normalized?)
            # Dual axis horizontal bar is tricky. Let's stick to Rating and show Reviews in hover or as text.
            fig1.update_traces(text=c1_agg['Reviews'].apply(lambda x: f"{x:,.0f} Reviews"), textposition='auto')
            
            fig1 = update_chart_layout(fig1, "Top 10 Categories: Avg Rating & Reviews")
            fig1.update_layout(
                xaxis=dict(title='Avg Rating', range=[0, 5], showgrid=False),
                yaxis=dict(showgrid=False),
                barmode='group'
            )
            charts['chart1'] = json.loads(json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder))
        else:
            charts['chart1'] = None
    except Exception as e:
        print(f"Error Chart 1: {e}")
        charts['chart1'] = None

    # --- CHART 2: Free vs Paid (Donut Chart) ---
    try:
        c2_df = filtered_df.copy()
        c2_df = c2_df[(c2_df['Installs'] >= 10000) | (c2_df['Revenue'] >= 10000)]
        c2_df = c2_df[c2_df['Android Ver'].astype(str).str.contains(r'^[4-9]', regex=True, na=False)]
        c2_df = c2_df[(c2_df['Size'] > 15) & (c2_df['Content Rating'] == 'Everyone')]
        
        if not c2_df.empty:
            c2_agg = c2_df.groupby('Type').agg({'Installs': 'sum', 'Revenue': 'sum'}).reset_index()
            
            fig2 = go.Figure(data=[go.Pie(
                labels=c2_agg['Type'], 
                values=c2_agg['Installs'], 
                hole=.4,
                marker_colors=['#00CC96', '#EF553B'],
                textinfo='label+percent',
                hoverinfo='label+value+percent'
            )])
            
            fig2 = update_chart_layout(fig2, "Free vs Paid: Installs Distribution")
            fig2.update_layout(showlegend=True)
            charts['chart2'] = json.loads(json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder))
        else:
            charts['chart2'] = None
    except Exception as e:
        print(f"Error Chart 2: {e}")
        charts['chart2'] = None

    # --- CHART 3: Global Installs (Map) ---
    try:
        c3_df = filtered_df.copy()
        c3_df = c3_df[~c3_df['Category'].str.startswith(('A', 'C', 'G', 'S'), na=False)]
        
        if not c3_df.empty:
            top_5_cats = c3_df.groupby('Category')['Installs'].sum().nlargest(5).index.tolist()
            c3_df = c3_df[c3_df['Category'].isin(top_5_cats)]
            c3_agg = c3_df.groupby('Country')['Installs'].sum().reset_index()
            
            fig3 = px.choropleth(
                c3_agg, locations='Country', locationmode='country names',
                color='Installs', color_continuous_scale='Viridis',
                projection='natural earth'
            )
            fig3 = update_chart_layout(fig3, "Global Installs by Region")
            fig3.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)', showframe=False, showcoastlines=True, landcolor='#262730'))
            charts['chart3'] = json.loads(json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder))
        else:
             charts['chart3'] = None
    except Exception as e:
        print(f"Error Chart 3: {e}")
        charts['chart3'] = None

    # --- CHART 4: Cumulative Trend ---
    try:
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
            charts['chart4'] = json.loads(json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder))
        else:
             charts['chart4'] = None
    except Exception as e:
        print(f"Error Chart 4: {e}")
        charts['chart4'] = None

    # --- CHART 5: Bubble Size vs Rating ---
    try:
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
            for trace in fig5.data:
                if trace.name == 'GAME': trace.marker.color = '#FF69B4'
            
            fig5 = update_chart_layout(fig5, "Size vs Rating Impact")
            charts['chart5'] = json.loads(json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder))
        else:
            charts['chart5'] = None
    except Exception as e:
        print(f"Error Chart 5: {e}")
        charts['chart5'] = None

    # --- CHART 6: Growth Trends ---
    try:
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
                line_shape='spline', # Spline for smoother trend
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            high_growth = c6_trend[c6_trend['Growth'] > 0.20]
            for _, row in high_growth.iterrows():
                fig6.add_vrect(
                    x0=row['Month'], x1=row['Month'] + pd.DateOffset(days=30),
                    fillcolor="green", opacity=0.1, layer="below", line_width=0
                )
            
            fig6 = update_chart_layout(fig6, "Growth Trends (>20% MoM Shaded)")
            charts['chart6'] = json.loads(json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder))
        else:
            charts['chart6'] = None
    except Exception as e:
        print(f"Error Chart 6: {e}")
        charts['chart6'] = None

    return jsonify({
        'metrics': metrics,
        'charts': charts,
        'last_updated': datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d %b %Y %I:%M %p IST')
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
