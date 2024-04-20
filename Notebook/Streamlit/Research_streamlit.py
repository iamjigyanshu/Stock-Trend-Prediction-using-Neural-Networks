import streamlit as st
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objs as go



# Sample data (replace with your actual data)
models = ["Model A", "Model B", "Model C"]
accuracy = [0.85, 0.82, 0.78]
data = pd.read_csv('Nasdaq100_interpolate.csv')
data.columns = ['Date', 'Nasdaq100_price', 'crude_price', 'GDP', 'FED rate']
data['SMA_50'] = data['Nasdaq100_price'].rolling(50).mean()
data['SMA_100'] = data['Nasdaq100_price'].rolling(100).mean()
data.dropna(inplace = True)
data.reset_index(drop=True, inplace = True)


def main():
  # Header section
  st.title("Stock Market Trend Prediction Using Neural Networks")
  st.markdown("This app helps you compare different machine learning models.")

  # Navigation tabs
  tabs = st.tabs(["Data", "Model", "About"])

  # Content for each tab
  with tabs[0]:
    # Home tab content
    st.write("""
    This application is designed to provide insights into the performance of various machine learning models. 
    It allows you to explore the data used to train these models, compare their accuracy, and gain a better understanding of their strengths and weaknesses.
    """)

    # Show a glimpse of the data using head()
    if data is not None:
        st.subheader("Data Sneak Peek")
        st.write(data.head())  # Display the first few rows of the data
    else:
        st.warning("Data is not yet loaded. Please check your data import.")

    # Add your content about data collection and pre-processing here  
    # Data Collection section
    st.subheader("Data Collection")
    st.write("""
    * **Stock Prices:** Nasdaq 100 index price and crude oil price were obtained via the yfinance API.
    * **Economic Indicators:** US GDP and Fed rate data were retrieved from fred.stlouisfed.org.
    """)

    # Pre-processing section
    st.subheader("Data Pre-processing")
    st.write("""
    * **Interpolation:** US GDP and Fed rate data have different update frequencies. To ensure consistent time steps for modeling, an interpolation function was applied to fill in missing values. It's important to note that this assumes a linear relationship between data points, which may not always be perfectly accurate.
    * **Missing Value Removal:** Any remaining missing values in the data were identified and removed. 
    * **Scaling:** The data was scaled to a common range to improve the performance of machine learning models.
    * **Time Series Segmentation:** The data was transformed into a format suitable for LSTM models. This typically involves creating sequences or "windows" of past observations to predict future values.
    """)

   # Data plots section
    st.subheader("Data Visualizations")

    # Subplots for all features
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Nasdaq 100", "Crude Price", "GDP", "Fed Rate"))

    fig.add_trace(go.Scatter(x=data['Date'], y=data['Nasdaq100_price'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], name='SMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_100'], name='SMA 100'), row=1, col=1)

    fig.add_trace(go.Scatter(x=data['Date'], y=data['crude_price'], name='Price'), row=1, col=2)

    fig.add_trace(go.Scatter(x=data['Date'], y=data['GDP'], name='GDP'), row=2, col=1)

    fig.add_trace(go.Scatter(x=data['Date'], y=data['FED rate'], name='Fed Rate'), row=2, col=2)

    # Update layout for all subplots
    fig.update_layout(height=600, width=800, title_text="Time Series Plots")
    fig.update_xaxes(showgrid=False)  # Remove gridlines from all x-axes
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig)

    # Correlation Matrix section
    st.subheader("Correlation Matrix")

    # Calculate correlation matrix (assuming data is your DataFrame)
    corr_matrix = data.iloc[:,:5].corr()

    # Create plotly heatmap for correlation matrix (using blue-green colors)
    bluegreen_colors = ['#FFDDE3', '#C76D7E', '#9F8082', '#AD9B9A', '#6ACBDE']  # Sample blue-green palette
    fig_cm = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=bluegreen_colors,  # Use the custom blue-green palette
        colorbar_title='Correlation Coefficient'
    ))
    fig_cm.update_layout(
        title='Correlation Matrix',
        xaxis_title='Features',
        yaxis_title='Features'
    )
    st.plotly_chart(fig_cm)

    show_chart = st.button("Explore Nasdaq 100 Chart")

    # Chart logic (hidden initially)
    chart_container = st.empty()  # Empty container to hold the chart
  if show_chart:
        
      # Buttons for different time intervals
      selected_interval = st.radio("Select Time Interval:", options=["1D", "1W", "1Y", "5Y", "Max"])

      st.session_state.selected_interval = selected_interval

      # Filter data based on selected interval
      if selected_interval == "1D":
          filtered_data = data[-1:]  # Last day
      elif selected_interval == "1W":
          filtered_data = data[-7:]  # Last week
      elif selected_interval == "1Y":
          filtered_data = data[-365:]  # Last year
      elif selected_interval == "5Y":
          filtered_data = data[-1825:]  # Last 5 years
      else:
          filtered_data = data  # All data

      # Create Plotly figure
      fig = make_subplots(rows=1, cols=1,
                          shared_xaxes=True,
                          subplot_titles=("Nasdaq100 Price and Moving Averages"))

      # Add traces for Nasdaq100 price and moving averages
      fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['Nasdaq100_price'], mode='lines', name='Nasdaq100 Price'), row=1, col=1)
      fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['SMA_50'], mode='lines', name='SMA 50'), row=1, col=1)
      fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['SMA_100'], mode='lines', name='SMA 100'), row=1, col=1)

      # Update layout
      fig.update_layout(title_text="Nasdaq100 Price and Moving Averages",
                        xaxis_rangeslider_visible=False,
                        xaxis_title="Date",
                        yaxis_title="Price",
                        showlegend=True)

      # Display the Plotly figure using Streamlit
      st.plotly_chart(fig)
  else:
      st.empty()  # Clear the container if button is not clicked



  with tabs[1]:
    # Analysis tab content
    analysis_subtabs = st.tabs(["Model Comparison"])
    with analysis_subtabs[0]:
      selected_models = st.selectbox("Select Models for Comparison", models)
      # Replace with your chart generation logic using libraries like plotly.express
    #   st.bar_chart(accuracy, labels=selected_models)

  with tabs[2]:
    # About tab content
    st.write("This app is developed by Jigyanshu Singh✌️.")
    st.markdown(" ")

if __name__ == "__main__":
  main()
