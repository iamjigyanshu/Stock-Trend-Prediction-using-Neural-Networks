import streamlit as st
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objs as go




models = ["Model A", "Model B", "Model C"]
accuracy = [0.85, 0.82, 0.78]
data = pd.read_csv('Nasdaq100_interpolate.csv')
data.columns = ['Date', 'Nasdaq100_price', 'crude_price', 'GDP', 'FED rate']
data['SMA_50'] = data['Nasdaq100_price'].rolling(50).mean()
data['SMA_100'] = data['Nasdaq100_price'].rolling(100).mean()
data.dropna(inplace = True)
data.reset_index(drop=True, inplace = True)


def main():

  st.title("Stock Market Trend Prediction Using Neural Networks")
  st.markdown("This app helps you compare different machine learning models.")


  tabs = st.tabs(["Data", "Model", "About"])


  with tabs[0]:

    st.write("""
    This application is designed to provide insights into the performance of various machine learning models. 
    It allows you to explore the data used to train these models, compare their accuracy, and gain a better understanding of their strengths and weaknesses.
    """)


    if data is not None:
        st.subheader("Data Sneak Peek")
        st.write(data.head())  
    else:
        st.warning("Data is not yet loaded. Please check your data import.")



    st.subheader("Data Collection")
    st.write("""
    * **Stock Prices:** Nasdaq 100 index price and crude oil price were obtained via the yfinance API.
    * **Economic Indicators:** US GDP and Fed rate data were retrieved from fred.stlouisfed.org.
    """)


    st.subheader("Data Pre-processing")
    st.write("""
    * **Interpolation:** US GDP and Fed rate data have different update frequencies. To ensure consistent time steps for modeling, an interpolation function was applied to fill in missing values. It's important to note that this assumes a linear relationship between data points, which may not always be perfectly accurate.
    * **Missing Value Removal:** Any remaining missing values in the data were identified and removed. 
    * **Scaling:** The data was scaled to a common range to improve the performance of machine learning models.
    * **Time Series Segmentation:** The data was transformed into a format suitable for LSTM models. This typically involves creating sequences or "windows" of past observations to predict future values.
    """)


    st.subheader("Data Visualizations")


    fig = make_subplots(rows=2, cols=2, subplot_titles=("Nasdaq 100", "Crude Price", "GDP", "Fed Rate"))

    fig.add_trace(go.Scatter(x=data['Date'], y=data['Nasdaq100_price'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], name='SMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_100'], name='SMA 100'), row=1, col=1)

    fig.add_trace(go.Scatter(x=data['Date'], y=data['crude_price'], name='Price'), row=1, col=2)

    fig.add_trace(go.Scatter(x=data['Date'], y=data['GDP'], name='GDP'), row=2, col=1)

    fig.add_trace(go.Scatter(x=data['Date'], y=data['FED rate'], name='Fed Rate'), row=2, col=2)


    fig.update_layout(height=600, width=800, title_text="Time Series Plots")
    fig.update_xaxes(showgrid=False)  
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig)


    st.subheader("Correlation Matrix")


    corr_matrix = data.iloc[:,:5].corr()


    bluegreen_colors = ['#FFDDE3', '#C76D7E', '#9F8082', '#AD9B9A', '#6ACBDE'] 
    fig_cm = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=bluegreen_colors,  
        colorbar_title='Correlation Coefficient'
    ))
    fig_cm.update_layout(
        title='Correlation Matrix',
        xaxis_title='Features',
        yaxis_title='Features'
    )
    st.plotly_chart(fig_cm)

    show_chart = st.button("Explore Nasdaq 100 Chart")

  
    chart_container = st.empty()  
  if show_chart:
        
     
      selected_interval = st.radio("Select Time Interval:", options=["1D", "1W", "1Y", "5Y", "Max"])

      st.session_state.selected_interval = selected_interval

      
      if selected_interval == "1D":
          filtered_data = data[-1:]  
      elif selected_interval == "1W":
          filtered_data = data[-7:]  
      elif selected_interval == "1Y":
          filtered_data = data[-365:]  
      elif selected_interval == "5Y":
          filtered_data = data[-1825:]  
      else:
          filtered_data = data  


      fig = make_subplots(rows=1, cols=1,
                          shared_xaxes=True,
                          subplot_titles=("Nasdaq100 Price and Moving Averages"))


      fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['Nasdaq100_price'], mode='lines', name='Nasdaq100 Price'), row=1, col=1)
      fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['SMA_50'], mode='lines', name='SMA 50'), row=1, col=1)
      fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['SMA_100'], mode='lines', name='SMA 100'), row=1, col=1)


      fig.update_layout(title_text="Nasdaq100 Price and Moving Averages",
                        xaxis_rangeslider_visible=False,
                        xaxis_title="Date",
                        yaxis_title="Price",
                        showlegend=True)


      st.plotly_chart(fig)
  else:
      st.empty()



  with tabs[1]:

    analysis_subtabs = st.tabs(["Model Comparison"])
    with analysis_subtabs[0]:
      selected_models = st.selectbox("Select Models for Comparison", models)

  with tabs[2]:

    st.write("This app is developed by Jigyanshu Singh✌️.")
    st.markdown(" ")

if __name__ == "__main__":
  main()
