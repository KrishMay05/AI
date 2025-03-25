import streamlit as st
import base as bt  # Import base.py as 'bt'

def main():
    # Option can be a string or any input parameter you want for your chart
    option = "Microsoft Stock Prediction"
    fig = bt.create_chart(option)  # Call the function from base.py
    st.pyplot(fig)  # Display the chart in Streamlit

if __name__ == "__main__":
    main()
