import streamlit as st

import base as bt

def main():
    st.title("Machine Learning Stock Predictor")
    option = st.selectbox("Select an option:", ["MSFT","IBM","SBUX","AAPL","GSPC","Date"])
    if st.button("Generate ML Chart"):
        # Create the chart using the function from base.py
        loss = 1
        while loss > 0.01:
            fig, loss = bt.create_chart(option)
        # Display the chart in the Streamlit app
        st.subheader(loss)
        st.pyplot(fig)

if __name__ == "__main__":
    main()