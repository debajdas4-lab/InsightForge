import streamlit as st
import pandas as pd
 
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
st.dataframe(df, width="stretch")
st.bar_chart(df)