import streamlit as st

import pandas as pd
import numpy as np
from numpy.random import default_rng as rng


test_data = pd.DataFrame(
    {
        "col1": list(range(20)) * 3,
        "col2": rng(0).standard_normal(60),
        "col3": ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
    }
)


def dashboard():
    st.text('Data')
    st.data_editor(test_data)
    st.line_chart(test_data, x='col1', y='col2')


def form():
    with st.form('tax-form'):
        st.text('Example Tax Form')
        st.checkbox('Checkbox input', key='checkbox_input')
        st.text_input('Text input', key='text_input')
        st.number_input(
            'Number input',
            placeholder='Enter a number here...',
            key='number_input',
        )
        st.selectbox(
            'Dropdown',
            (1, 2, 3),
            key='dropdown_input',
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.switch_page(dashboard_page)


dashboard_page = st.Page(dashboard, title='Dashboard')
navigation = st.navigation([form, dashboard])
navigation.run()
