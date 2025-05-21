
import streamlit as st
from sklearn.preprocessing import StandardScaler # z score normalization
from sklearn.decomposition import PCA # dimension reduce 

def app():
    st.title('Preprocessing')
    df = st.session_state.get('df')
    if df is None:
        st.warning('Please upload data first.')
        return

    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        st.error('No numeric columns found. Please upload a dataset with numeric features.')
        return

    cols = st.multiselect('Select numeric features', numeric_cols, default=numeric_cols)
    if not cols:
        st.info('Please select at least one feature.')
        return

    data = df[cols].values

    if st.checkbox('Standardize'):
        data = StandardScaler().fit_transform(data)

    if st.checkbox('Apply PCA for dimensionality reduction'):
        max_comp = min(len(cols), data.shape[1])
        n_components = st.slider(
            'Number of principal components',
            min_value=2,
            max_value=max_comp,
            value=min(2, max_comp)
        )
        data = PCA(n_components=n_components).fit_transform(data)

    st.session_state['data'] = data
    st.success(f'Preprocessing complete. Data matrix is now {data.shape}.')
    st.write(data[:5])
