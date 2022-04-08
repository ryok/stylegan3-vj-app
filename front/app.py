import os
import httpx
import streamlit as st
import pandas as pd
from typing import List
import requests
import json
from pydantic import BaseModel


def format_results(result_files: List[str]) -> pd.DataFrame:
    job_indices, filenames = [], []
    for _, job_id, filename in map(lambda s: s.split('/'), result_files):
        job_indices.append(job_id)
        filenames.append(filename)
    df = pd.DataFrame({'job_id': job_indices, 'filename': filenames})
    return df


BACKEND_PORT = os.environ.get('BACKEND_PORT')

st.title('StyleGAN3-VJ App')
st.text("AI that generate audio reactive movies")

image_files = st.file_uploader('Target audio file',
                               type=['wav', 'mp3'],
                               accept_multiple_files=True)

if len(image_files) > 0 and st.button('Submit'):
    files = [('files', file) for file in image_files]

    r = httpx.post(f'http://backend.docker:{BACKEND_PORT}/predict', files=files)
    st.success(r.json())

if st.button('Refresh'):
    st.success('Refreshed')
    
r = httpx.get(f'http://backend.docker:{BACKEND_PORT}/results')
# df_results = format_results(r.json())
# st.write(df_results)
video_bytes = r.read()
st.video(video_bytes)