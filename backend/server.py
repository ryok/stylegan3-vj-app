import os
import sys
import time
import asyncio
import logging
import numpy as np
import matplotlib.pyplot as plt
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, File, UploadFile
from fastapi.responses import FileResponse
from stylegan3vj.feature_extraction import extract_feature
from stylegan3vj.gen_images import gen_images
from stylegan3vj.create_video import create_video


plt.switch_backend('Agg')

uploadedpath = "/uploads"
app = FastAPI()
logger = logging.getLogger('uvicorn')
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
log_file = os.path.join('/', os.path.basename(__file__) + '.log')
handler = FileHandler(log_file, 'a')
handler.setFormatter(log_fmt)
handler.setLevel(DEBUG)
logger.addHandler(handler)


def save_file(files):
    os.makedirs(uploadedpath, exist_ok=True)
    paths = []
    for file in files:
        path = os.path.join(uploadedpath, file.filename)
        fout = open(path, 'wb')
        while 1:
            chunk = file.read()
            if not chunk: break
            fout.write(chunk)
        fout.close()
        paths.append(path)
    return paths


def delete_files():
    p = Path('results')
    result_files = [str(pp) for pp in p.glob('*/*')]
    for file in result_files:
        if os.path.isfile(file):
            os.remove(file)


def gen_vj(files: List[UploadFile], job_id: str) -> None:
    savedir = Path(f'./results/{job_id}')
    if not savedir.exists():
        savedir.mkdir(parents=True)
    
    for path_to_audio in files:
        logger.debug('extract_feature start..')
        extract_feature(90, path_to_audio)
        logger.debug('gen_images start..')
        gen_images()
        logger.debug('create_video start..')
        create_video(path_to_audio, savedir)
        logger.debug('create_video end..')


@app.post('/predict')
async def predict(files: List[UploadFile] = File(...),
                  background_tasks: BackgroundTasks = None):
    delete_files()
    
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(job_id)

    # paths = save_file(files)
    os.makedirs(uploadedpath, exist_ok=True)
    paths = []
    for file in files:
        path = os.path.join(uploadedpath, file.filename)
        fout = open(path, 'wb')
        while 1:
            chunk = await file.read()
            if not chunk: break
            fout.write(chunk)
        fout.close()
        paths.append(path)

    background_tasks.add_task(gen_vj, files=paths, job_id=job_id)
    return f'{len(files)} files are submitted'


@app.get('/result_files')
async def results():
    p = Path('results')
    # results/yyyymmdd_hhmmss/(wav|mp3)
    result_files = [str(pp) for pp in p.glob('*/*')]
    logger.info(result_files)
    return result_files


@app.get('/results')
async def results():
    p = Path('results')
    # results/yyyymmdd_hhmmss/(wav|mp3)
    result_files = [str(pp) for pp in p.glob('*/*')]
    logger.info(result_files)
    if len(result_files) > 0:
        return FileResponse(path=result_files[0])