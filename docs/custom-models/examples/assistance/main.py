# assistance.py

# How to run the script:
# python version 3.10 (important)
# (Mac only) brew install openblas cmake
# (Mac only) export BLAS=/usr/local/opt/openblas/lib/libblas.dylib
# (Mac only) export LAPACK=/usr/local/opt/openblas/lib/liblapack.dylib
# (Mac only) export PIP_ONLY_BINARY=cmake
# pip install uvicorn pydantic fastapi loguru torch torchvision openmim mmdet mmtrack
# mim install mmcv-full
# wget https://raw.githubusercontent.com/open-mmlab/mmtracking/master/configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py
# wget https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth
# export LOG_LEVEL=DEBUG
# export ML_CUSTOM_ASSISTANCE_TRACKING_CONFIG_PATH=./siamese_rpn_r50_20e_lasot.py
# export ML_CUSTOM_ASSISTANCE_TRACKING_CHECKPOINT_PATH=./siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth
# export ML_CUSTOM_ASSISTANCE_TRACKING_DEVICE=cpu
# uvicorn main:app

# How to test the script:
# wget https://raw.githubusercontent.com/F8Federal/public/main/docs/custom-models/examples/assistance/test.json
# curl -v -XPOST -H 'Content-Type: application/json' -H 'X-API-Key: not_needed_in_this_example' -d @test.json localhost:8000/tracking

from os import environ
import base64
from fastapi import FastAPI
from typing import Dict
from pydantic import BaseModel
import imghdr
from loguru import logger
from mmcv.image import imfrombytes
from mmtrack.apis import inference_sot, init_model

app = FastAPI()

# Define our request schema here
class VideoAnnotationOutput(BaseModel):
    shapes: Dict[str, Dict]
    frames: Dict[str, Dict]

class RequestBody(BaseModel):
    annotation: VideoAnnotationOutput
    update: VideoAnnotationOutput
    model_id: str
    job_id: str
    worker_id: str
    unit_id: str
    video_id: str
    frames: Dict[str, str]

@app.post("/tracking")
def tracking(req: RequestBody):
    # Frames should be same type as mmcv.imread or mmcv.VideoReader
    frames = []
    for frame_number, frame_b64 in req.frames.items():
        # Here we go through each frame and decode it from base64
        frame_bytes = base64.b64decode(frame_b64)
        if imghdr.what(None, h=frame_bytes):
            frame_data = imfrombytes(base64.b64decode(frame_b64))
            frames.append(frame_data)
        else:
            frames.append(None)
            data_slice = str(frame_bytes[:20])
            logger.error(f"Frame data at frame number {frame_number} is not an image. Skipping all predictions for this frame. The frame request to video-annotation service probably failed. First 20 bytes: {data_slice}")

    # Get initial bounding boxes by using the first frame in the update object
    bounding_boxes = {}
    models = {}
    for shape_id, shape in list(req.update.frames.values())[0][
        "shapesInstances"
    ].items():
        # Request JSON boxes are in the format (x, y, width, height)
        # Convert to mmtrack format, which is (x1, y1, x2, y2)
        bounding_boxes[shape_id] = (
            shape["x"],
            shape["y"],
            shape["x"] + shape["width"],
            shape["y"] + shape["height"],
        )
        # Each shape needs its own model object, otherwise the inferences get jumbled
        # Models are built from a config file and existing checkpoint file
        models[shape_id] = init_model(
            environ["ML_CUSTOM_ASSISTANCE_TRACKING_CONFIG_PATH"],
            environ["ML_CUSTOM_ASSISTANCE_TRACKING_CHECKPOINT_PATH"],
            device=environ["ML_CUSTOM_ASSISTANCE_TRACKING_DEVICE"],
        )

    # For each shape on that has a bounding box (e.g. each shape on the first update frame), make predictions
    for i, frame in enumerate(req.update.frames.values()):
        if frames[i] is None:
            continue
        for shape_id, shape in frame["shapesInstances"].items():
            if shape_id in bounding_boxes:
                # This is where the actual prediction is made
                inference = inference_sot(
                    models[shape_id], frames[i], bounding_boxes[shape_id], frame_id=i
                )["track_bboxes"]
                # We store our updates directly in the update object
                shape["x"] = float(inference[0])
                shape["y"] = float(inference[1])
                shape["width"] = float(inference[2] - inference[0])
                shape["height"] = float(inference[3] - inference[1])
                # You can optionally set a confidence score for the prediction, which will be displayed in the UI to annotators
                shape["confidence"] = 1 if inference[4] == -1 else round(float(inference[4]), 4)
                shape["isPrediction"] = True
                logger.debug(
                    f"FRAME INDEX: {i}, BBOX: {bounding_boxes[shape_id]}, INFERENCE: {inference}, CONFIDENCE: {shape['confidence']}"
                )

    # We modified the update object directly, so just return it
    return {"update": req.update}
