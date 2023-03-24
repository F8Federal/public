# How to run the script:
# pip install uvicorn requests fastapi tensorflow
# wget https://github.com/VPanjeta/hotdog-or-not-hotdog/raw/master/hot_dog_graph.pb -O model.pb
# uvicorn main:app

# How to test the script:
# curl -v -XPOST -H 'Content-Type: application/json' -d '{"image_url": "https://cf-public-view.s3.amazonaws.com/Product/mlait/menu1.jpg" }' localhost:8000/predict

from fastapi import FastAPI, Request
from uuid import uuid4
import requests
import tensorflow.compat.v1 as tf

app = FastAPI()

# We define a FastAPI endpoint here to handle our prediction requests
@app.post("/predict")
async def predict(request: Request):
    # We first get the request body as JSON
    body = await request.json()
    # We get the image URL and download it
    image_data: bytes = requests.get(body['image_url']).content
    try:
        # Then make the prediction here
        annotation = is_hotdog(image_data)
    except Exception as e:
        print(e)
        return {}, 500
    # Our output is simply a boolean value
    return {"annotation": annotation}

# This function determines if an image is a hotdog or not.
# To do this, it loads a pre-trained model stored in model.pb using Tensorflow.
def is_hotdog(image_data: bytes):
    with tf.gfile.FastGFile("model.pb", 'rb') as inception_graph:
        definition = tf.GraphDef()
        definition.ParseFromString(inception_graph.read())
        tf.import_graph_def(definition, name='')

    with tf.Session() as session:
        tensor = session.graph.get_tensor_by_name('final_result:0')
        result = session.run(tensor, {'DecodeJpeg/contents:0': image_data})
        return bool(result[0][0] > result[0][1])
