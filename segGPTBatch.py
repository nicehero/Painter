import sys
import io
import requests
import json
import base64
from PIL import Image
import numpy as np
import os

inputPath = "./frames"
maskOutPut = "./masks"
previewOutPut = "./preview"
promptFile = "./prompt.jpg"
promptMaskFile = "./promptMask.jpg"
if not os.path.exists(maskOutPut):
  os.makedirs(maskOutPut)
if not os.path.exists(previewOutPut):
  os.makedirs(previewOutPut)

def resizeImg(img):
    res, hres = 448, 448
    #img = Image.fromarray(img).convert("RGB")
    img = img.resize((res, hres))
    temp = io.BytesIO()
    img.save(temp, format="WEBP")
    return base64.b64encode(temp.getvalue()).decode('ascii')
    
def inference_mask1(prompt,
                mask,
                img,
                img_):
    files = {
        "useSam" : 1,
        "pimage" : resizeImg(prompt),
        "pmask" : resizeImg(mask),
        "img" : resizeImg(img),
        "img_" : resizeImg(img_)
    }
    #r = requests.post("https://flagstudio.baai.ac.cn/painter/run", json = files)
    r = requests.post("http://120.92.79.209/painter/run", json = files)
    a = json.loads(r.text)
    res = []
    for i in range(len(a)):
        #res.append(np.uint8(np.array(Image.open(io.BytesIO(base64.b64decode(a[i]))))))
        res.append(Image.open(io.BytesIO(base64.b64decode(a[i]))))
    #print(len(res))
    return res[1:]

for file_name in os.listdir(inputPath):
    f = os.path.join(inputPath,file_name)
    f2 = os.path.join(maskOutPut,file_name)
    f3 = os.path.join(previewOutPut,file_name)
    print(f2)
    prompt = Image.open(promptFile).convert('RGB')
    promptMask = Image.open(promptMaskFile).convert('RGB')
    img = Image.open(f).convert('RGB')
    res = inference_mask1(prompt,promptMask,img,img)
    res[0].save(f2)
    res[1].save(f3)
