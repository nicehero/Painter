# -*- coding: utf-8 -*-

import sys
import io
import requests
import json
import base64
from PIL import Image
import numpy as np
import gradio as gr

def inference_mask1(prompt,
              img,
              img_):
    files = {
        "useSam" : 1,
        "pimage" : resizeImg(prompt["image"]),
        "pmask" : resizeImg(prompt["mask"]),
        "img" : resizeImg(img),
        "img_" : resizeImg(img_)
    }
    #r = requests.post("https://flagstudio.baai.ac.cn/painter/run", json = files)
    r = requests.post("http://120.92.79.209/painter/run", json = files)
    a = json.loads(r.text)
    res = []
    for i in range(len(a)):
        #out = Image.open(io.BytesIO(base64.b64decode(a[i])))
        #out = out.resize((224, 224))
        #res.append(np.uint8(np.array(out)))
        res.append(np.uint8(np.array(Image.open(io.BytesIO(base64.b64decode(a[i]))))))
    print(len(res))
    return res[1:]

def resizeImg(img):
    res, hres = 448, 448
    img = Image.fromarray(img).convert("RGB")
    img = img.resize((res, hres))
    temp = io.BytesIO()
    img.save(temp, format="WEBP")
    return base64.b64encode(temp.getvalue()).decode('ascii')

def inference_mask_cat(
              prompt,
              img,
              img_,
              ):
    output_list = [img, img_]
    return output_list


# define app features and run

demo_mask = gr.Interface(fn=inference_mask1, 
                   inputs=[gr.ImageMask(brush_radius=8, label="prompt (提示图)").style(height=448, width=448), gr.Image(label="img1 (测试图1)"), gr.Image(label="img2 (测试图2)")], 
                    #outputs=[gr.Image(shape=(448, 448), label="output1 (输出图1)"), gr.Image(shape=(448, 448), label="output2 (输出图2)")],
                    outputs=[gr.Image(label="output mask").style(height=384, width=384), gr.Image(label="output1 (输出图1)").style(height=384, width=384), gr.Image(label="output2 (输出图2)").style(height=384, width=384)],
                    #outputs=gr.Gallery(label="outputs (输出图)"),
                    #title="SegGPT for Any Segmentation<br>(Painter Inside)",
                    description="",
                   allow_flagging="never",
                   css='.fixed-height.svelte-rlgzoo {height: 100%;}'
                   )


title = "SegGPT: Segmenting Everything In Context"

#demo = gr.TabbedInterface([demo_mask, ], ['General 1-shot', ], title=title)

#demo.launch(share=True, auth=("baai", "vision"))
demo_mask.launch(enable_queue=False, server_port=34311)
#demo.launch(server_name="0.0.0.0", server_port=34311)
# -
