import os

import openai

import base64
import requests
import json

from PIL import Image

import io
#import tensorflow as tf
import numpy as np
from collections import Counter
import webcolors
import math

from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights

import matplotlib.pyplot as plt

import collections

#Getting API Key for OPENAI API authentication
openai.api_key = os.getenv("OPENAI_API_KEY")


def color_detection(image):   

    image = Image.open(image)
    # Resizing image to reduce the number of pixels
    image = image.resize((150, 150))

    # Getting the colors from the image
    pixels = image.load()
    width, height = image.size
    colors = [pixels[x, y] for x in range(width) for y in range(height)]

    # Getting the most common color in the image
    most_common_color = Counter(colors).most_common(1)[0][0]
    # Getting the least common color in the image
    least_common_color = Counter(colors).most_common()[-1][0]

    # Converting the color to a human-readable format
    #Function to find nearest color using 'Euclidean distance' if color not found in the webcolors.CSS3_HEX_TO_NAMES_MAP dictionary
    def closest_color(requested_color):
        min_colors = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_color[0]) ** 2
            gd = (g_c - requested_color[1]) ** 2
            bd = (b_c - requested_color[2]) ** 2
            min_colors[math.sqrt(rd + gd + bd)] = name
        return min_colors[min(min_colors.keys())]      

    try:
        max_color = webcolors.rgb_to_name(most_common_color)
    except ValueError:
        max_color = closest_color(most_common_color)

    try:
        min_color = webcolors.rgb_to_name(least_common_color) 
    except ValueError:
        min_color = closest_color(least_common_color)

    print(f"Max color in the image is: {max_color}")
    print(f"Min color in the image is: {min_color}")
    return max_color, min_color


def object_detection(image): 
    img = read_image(image)
    weights = FCOS_ResNet50_FPN_Weights.DEFAULT
    model = fcos_resnet50_fpn(weights=weights, score_thresh=0.35)
    model.eval()

    #Processing and generating predictions
    preprocess = weights.transforms()
    batch = [preprocess(img)]
    prediction = model(batch)[0]

    #All identified labels
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]

    print(f"All identified objects: \n{labels}\n\n")
    
    #Main obj => High prediction confidence 
    scores = prediction["scores"]
    main_object_index = scores.argmax()
    main_obj = labels[main_object_index]

    print(f"The main object in the image is: {main_obj}")
 
    #Counter to get max repeated and min objects
    counter = collections.Counter(labels)
    most_common_obj = counter.most_common(1)[0][0]
    print(f"The most common object is: {most_common_obj}")
    least_common_obj = counter.most_common()[-1][0]
    print(f"The least common object is: {least_common_obj}")  

    #Boxing with Labels and displaying them
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                          labels=labels,
                          colors="cyan",
                          width=2, 
                          font_size=30,
                          font='ARIALN'
                                       )

    im = to_pil_image(box.detach())

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(im)
    plt.show() 
    
    return main_obj, most_common_obj, least_common_obj 




def generate_prompt(genre, max_color, min_color, main_obj, most_common_obj, least_common_obj):
    if genre == "Horror":
        prompt = f"generate horror story that MUST include the following keywords: {max_color}, {min_color}, {main_obj}, {most_common_obj}, {least_common_obj}"

    elif genre == "Fantasy":
        prompt = f"generate fantasy story that MUST include the following keywords: {max_color}, {min_color}, {main_obj}, {most_common_obj}, {least_common_obj}"

    elif genre == "Fiction":
        prompt = f"generate fiction story that MUST include the following keywords: {max_color}, {min_color}, {main_obj}, {most_common_obj}, {least_common_obj}"

    else:
        prompt = f"generate fun story on novels 'Magic School Bus', 'Noddy Goes to Toyland' and 'Magic Tree House'. Story MUST include the following keywords: {max_color}, {min_color}, {main_obj}, {most_common_obj}, {least_common_obj}"

    return prompt

model = "text-davinci-003"

# Generate the story
#app = Flask(__name__)
#@app.route("/generate_story", methods=("GET", "POST"))
def generate_story():
    #if request.method == "POST":
    #image = request.form["image"]    
    #genre = request.form["genre"]
    image = "./imgs/b6.jpg"
    #read the image
    #im = Image.open("./imgs/b8.jpg")
    #show image
    #im.show()
    genre=input("\n\n\nSelect Genre Horror/Fantasy/Fiction: ")
    print("\n")

    obj=object_detection(image)    
    main_obj, most_common_obj, least_common_obj = obj    

    clr=color_detection(image)
    max_color, min_color = clr
    

    prompt=generate_prompt(genre, max_color, min_color, main_obj, most_common_obj, least_common_obj)


    completions = openai.Completion.create(
        #model="text-davinci-003",
        engine=model,        
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        #stop="The End",
        temperature=1,
        )
    story = completions.choices[0].text
    #return redirect(url_for("index", story=story))
    #return render_template("index.html", story=story)
    print("\n********************     STORY     ********************\n")
    print(story)

#if __name__ == '__main__':
#   app.run()
generate_story()