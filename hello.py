from image_transformer import *
from flask import Flask
import base64
from io import BytesIO
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential, load_model
from PIL import Image


app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Hello World</h1>'

# example of dynamic routing
@app.route('/user/<name>')
def user(name):
    return '<h1>Hello, {}!</h1>'.format(name)

# generating matplotlib figures guide
# https://matplotlib.org/devdocs/gallery/user_interfaces/web_application_server_sgskip.html
@app.route('/graph')
def hello():
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.plot([1, 2])
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"

# show the graph of confidence and classification for 90 tests on an image,
# one test for each angle of rotation
@app.route('/test1')
def test_alpha():
    # load model
    model = load_model('./training/TSR_20.h5')
    return find_confidence_limit(model)


def test_on_img(img,model):
    data=[]
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
    X_test=np.array(data)
    predict_x=model.predict(X_test)
    Y_pred=np.argmax(predict_x,axis=1)
    # Y_pred = model.predict_classes(X_test)
    return image,Y_pred,predict_x

"""
Increments the angle of rotation to find where the model
begins to misclassify manipulated images. Looks at the first 3
images/classes from Test.csv
"""
def find_confidence_limit(model,filecsv='Test.csv', mb=1):
    classification_arr = []
    confidence_arr = []
    # Make output dir
    if os.path.isdir('output'):
        shutil.rmtree('output') # remove the directory if it exists
    os.mkdir('output')

    y_test = pd.read_csv(filecsv)
    imgs = y_test["Path"].values
    labels = y_test["ClassId"].values
    # Rotation range
    rot_range = 90
    # Ideal image shape (w, h)
    img_shape = None
    for i in range(1):
        # Input image path
        img_path = imgs[i]
        # Correct class
        true_class = labels[i]
        print("true class",true_class)
        # Instantiate the class
        it = ImageTransformer(img_path, img_shape)
        # Iterate through rotation range
        for ang in range(rot_range):
            # NOTE: Here we can change which angle, axis, shift
            """ Example of rotating an image along x and y axis """
            rotated_img = it.rotate_along_axis(theta = ang)
            save_image('output/{}.jpg'.format(str(ang).zfill(3)), rotated_img)
            plot,prediction,confidence_array = test_on_img(r'./output/{}.jpg'.format(str(ang).zfill(3)),model)
            s = [str(i) for i in prediction]
            predicted_class = int("".join(s))
            confidence = confidence_array[0][np.argmax(confidence_array)]

            classification_arr.append(predicted_class)
            confidence_arr.append(confidence)


        # Generate the figure **without using pyplot**.
        fig = Figure()
        # create subplot axees
        ax1, ax2 = fig.subplots(nrows=2,ncols=1)
        accuracy = classification_arr.count(true_class)/rot_range
        ax1.plot(confidence_arr)
        ax1.set_title('Confidence VS Angle of Rotation', fontsize='20')
        ax1.set_ylabel('Confidence', fontsize='16')
        ax1.grid(True)
        ax2.plot(classification_arr, color='red')
        ax2.set_title('Classification VS Angle of Rotation\nTrue Classification: #{}'.format(true_class), fontsize='20')
        ax2.set_ylabel('Classification #ID', fontsize='16')
        ax2.grid(True)
        fig.supxlabel('Angle of Rotation', fontsize='18')
        fig.suptitle('Comparing Confidence and Predicted Class\nfor a Model with 20 Epochs\nAccuracy {:.2%}'.format(accuracy), fontsize='24')
        fig.tight_layout()
        # fig.bbox_inches()
        # fig.show()

        # Save it to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")

    return f"<img src='data:image/png;base64,{data}'/>"
# need to implement save to file and file recovery for common calculations (which is eventually all of them?)
# implement accuracy graph for tons of images
# backup to github
# deploy with amazon web services, check out heuroku first, how much is that?
