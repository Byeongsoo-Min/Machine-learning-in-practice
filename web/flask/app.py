from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

count = 0

app = Flask (__name__, static_folder='static')

"""
@app.route('/')
def home():
    global count
    count += 1
    return jsonify(
        text='Hello, world',
        count=count
    )
"""

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/imageUpload', methods = ['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    input_image = "c:/deep/Machine-learning-in-practice/web/flask/uploads/input.jpg"
    if os.path.exists(input_image):
      os.system("del c:\\deep\\Machine-learning-in-practice\\web\\flask\\uploads\\input.jpg")
    f = request.files['inputImage']
    img_path = "c:\\deep\\Machine-learning-in-practice\\web\\flask\\uploads\\" + secure_filename(f.filename)
    f.save(img_path)
    os.system("ren {} {}".format(img_path, 'input.jpg')) 
    video_path = 'c:/deep/Machine-learning-in-practice/web/flask/static/out.mp4'
    os.system("python c:\deep\Machine-learning-in-practice\demo.py --input_path {} --out_video_name {}".format(input_image, video_path))
    cyroot = "C:\\deep\\pytorch-CycleGAN-and-pix2pix"
    os.system("python {}\\test.py --dataroot C:\\deep\\Machine-learning-in-practice\\web\\flask\\uploads --name {}\\checkpoints\\civ2_cyclegan --model test --no_dropout".format(cyroot, cyroot))
    os.system("move C:\\deep\\pytorch-CycleGAN-and-pix2pix\\checkpoints\\civ2_cyclegan\\test_latest\\images\\input_fake.png C:\\deep\\Machine-learning-in-practice\\web\\flask\\static\\out.png")
    return render_template('transform.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5001', debug=True)