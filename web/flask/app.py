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
    f = request.files['inputImage']
    img_path = 'c:/deep/Machine-learning-in-practice/web/flask/uploads/' + secure_filename(f.filename)
    f.save(img_path)
    video_path = 'c:/deep/Machine-learning-in-practice/web/flask/static/out.mp4'
    os.system("python c:\deep\Machine-learning-in-practice\demo.py --input_path {} --out_video_name {}".format(img_path, video_path))
    return render_template('transform.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5001', debug=True)