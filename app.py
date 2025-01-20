from flask import Flask, request, redirect, url_for, render_template
import os
from main import *

app = Flask(__name__, 
    template_folder='templates',
    static_folder='static/css')

UPLOAD_FOLDER = 'static/videos/workoutvideos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/jumpingjacks', methods=['POST', 'GET'])
def jump():
    return render_template('jumpingjacks.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return "No file part"
    
    file = request.files['video']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
    
    baseline_path = "/Users/rohansatram/code/WorkoutApp/static/videos/WorkoutVideos/jumpingjacks.mp4"
    rating = similarity(baseline_path, file_path)
    return render_template('result.html', result = rating)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5001)