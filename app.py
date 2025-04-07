from flask import Flask, request, redirect, url_for, render_template
import os
from main import *
import tempfile
import base64

app = Flask(__name__, 
    template_folder='templates',
    static_folder='static')

UPLOAD_FOLDER = 'static/videos/workoutvideos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/workouts')
def workout():
    return render_template('workouts.html')

@app.route('/jumpingjacks', methods=['POST', 'GET'])
def jump():
    return render_template('jumpingjacks.html')

@app.route('/sideplank', methods=['POST', 'GET'])
def sideplank():
    return render_template('sideplank.html')

@app.route('/mountainpose', methods=['POST', 'GET'])
def mountain():
    return render_template('mountainpose.html')

@app.route('/vrikshasana', methods=['POST', 'GET'])
def vrik():
    return render_template('vrikshasana.html')

@app.route('/warrior2', methods=['POST', 'GET'])
def warrior2():
    return render_template('warrior2.html')

@app.route('/jumpingjacksresult', methods=['POST'])
def upload_file1():
    # Check if the post request has the video file
    if 'video' not in request.files:
        return "No video received", 400
    
    file = request.files['video']
    
    # Save the recorded video to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    file.save(temp_file.name)
    temp_file.close()
    
    try:
        # Process the video
        baseline_path = "static/videos/WorkoutVideos/jumpingjacks.mp4"
        rating = similarity(baseline_path, temp_file.name)
        
        return render_template('result.html', result=rating)
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return "Error processing video", 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        # Also clean up converted MP4 if it exists
        mp4_path = temp_file.name.replace('.webm', '.mp4')
        if os.path.exists(mp4_path):
            os.unlink(mp4_path)

@app.route('/sideplankresult', methods=['POST'])
def upload_file2():
    # Check if the post request has the video file
    if 'video' not in request.files:
        return "No video received", 400
    
    file = request.files['video']
    
    # Save the recorded video to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    file.save(temp_file.name)
    temp_file.close()
    
    try:
        # Process the video
        baseline_path = "static/videos/WorkoutVideos/Side Plank.mp4"
        rating = similarity(baseline_path, temp_file.name)
        
        return render_template('result.html', result=rating)
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return "Error processing video", 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        # Also clean up converted MP4 if it exists
        mp4_path = temp_file.name.replace('.webm', '.mp4')
        if os.path.exists(mp4_path):
            os.unlink(mp4_path)

@app.route('/mountainposeresult', methods=['POST'])
def upload_file3():
    # Check if the post request has the video file
    if 'video' not in request.files:
        return "No video received", 400
    
    file = request.files['video']
    
    # Save the recorded video to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    file.save(temp_file.name)
    temp_file.close()
    
    try:
        # Process the video
        baseline_path = "static/videos/WorkoutVideos/Mountain Pose.mp4"
        rating = similarity(baseline_path, temp_file.name)
        
        return render_template('result.html', result=rating)
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return "Error processing video", 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        # Also clean up converted MP4 if it exists
        mp4_path = temp_file.name.replace('.webm', '.mp4')
        if os.path.exists(mp4_path):
            os.unlink(mp4_path)

@app.route('/vrikshasanaresult', methods=['POST'])
def upload_file4():
    # Check if the post request has the video file
    if 'video' not in request.files:
        return "No video received", 400
    
    file = request.files['video']
    
    # Save the recorded video to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    file.save(temp_file.name)
    temp_file.close()
    
    try:
        # Process the video
        baseline_path = "static/videos/WorkoutVideos/vrikshasana.mp4"
        rating = similarity(baseline_path, temp_file.name)
        
        return render_template('result.html', result=rating)
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return "Error processing video", 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        # Also clean up converted MP4 if it exists
        mp4_path = temp_file.name.replace('.webm', '.mp4')
        if os.path.exists(mp4_path):
            os.unlink(mp4_path)

@app.route('/warrior2result', methods=['POST'])
def upload_file5():
    # Check if the post request has the video file
    if 'video' not in request.files:
        return "No video received", 400
    
    file = request.files['video']
    
    # Save the recorded video to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    file.save(temp_file.name)
    temp_file.close()
    
    try:
        # Process the video
        baseline_path = "static/videos/WorkoutVideos/WarriorPose2.mp4"
        rating = similarity(baseline_path, temp_file.name)
        
        return render_template('result.html', result=rating)
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return "Error processing video", 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        # Also clean up converted MP4 if it exists
        mp4_path = temp_file.name.replace('.webm', '.mp4')
        if os.path.exists(mp4_path):
            os.unlink(mp4_path)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5001)