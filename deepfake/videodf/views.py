import os

import moviepy.editor
from moviepy.editor import VideoFileClip

from django.conf import settings
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import re
import urllib.parse
from django.http import JsonResponse

from common.deepfake_logic import load_models,predict

# from common.old_logic_deepfake import load_models,predict

# ---------------------- all paths --------------------------
# VIDEO_UPLOAD_PATH = os.path.join(settings.BASE_DIR, "video_predict/video/")  # x
VIDEO_UPLOAD_PATH = os.path.join(settings.BASE_DIR, "uploaded_files/video_predict/video/") 


VIDEO_GRAPH_LOCATION = os.path.join(settings.BASE_DIR, 'uploaded_files/video_predict/graph/audio_graph.png')

# Define the file paths
output_audio_video_location = os.path.join(settings.BASE_DIR, 'uploaded_files/video_predict/video/output_audio_video.mp4')
output_video_location = os.path.join(settings.BASE_DIR, 'uploaded_files/video_predict/video/output_video.mp4')

# output_grad_video_location = os.path.join(settings.BASE_DIR, 'uploaded_files/video_predict/video/grad_video.mp4')


# Define paths for uploaded and trimmed videos
TRIMMED_VIDEO_PATH = os.path.join(settings.MEDIA_ROOT, "video_predict/video/trimmed/")

# -------------------- Load Models --------------------
mtcnn, model_face, model_audio = load_models()

# -------------------- Functions --------------------

# clear all files in the main directory(uploaded_files) and all the files in the subdirectories
import os

def clear_directory_files(directory_path):
    """
    Clears all files inside the given directory and its subdirectories.
    Subfolders are not deleted, only the files within them are removed.
    """
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                # print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Failed to delete file {file_path}: {e}")

# Video upload view
@csrf_exempt
def video_upload(request):
    if request.method == 'POST' and request.FILES['trim_video_file']:
        
        # Clear all session data at the start
        request.session.flush()

        # Clear all files in the video_predict directory
        video_predict_path = os.path.join(settings.BASE_DIR, "uploaded_files/video_predict/")
        clear_directory_files(video_predict_path)


        # video_file = request.FILES['video_file']
        video_file = request.FILES['trim_video_file']
        sanitized_filename = video_file.name

        # Save the uploaded video
        save_path = os.path.join(VIDEO_UPLOAD_PATH, sanitized_filename)
        os.makedirs(VIDEO_UPLOAD_PATH, exist_ok=True)

        with open(save_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)
        print(f"Video file '{sanitized_filename}' saved successfully.")


         # Check if "Also Predict Audio" checkbox is selected
        predict_audio_flag = 'predict_audio' in request.POST


        # Put inside try block - so if any error occurs then it shows error page
        try:
            # Call the predict function with the appropriate flag
            if predict_audio_flag:
                real_avg_video, fake_avg_video, real_audio_confidence, fake_audio_confidence, graph_generated  = predict(input_path=save_path, mtcnn=mtcnn, model_face=model_face, model_audio=model_audio, predict_audio_flag=True, fake_frames=True, graph_path=VIDEO_GRAPH_LOCATION)
                # User can pass audio_batch_size and video_batch_size as per their system memory
                # real_avg_video, fake_avg_video, real_audio_confidence, fake_audio_confidence = predict(input_path=save_path, mtcnn=mtcnn, model_face=model_face, model_audio=model_audio, predict_audio_flag=True, fake_frames=True, graph_path=VIDEO_GRAPH_LOCATION, audio_batch_size=64, video_batch_size=64)

                if real_audio_confidence == 0.0 and fake_audio_confidence == 0.0:
                    print("Silent audio detected in the uploaded video.")
                    request.session['real_audio_confidence'] = "N/A" 
                    request.session['fake_audio_confidence'] = "N/A"
                else:
                    request.session['real_audio_confidence'] = f"{real_audio_confidence:.2f}%" 
                    request.session['fake_audio_confidence'] = f"{fake_audio_confidence:.2f}%"
            else:
                real_avg_video, fake_avg_video = predict(input_path=save_path, mtcnn=mtcnn, model_face=model_face, model_audio=model_audio, fake_frames=True)
                # optional
                request.session['graph_path'] = None  # No audio graph
            
            output_video_path = None
            if os.path.exists(os.path.join(VIDEO_UPLOAD_PATH, "output_audio_video.mp4")):
                output_video_path = os.path.join(settings.MEDIA_URL, "video_predict/video/output_audio_video.mp4")
            elif os.path.exists(os.path.join(VIDEO_UPLOAD_PATH, "output_video.mp4")):
                output_video_path = os.path.join(settings.MEDIA_URL, "video_predict/video/output_video.mp4")


             # Store the data in session
            request.session['uploaded_video_file'] = output_video_path

            request.session['grad_output_video'] = f"{settings.MEDIA_URL}video_predict/video/grad_video.mp4"

            # request.session['graph_path'] = f"{settings.MEDIA_URL}video_predict/graph/audio_graph.png"
            if graph_generated:
                request.session['graph_path'] = f"{settings.MEDIA_URL}video_predict/graph/audio_graph.png"
            else:
                request.session['graph_path'] = None  # or just skip setting it


            request.session['real_avg_video'] = f"{real_avg_video:.2f}%"
            request.session['fake_avg_video'] = f"{fake_avg_video:.2f}%"

            return redirect('video_display')

        except Exception as e:
            print(f"Prediction Error: {e}")  # Log error for debugging
            return redirect("error_page")  # Redirect if prediction fails

    return render(request, 'videodf/video_upload.html')




# Video display view
def video_display(request):
    # Retrieve all required data from the session
    video_path = request.session.get('uploaded_video_file', None)
    grad_video_path = request.session.get('grad_output_video', None)
    graph_path = request.session.get('graph_path', None)
    real_avg_video = request.session.get('real_avg_video', None)
    fake_avg_video = request.session.get('fake_avg_video', None)
    real_audio_confidence = request.session.get('real_audio_confidence', None)
    fake_audio_confidence = request.session.get('fake_audio_confidence', None)

    # Get real and fake frames
    real_dir = os.path.join(settings.MEDIA_ROOT, 'video_predict', 'Real_frames')
    fake_dir = os.path.join(settings.MEDIA_ROOT, 'video_predict', 'Fake_frames')

    # Get fake grad frames
    grad_fake_dir = os.path.join(settings.MEDIA_ROOT, 'video_predict', 'Grad_Fake_frames')

    real_images = [
        os.path.join(settings.MEDIA_URL, 'video_predict', 'Real_frames', img)
        for img in os.listdir(real_dir) if img.endswith('.jpg')
    ]
    fake_images = [
        os.path.join(settings.MEDIA_URL, 'video_predict', 'Fake_frames', img)
        for img in os.listdir(fake_dir) if img.endswith('.jpg')
    ]

    grad_fake_images = [
        os.path.join(settings.MEDIA_URL, 'video_predict', 'Grad_Fake_frames', img)
        for img in os.listdir(grad_fake_dir) if img.endswith('.jpg')
    ]


    # Pass the data to the template
    context = {
        'video_path': video_path,
        'grad_video_path': grad_video_path,
        'graph_path': graph_path,
        'real_avg_video': real_avg_video,
        'fake_avg_video': fake_avg_video,
        'real_audio_confidence': real_audio_confidence,
        'fake_audio_confidence': fake_audio_confidence,

        'real_images': real_images,
        'fake_images': fake_images,

        'grad_fake_images': grad_fake_images,
    }

    return render(request, 'videodf/video_display.html', context)


@csrf_exempt
def trim_video(request):
    if request.method == 'POST':
        video_file = request.FILES.get('video_file')
        start_time = request.POST.get('start_time', '0')  # Default to 0
        end_time = request.POST.get('end_time')

        if not video_file:
            return JsonResponse({'error': 'No video file provided.'}, status=400)

        # Sanitize the filename
        original_filename = video_file.name
        sanitized_filename = re.sub(r'[^\w\s.-]', '_', original_filename)

        # Save the uploaded video with a temporary name
        temp_video_path = os.path.join(VIDEO_UPLOAD_PATH, f"temp_{sanitized_filename}")
        os.makedirs(VIDEO_UPLOAD_PATH, exist_ok=True)

        with open(temp_video_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)

        try:
            # Load the video file and trim it
            clip = VideoFileClip(temp_video_path)
            start_time = float(start_time)
            end_time = float(end_time) if end_time else clip.duration
            trimmed_clip = clip.subclip(start_time, end_time)

            # Save the trimmed video with the sanitized original name
            os.makedirs(TRIMMED_VIDEO_PATH, exist_ok=True)
            trimmed_video_path = os.path.join(TRIMMED_VIDEO_PATH, sanitized_filename)
            trimmed_clip.write_videofile(trimmed_video_path, codec="libx264", audio_codec="aac")

            # Close video clips
            clip.close()
            trimmed_clip.close()

            # Remove the temporary uploaded video
            os.remove(temp_video_path)

            # Return the trimmed video path relative to MEDIA_URL
            relative_trimmed_path = os.path.relpath(trimmed_video_path, settings.MEDIA_ROOT)
            return JsonResponse({'trimmed_video_path': relative_trimmed_path}, status=200)

        except Exception as e:
            # Remove the temporary file in case of any errors
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)



