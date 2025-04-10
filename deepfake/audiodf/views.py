import os

from django.conf import settings
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import re
import urllib.parse

from common.deepfake_logic import load_models,predict
# from common.old_logic_deepfake import load_models,predict


# -------------------- All Paths -------------------- 
AUDIO_UPLOAD_PATH = os.path.join(settings.BASE_DIR, "uploaded_files/audio_predict/audio/")
AUDIO_GRAPH_LOCATION = os.path.join(settings.BASE_DIR, "uploaded_files/audio_predict/graph/audio_graph.png")


# -------------------- Load Models --------------------
mtcnn, model_face, model_audio = load_models()

# ------------------------------------------------
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

# ----------------- For predict audio -----------------

# View for uploading and processing audio
@csrf_exempt
def audio_upload(request):
    # Get audio
    if request.method == "POST" and request.FILES.get("audio_file"):

        # Clear all files in the video_predict directory
        video_predict_path = os.path.join(settings.BASE_DIR, "uploaded_files/audio_predict/")
        clear_directory_files(video_predict_path)

         # Clear all session data at the start
        request.session.flush()

        audio_file = request.FILES["audio_file"]

        # Sanitize the filename
        original_filename = audio_file.name   
        # May be file name contains some special character which may contains conflict during saving(so we need to sanitize it)
        sanitized_filename = re.sub(r'[^\w\s.-]', '_', original_filename)  # Replace unsafe characters with a underscore
        print(f"Original filename: {original_filename}, Sanitized filename: {sanitized_filename}")


        # save audio
        save_path = os.path.join(AUDIO_UPLOAD_PATH, sanitized_filename)
        os.makedirs(AUDIO_UPLOAD_PATH, exist_ok=True)

        with open(save_path, 'wb+') as destination:
        # with open(save_path, 'wb') as destination:   # wb+means both read and write
            for chunk in audio_file.chunks():
                destination.write(chunk)
        print(f"Audio file '{sanitized_filename}' saved successfully.")

        # Put inside try block - so if any error occurs then it shows error page
        try:
            real_confidence, fake_confidence = predict(save_path, mtcnn, model_face, model_audio, predict_audio_flag=True, graph_path=AUDIO_GRAPH_LOCATION )

            request.session['graph_path'] = f"{settings.MEDIA_URL}audio_predict/graph/audio_graph.png"
            request.session['uploaded_audio_file'] = f"{settings.MEDIA_URL}audio_predict/audio/{urllib.parse.quote(sanitized_filename)}"
            request.session['real_confidence'] = f"{real_confidence:.2f}"
            request.session['fake_confidence'] = f"{fake_confidence:.2f}"

            # # Redirect to `audio_display/`
            return redirect("audio_display")   # This is the name of the URL where we want to redirect

        except Exception as e:
            print(f"Prediction Error: {e}")  # Log error for debugging
            return redirect("error_page")  # Redirect if prediction fails

    return render(request, "audiodf/audio_upload.html")


def audio_display(request):
    #Retrieve session data
    audio_file = request.session.get('uploaded_audio_file', None)
    real_confidence = request.session.get('real_confidence', None)
    fake_confidence = request.session.get('fake_confidence', None)
    graph_path = request.session.get('graph_path', None)

    if not (audio_file and real_confidence and fake_confidence and graph_path):
        return redirect("audio_upload")

    return render(request, "audiodf/audio_display.html", {
        "audio_file": audio_file,
        "graph_path": graph_path,
        "real_confidence": real_confidence,
        "fake_confidence": fake_confidence,
    })
