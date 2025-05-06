# from django.http import HttpResponse
import uuid
from django.shortcuts import render

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
IMAGE_UPLOAD_PATH = os.path.join(settings.BASE_DIR, "uploaded_files/image_predict/image/")



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

# ----------------- For predict image -----------------


# View for uploading and processing audio
@csrf_exempt
def image_upload(request):
    # Get audio
    if request.method == "POST" and request.FILES.get("image_file"):

        # Clear all files in the video_predict directory
        video_predict_path = os.path.join(settings.BASE_DIR, "uploaded_files/image_predict/")
        clear_directory_files(video_predict_path)

         # Clear all session data at the start
        request.session.flush()

        image_file = request.FILES["image_file"]

        # Sanitize the filename
        original_filename = image_file.name   
        # May be file name contains some special character which may contains conflict during saving(so we need to sanitize it)
        sanitized_filename = re.sub(r'[^\w\s.-]', '_', original_filename)  # Replace unsafe characters with a underscore
        print(f"Original filename: {original_filename}")
        print(f"Sanitized filename: {sanitized_filename}")


        # save image
        os.makedirs(IMAGE_UPLOAD_PATH, exist_ok=True)
        save_path = os.path.join(IMAGE_UPLOAD_PATH, sanitized_filename)
        

        with open(save_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        print(f"Image file '{sanitized_filename}' saved successfully.")


        # ✅ **TRY calling the predict function and handle errors** - If face not found
        try:
            task_id = request.POST.get('task_id', str(uuid.uuid4()))

            # all_confidence_list = predict(save_path, mtcnn, model_face, model_audio)
            all_confidence_list = predict(input_path = save_path, mtcnn = mtcnn, model_face=model_face, model_audio =model_audio, fake_frames=True, task_id=task_id)
            if all_confidence_list is None:
                raise ValueError("Predict function returned None")
            
            request.session['annotated_img_path'] = f"{settings.MEDIA_URL}image_predict/image/annotated_image.jpg"
            request.session['all_confidence_list'] = all_confidence_list

            # ✅ Only add masked image path if any fake confidence exists
            fake_exists = False
            for confidence in all_confidence_list:
                match = re.findall(r"(\d+\.\d+)%", confidence)
                if len(match) == 2:
                    real_conf = float(match[0])
                    fake_conf = float(match[1])
                    if fake_conf > real_conf:
                        fake_exists = True
                        break

            if fake_exists:
                request.session['annotated_masked_image'] = f"{settings.MEDIA_URL}image_predict/image/annotated_masked_image.jpg"


            return redirect("image_display")
        
        except Exception as e:
            print(f"Prediction Error: {e}")  # Log error for debugging
            return redirect("error_page")  # Redirect if prediction fails

    return render(request, "imagedf/image_upload.html")


def image_display(request):
    image_file = request.session.get('annotated_img_path', None)
    masked_image_file = request.session.get('annotated_masked_image', None)
    all_confidence_list = request.session.get('all_confidence_list', None)

    processed_confidence = []
    show_message_only = False
    real_frame_message = "As this image frame is predicted to be real, Grad-CAM fake frame visualization is not necessary."

    if all_confidence_list:
        for confidence in all_confidence_list:
            match = re.findall(r"(\d+\.\d+)%", confidence)
            if len(match) == 2:
                real_conf = float(match[0])
                fake_conf = float(match[1])
                confidence_image = "images/thumpup.png" if real_conf > fake_conf else "images/thumpdown.png"
                processed_confidence.append({
                    "text": confidence,
                    "image": confidence_image
                })

    # If masked image isn't available, we only show main image + message
    if not masked_image_file:
        show_message_only = True

    return render(request, "imagedf/image_display.html", {
        "image_file": image_file,
        "masked_image_file": masked_image_file,
        "processed_confidence": processed_confidence,
        "show_message_only": show_message_only,
        "real_frame_message": real_frame_message,
    })



