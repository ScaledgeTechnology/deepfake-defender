from django.shortcuts import render,redirect
from django.utils.timezone import now
from django.http import JsonResponse
from django.core.cache import cache
import time


def index(request):
    return render(request,'index.html')

def about(request):
    return render(request,'about.html')

def service(request):
    return render(request,'service.html')

def pricing(request):
    return render(request,'pricing.html')

def contact(request):
    return render(request,'contact.html')

def error_page(request):
    return render(request, "error.html")



def send_progress_update(task_type, task_id, progress, message=""):
    """Store progress updates in cache"""
    cache.set(
        f'progress_{task_type}_{task_id}',
        {
            'progress': progress,
            'message': message,
            'timestamp': now().isoformat(),
            'complete': progress >= 100,
            'error': None  # Initialize error as None
        },
        timeout=3600
    )

def send_progress_error(task_type, task_id, error_message):
    """Store error state in cache"""
    cache.set(
        f'progress_{task_type}_{task_id}',
        {
            'error': error_message,
            'timestamp': now().isoformat()
        },
        timeout=3600
    )


def get_progress(request, task_type, task_id):
    """Long polling endpoint for progress updates with error handling"""
    timeout = 30  # seconds
    interval = 0.5
    start_time = time.time()
    
    last_data = None
    while time.time() - start_time < timeout:
        data = cache.get(f'progress_{task_type}_{task_id}')
        
        # Check for error flag
        if data and data.get('error'):
            return JsonResponse({'error': data['error']}, status=500)
            
        if data:
            if not last_data or data['timestamp'] != last_data['timestamp']:
                return JsonResponse(data)
            last_data = data
            
        time.sleep(interval)
    
    return JsonResponse({'error': 'timeout'}, status=408)


