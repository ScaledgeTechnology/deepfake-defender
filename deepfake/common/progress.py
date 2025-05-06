from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

def send_progress_update(task_type, task_id, progress, message=""):
    """Send progress updates to both audio and video WebSocket groups"""
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f'{task_type}_progress_{task_id}',
        {
            'type': 'progress_update',
            'progress': progress,
            'message': message,
            'task_type': task_type
        }
    )