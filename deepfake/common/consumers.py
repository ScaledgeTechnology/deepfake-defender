import json
from channels.generic.websocket import AsyncWebsocketConsumer

class AnalysisProgressConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.task_type = self.scope['url_route']['kwargs']['task_type']
        self.task_id = self.scope['url_route']['kwargs']['task_id']
        self.group_name = f'{self.task_type}_progress_{self.task_id}'
        
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    async def progress_update(self, event):
        await self.send(text_data=json.dumps({
            'progress': event['progress'],
            'message': event['message'],
            'task_type': event['task_type']
        }))