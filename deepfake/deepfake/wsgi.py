"""
WSGI config for deepfake project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/wsgi/
"""

import os
from common.deepfake_logic import AudioClassifier, AudioEncoder, Conv1d, ResidualAttentionBlock, LayerNorm, Linear, ModelDimensions
import sys

# Patch ALL required classes into __main__
sys.modules['__main__'].AudioClassifier = AudioClassifier
sys.modules['__main__'].AudioEncoder = AudioEncoder
sys.modules['__main__'].Conv1d = Conv1d                  # Add this
sys.modules['__main__'].ResidualAttentionBlock = ResidualAttentionBlock
sys.modules['__main__'].LayerNorm = LayerNorm
sys.modules['__main__'].Linear = Linear
sys.modules['__main__'].ModelDimensions = ModelDimensions


from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake.settings')

application = get_wsgi_application()
