#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

# This is for accessing the BASE path anywhere(means outside app in any other folder also )
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake.settings')  
django.setup()  

# Add this line

# Import the models here (else it gives error)
from common.deepfake_logic import AudioClassifier,AudioEncoder,Conv1d,ResidualAttentionBlock,LayerNorm,Linear,ModelDimensions

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
