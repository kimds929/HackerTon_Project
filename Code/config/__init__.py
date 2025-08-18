# __init__.py (루트 디렉토리)
from __future__ import absolute_import, unicode_literals

# Celery 앱을 가져오기 위한 설정
from .celery import app as celery_app

__all__ = ('celery_app',)