from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_video, name='upload_video'),
    path('ask/', views.ask_question, name='ask_question'),
]
