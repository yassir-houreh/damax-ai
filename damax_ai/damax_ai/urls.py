from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from damax_app import views as damax_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('video/', include('damax_app.urls')),
    path('', damax_views.upload_video, name='home'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
