from django.contrib import admin
from django.urls import path
from home import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('', views.homepage),
 
    path('realdata/some/', views.some),
    path('upload/', views.upload),
]+ static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
