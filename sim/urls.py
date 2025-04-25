"""
URL configuration for sim project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from home import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('realdata/some/', views.some ),
    path('admin/', admin.site.urls),
    path('', views.homepage),
    path('mg/',views.mg),
    path('mm/',views.mm),
    path('gg/',views.gg),
    path('mmd/',views.mmd, name='mmd'),
    path('ggd/',views.ggd, name='ggd'),
    path('mgd/',views.mgd, name='mgd'),
    path('realdata/',views.realdata, name='realdata'),
    path('mmd/mmdfunc/',views.mmdfunc),
    path('mgd/mgdfunc/',views.mgdfunc),
    path('ggd/ggdfunc/',views.ggdfunc),

 
    
]




