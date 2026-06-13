from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('search-engine/', views.search_view, name='search_view'),
    path('analytics/', views.analytics_view, name='analytics_view'),
    path('contact/', views.contact_view, name='contact_view'),
    path('search/', views.search, name='search'),
    path('contact-submit/', views.contact, name='contact'),
]
