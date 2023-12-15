from django.urls import path
from . import views

urlpatterns = [
    path('calculate1/', views.hello, name='calculate1'),
    path('calculate2/', views.knn, name='calculate2'),
]



