from django.urls import path

from . import views
# from home.views import HomeView, PreviewView
from home.views import HomeView, LoginView

urlpatterns = [
    path('', HomeView.as_view(), name="home"),
    path('', LoginView.as_view(), name="login"),
]
