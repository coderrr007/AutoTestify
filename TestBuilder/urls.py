from django.urls import path
from .views import repo_form_view, openai_form_view

urlpatterns = [
    path("", repo_form_view, name="repo_form"),
    path("openai/", openai_form_view, name="openai_form"),
]
