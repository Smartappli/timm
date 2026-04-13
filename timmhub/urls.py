from django.urls import path

from timmhub import views

app_name = "timmhub"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/models/", views.model_list_json, name="model_list_json"),
    path("api/model-meta/", views.model_meta_json, name="model_meta_json"),
]
