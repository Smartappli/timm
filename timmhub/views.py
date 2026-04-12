from __future__ import annotations

from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from timmhub.forms import InferenceForm
from timmhub.services.timm_service import timm_service


def index(request: HttpRequest):
    result = None
    error = None

    if request.method == "POST":
        form = InferenceForm(request.POST, request.FILES)
        if form.is_valid():
            model_name = form.cleaned_data["model_name"]
            task_type = form.cleaned_data["task_type"]
            image_bytes = form.cleaned_data["image"].read()
            top_k = form.cleaned_data["top_k"]
            try:
                if task_type == "classification":
                    result = timm_service.classify(model_name, image_bytes, top_k=top_k)
                elif task_type == "embedding":
                    result = timm_service.embedding(model_name, image_bytes)
                else:
                    result = timm_service.feature_maps(model_name, image_bytes)
            except Exception as exc:
                error = str(exc)
    else:
        form = InferenceForm(initial={"model_name": "mobilenetv3_large_100"})

    return render(
        request,
        "timmhub/index.html",
        {
            "form": form,
            "result": result,
            "error": error,
            "suggested_models": timm_service.list_models(limit=24),
        },
    )


@require_GET
def model_list_json(request: HttpRequest) -> JsonResponse:
    return JsonResponse({"models": timm_service.list_models(limit=100)})


@require_GET
def model_meta_json(request: HttpRequest) -> JsonResponse:
    model_name = request.GET.get("model_name", "").strip()
    if not model_name:
        return JsonResponse({"error": "model_name is required"}, status=400)
    try:
        payload = timm_service.get_model_metadata(model_name=model_name)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    return JsonResponse(payload)
