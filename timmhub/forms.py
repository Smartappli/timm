from django import forms


TASK_CHOICES = [
    ("classification", "Classification"),
    ("embedding", "Embedding"),
    ("feature_maps", "Feature maps"),
]


class InferenceForm(forms.Form):
    model_name = forms.CharField(
        max_length=255,
        initial="mobilenetv3_large_100",
        help_text="Any timm model id, including hf_hub:... models.",
    )
    task_type = forms.ChoiceField(choices=TASK_CHOICES, initial="classification")
    image = forms.ImageField()
    top_k = forms.IntegerField(min_value=1, max_value=20, initial=5)
