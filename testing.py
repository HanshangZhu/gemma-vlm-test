######## script if your model is not loaded properly ########



from transformers import PaliGemmaForConditionalGeneration

model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-224")
print(model)
