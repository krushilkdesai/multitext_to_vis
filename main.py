# Import statements
from googletrans import Translator
from diffusers import StableDiffusionPipeline
import torch

# Configuration class
class CFG:
    device = "cuda"  # or "cpu" if you prefer

    generator = torch.Generator(device=CFG.device).manual_seed(42)

    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (900, 900)
    image_gen_guidance_scale = 9

# Initialize the image generation model
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id,
    torch_dtype=torch.float32,  # Use float32 instead of float16
    use_auth_token='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',
    guidance_scale=CFG.image_gen_guidance_scale
)
image_gen_model = image_gen_model.to(CFG.device)

# Function to get translation
def get_translation(text, dest_lang):
    translator = Translator()
    translated_text = translator.translate(text, dest=dest_lang)
    return translated_text.text

# Get translated text
translation = get_translation("india flag", "en")

# Generate and display image
generated_image = generate_image(translation, image_gen_model)
plt.imshow(generated_image)
plt.axis("off")
plt.show()
