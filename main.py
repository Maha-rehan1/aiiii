
import torch 
from diffusers import StableDiffusionPipeline 
blockedwords = [
    'porn' , 'nude' , 'sexual' , 'rape', 'kill', 'bomb', 'terrorist'
]
model_id = 'runwayml/stable-diffusion-v1-5'
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype = torch.float32
)
pipe = pipe.to('cpu')
print('welcome to AI image generator')
prompt = input('enter your image prompt: ').lower()
if any(word in prompt for word in blockedwords):
    print('unsafe prompt! pls enter a different prompt')
else:
    print('generating image...please wait')
    image = pipe(
        prompt,
        num_interference_steps = 20
    ).images[0]

    image.save('generated_image.png')
    print('image generated successfully!')
    print('saved as generated_image.png')
    image.show()