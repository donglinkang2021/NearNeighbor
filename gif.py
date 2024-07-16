from PIL import Image
import os
def create_gif(image_folder:str, is_delete:bool=True):
    output_gif = f'{image_folder}/output.gif'
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    frames = [Image.open(os.path.join(image_folder, img)) for img in images]
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], loop=0, duration=500)
    print(f'GIF saved as {output_gif}')
    if is_delete:
        for img in images:
            os.remove(os.path.join(image_folder, img))
        print(f'Images in {image_folder} are deleted')
