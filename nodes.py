import folder_paths
from rembg import remove,new_session
from PIL import Image
import torch
import numpy as np

models = folder_paths.get_filename_list("u2net")

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ImageRemoveBackgroundRembg:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
    
        return {
            "required": {
                "image": ("IMAGE",),
                "models": (["None"]+models,),
            },
            "optional":{
                "alpha_matting": ("BOOLEAN", {"default": False}),
                "alpha_matting_foreground_threshold": ("INT", {"min":0,"default": 240}),
                "alpha_matting_background_threshold": ("INT", {"min":0,"default": 10}),
                "alpha_matting_erode_size": ("INT", {"min":0,"default": 10}),
                "only_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image, models, only_mask,alpha_matting,alpha_matting_foreground_threshold,alpha_matting_background_threshold,alpha_matting_erode_size):
        model = folder_paths.get_full_path("u2net", models) if models != "None" else None
        image = pil2tensor(
            remove(
                tensor2pil(image),
                 session=new_session(model),
                only_mask=only_mask,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
            ))
        return (image,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Image Remove Background (rembg)": ImageRemoveBackgroundRembg
}
