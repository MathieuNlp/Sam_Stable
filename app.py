import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
import cv2

# points color and marker
colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]

device = "cuda"
sam_checkpoint = "./sam_vit_b_01ec64.pth"
model_type = "vit_b"
# SAM model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# SD model
pipe = StableDiffusionInpaintPipeline.from_pretrained(
"runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)

# Acceleration and reduction memory techniques

pipe = pipe.to(device)
#pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing()
#pipe.enable_vae_slicing()
#pipe.enable_vae_tiling()
pipe.enable_xformers_memory_efficient_attention()

with gr.Blocks() as demo:
    
    with gr.Row():
        gr.Markdown(
            '''# Stable SAM : Combinaison of Stable diffusion v1.5 and Segment Anything model for doing inpainting.
            SD model: runwayml stable diffusion v1.5 inpainting
            SAM: vit_b
            '''
        )

    with gr.Row():
        original_img = gr.State(value=None)

        input_img = gr.Image(label="Input Image")
        mask_img = gr.Image(label="Mask")
        output_img = gr.Image(label="Output Image")

    with gr.Row():
        with gr.Tab(label="Segmentation"):
            selected_points = gr.State([])
            masks = gr.State([])

            with gr.Row().style(equal_height=True):
                radio_point = gr.Radio(["Foreground", "Background"], label="Point label", value="Foreground")
                radio_mask = gr.Radio(["Mask_1", "Mask_2", "Mask_3"], label="Mask selection", value="Mask_1")
                undo_points_button = gr.Button("Undo point")
                reset_points_button = gr.Button("Reset points")
                segment_button = gr.Button("Generate mask")

    with gr.Row():
        with gr.Tab(label="Diffusion"):
            with gr.Row().style(equal_height=True):
                prompt_text = gr.Textbox(lines=1, label="Prompt")
                diffusion_button = gr.Button("Run diffusion")
        

    def store_original_image(image):
        return image, [] # reset the selected_points


    def point_selection(image, selected_points, point_type, evt: gr.SelectData):
        # get points
        if point_type == "Foreground":
            selected_points.append((evt.index, 1))

        elif point_type == "Background":
            selected_points.append((evt.index, 0))

        else:
            selected_points.append((evt.index, 1))
        #draw points
        for point, label in selected_points:
            cv2.drawMarker(image, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
        if image[..., 0][0, 0] == image[..., 2][0, 0]:  # BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return Image.fromarray(image) 


    def undo_points(original_image, selected_points):
        temp = original_image.copy()

        if len(selected_points) != 0:
            selected_points.pop()
            for point, label in selected_points:
                cv2.drawMarker(temp, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
                
        if temp[..., 0][0, 0] == temp[..., 2][0, 0]:  # BGR to RGB
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)

        return Image.fromarray(temp) 


    def reset_points(original_image):

        return original_image, []


    def generate_mask(image, selected_points, radio_mask):
        points = []
        labels = []

        for point, label in selected_points:
            points.append(point)
            labels.append(label)

        predictor.set_image(image)
        input_points = np.array(points)
        input_labels = np.array(labels)
        print(input_points)
        print(input_labels)
        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        # (n, sz, sz)
        mask_to_idx = {"Mask_1":0, "Mask_2":1, "Mask_3":2}
        chosen_mask = masks[mask_to_idx[radio_mask]]
        chosen_mask = Image.fromarray(chosen_mask[:, :])

        return chosen_mask, masks

    def plot_new_mask(selected_mask, masks):
        mask_to_idx = {"Mask_1":0, "Mask_2":1, "Mask_3":2}
        chosen_mask = masks[mask_to_idx[selected_mask]]
        chosen_mask = Image.fromarray(chosen_mask[:, :])

        return chosen_mask

    def inpaint(image, mask, prompt):
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        image = image.resize((512, 512))
        mask = mask.resize((512, 512))

        output = pipe(prompt=prompt, 
                      image=image, 
                      mask_image=mask).images[0]

        return output

    input_img.upload(
        store_original_image,
        inputs=[input_img],
        outputs=[original_img, selected_points]
    )
    
    input_img.select(
        point_selection,
        inputs=[input_img, selected_points, radio_point],
        outputs=[input_img]
    )

    undo_points_button.click(
        undo_points,
        inputs=[original_img, selected_points],
        outputs=[input_img]
    )

    reset_points_button.click(
        reset_points,
        inputs=[original_img],
        outputs=[input_img, selected_points]
    )

    segment_button.click(
        generate_mask, 
        inputs=[input_img, selected_points, radio_mask], 
        outputs=[mask_img, masks]
    )
    
    radio_mask.input(
        plot_new_mask,
        inputs=[radio_mask, masks],
        outputs=[mask_img]
    )

    diffusion_button.click(
        inpaint, 
        inputs=[input_img, mask_img, prompt_text], 
        outputs=[output_img]
    )

    if __name__ == "__main__":
        demo.launch(share=True)