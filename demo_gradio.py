import numpy as np

try:
    from CropFormer.demo_mask2former.demo import get_entityseg
    from detectron2.data.detection_utils import read_image
except:
    print("EntitySeg is not installed")
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except:
    print("SAM2 is not installed")

np.random.seed(0)
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from corrclip_segmentor import CorrCLIPSegmentation
from scipy.ndimage import label
import gradio as gr
from matplotlib.colors import hsv_to_rgb


def generate_distinct_colors(n):
    # 生成 n 个均匀分布的 HSV 颜色
    hues = np.linspace(0, 1, n, endpoint=False)
    sats = np.ones(n) * 0.8  # 饱和度设为 0.8
    vals = np.ones(n) * 0.9  # 明度设为 0.9
    hsv_colors = np.stack((hues, sats, vals), axis=-1)

    # 转换为 RGB 颜色
    rgb_colors = (hsv_to_rgb(hsv_colors) * 255).astype(np.uint8)
    return rgb_colors


# 将 set_sam_mask_generator 函数定义在全局作用域
def set_sam_mask_generator(model, sam_model, points_per_side=8, pred_iou_thresh=0.4, stability_score_thresh=0.4, multimask_output=False):
    """
    配置并设置模型的掩码生成器。
    """
    model.mask_generator_type = 'sam2'
    model.mask_generator = SAM2AutomaticMaskGenerator(
        model=sam_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        multimask_output=multimask_output,
    )


def set_entity_mask_generator(model, entity_model):
    """
    配置并设置模型的掩码生成器。
    """
    model.mask_generator_type = 'entityseg'
    model.confidence_threshold = 0.5
    model.mask_generator = entity_model


# --- 全局模型初始化 ---
model = CorrCLIPSegmentation(clip_type='metaclip_fullcc', model_type='ViT-L-14-quickgelu', dino_type='dino_vitb8',
                             name_path='./configs/my_name.txt', mask_generator=None)

try:
    sam2 = build_sam2("sam2_hiera_l.yaml", "sam2_hiera_large.pt", device='cuda', apply_postprocessing=False)
    sam2 = sam2.half()
except:
    print("SAM2 is not installed")
try:
    entity_model = get_entityseg(cfg_file="mask2former_hornet_3x.yaml", ckpt_path="Mask2Former_hornet_3x_576d0b.pth")
except:
    print("EntitySeg is not installed")

# --- 核心处理函数 ---
# 增加了一个新参数 `segmentation_method` 来接收单选按钮的值
def process_image(img_path, input_text, segmentation_method, points_per_side, pred_iou_thresh, stability_score_thresh, multimask_output):
    # 根据用户的选择，调用不同的配置函数
    if segmentation_method == 'SAM2':
        set_sam_mask_generator(model, sam2, points_per_side, pred_iou_thresh, stability_score_thresh, multimask_output)
    elif segmentation_method == 'EntitySeg':
        set_entity_mask_generator(model, entity_model)
    else:
        raise ValueError("Unknown segmentation method selected.")

    input_img = Image.open(img_path).convert("RGB")

    with open('./configs/my_name.txt', 'r') as file:
        lines = file.readlines()
    name_list = [line.strip() for line in lines]

    if set(name_list) != set(list(dict.fromkeys(item.strip() for item in input_text.split(',')))):
        name_list = list(dict.fromkeys(item.strip() for item in input_text.split(',')))
        with open('./configs/my_name.txt', 'w') as writers:
            for i in range(len(name_list)):
                if i == len(name_list) - 1:
                    writers.write(name_list[i])
                else:
                    writers.write(name_list[i] + '\n')
        writers.close()
        model.generate_category_embeddings('./configs/my_name.txt')

    class_names = {index: value for index, value in enumerate(name_list)}

    img_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])(input_img)
    img_tensor = img_tensor.unsqueeze(0).to('cuda')

    seg_pred = model.predict([img_tensor, img_path], data_samples=None)
    seg_pred = seg_pred.data.cpu().numpy().squeeze()

    # 计算每个类别的中心点
    regions = {}
    for label_id in np.unique(seg_pred):
        labeled_array, _ = label(seg_pred == label_id)
        sizes = np.bincount(labeled_array.ravel())
        max_idx = np.argmax(sizes[1:]) + 1  # 忽略背景区域 (idx=0)

        region = (labeled_array == max_idx)
        y, x = np.where(region)
        center_y, center_x = int(np.mean(y)), int(np.mean(x))

        if y[0] + 10 < region.shape[0] and x[0] + 10 < region.shape[1]:
            regions[label_id] = (center_y, x[0] + 10)
        else:
            regions[label_id] = (center_y, x[0])
    output_str = '; '.join(f"{class_names[idx]}" for idx, coor in regions.items())

    # 创建一个彩色图像用于显示类别名称
    palette = generate_distinct_colors(len(name_list))
    seg_pred_color = palette[seg_pred]

    # 将分割结果与原图融合
    seg_pred_color = (seg_pred_color * 0.7 + np.array(input_img) * 0.3).astype(np.uint8)

    # 创建一个绘图对象
    seg_pred_color = Image.fromarray(seg_pred_color)
    draw = ImageDraw.Draw(seg_pred_color)
    font = ImageFont.load_default(size=max(seg_pred.shape[0], seg_pred.shape[1]) // 50)
    # 遍历 regions 并在图像上绘制文本
    for class_id, (center_y, center_x) in regions.items():
        if class_id in class_names:
            text = class_names[class_id]
            draw.text((center_x, center_y), text, font=font, fill="white", anchor="lt")

    return seg_pred_color, output_str


example_list = [
    ["images/Golden Retriever,Husky,background.jpg", "golden retriever,husky,background"],
    ["images/pikachu,eevee,background.jpg", "pikachu,eevee,background"],
    ["images/animals.png", "cheetah, zebra, rhinoceros, elephant, buffalo, giraffe, antelope, lion, leopard, background"],
    ["images/fruit.jpg", "background, banana, pineapple, broccoli, potato, tomato, chili pepper, kiwi, avocado, orange, lemon, strawberry, cherry tomato, parsley"]
]


# 创建 Gradio 接口
with gr.Blocks() as iface:
    gr.Markdown("# CorrCLIP Segmentation")
    gr.Markdown("Upload an image, provide class names, choose a mask generator, and adjust parameters.")

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="filepath", label="Input Image")
            text_input = gr.Textbox(label="Class Names (comma-separated)", info="e.g., 'cat,dog,background'")

            # 新增：方法选择器
            method_selector = gr.Radio(
                choices=['EntitySeg', 'SAM2'],
                value='EntitySeg',  # 默认是 EntitySeg
                label="Mask Generator"
            )

            # 修改点 1: 在定义时就将折叠组设置为不可见 (visible=False)
            # open=True 确保了它在变得可见时是展开状态
            with gr.Accordion("SAM2 Parameters", open=True, visible=False) as sam_params_group:
                points_per_side_slider = gr.Slider(minimum=4, maximum=64, step=4, value=8, label="Points Per Side")
                pred_iou_thresh_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.4, label="Prediction IoU Threshold")
                stability_score_thresh_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.4, label="Stability Score Threshold")
                multimask_output_checkbox = gr.Checkbox(value=False, label="Multimask Output")

            submit_btn = gr.Button("Segment Image", variant="primary")

        with gr.Column(scale=1):
            img_output = gr.Image(type="pil", label="Segmentation Result")
            label_output = gr.Label(label="Detected Classes")

    gr.Examples(examples=example_list, inputs=[img_input, text_input])


    # --- 交互逻辑 ---
    # 修改点 2: 这个函数现在控制折叠组的可见性
    def toggle_sam_params(selection):
        # 当选择 'SAM2' 时，让参数组可见；否则，让它不可见。
        if selection == 'SAM2':
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)


    # 当方法选择器变化时，调用上面的函数来更新UI
    method_selector.change(
        fn=toggle_sam_params,
        inputs=method_selector,
        outputs=sam_params_group
    )

    # 按钮点击事件保持不变
    submit_btn.click(
        fn=process_image,
        inputs=[
            img_input,
            text_input,
            method_selector,
            points_per_side_slider,
            pred_iou_thresh_slider,
            stability_score_thresh_slider,
            multimask_output_checkbox
        ],
        outputs=[img_output, label_output]
    )


# 启动 Gradio 应用
iface.launch()
