import torch
from PIL import Image, ImageDraw, ImageFont


def create_logger(file_path=None):
    import logging


    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    if file_path is None:
        os.makedirs('./process/output', exist_ok=True)
        file_path = './process/output/output.log'
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)

    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    

    return logger
    
def get_num_of_gpu():
    if torch.cuda.is_available():
        return torch.cuda._device_count_nvml()
    else:
        return 0
    

def resize_and_concatenate(image1, image2):
    width1, height1 = image1.size
    width2, height2 = image2.size
    
    max_width = max(width1, width2)
    max_height = max(height1, height2)
    
    resized_image1 = image1.resize((max_width, max_height), Image.LANCZOS)
    resized_image2 = image2.resize((max_width, max_height), Image.LANCZOS)
    
    new_image = Image.new('RGB', (max_width * 2, max_height))
    
    new_image.paste(resized_image1, (0, 0))
    new_image.paste(resized_image2, (max_width, 0))
    
    return new_image



def concatenate_images_with_reference(image_list, font_size=20, margin=3):
    if not image_list:
        return None
    interval_margin = 40
    line_width = 5
    
    width, height = image_list[0].size
    # width_list = [_.size[0] for _ in image_list]
    # height_list = [_.size[1] for _ in image_list]
    # width = max(width_list)
    # height = max(height_list)
    
    new_height = height + font_size + 2 * margin
    
    total_width = width * len(image_list) + interval_margin * (len(image_list) - 1) + line_width * (len(image_list) - 1)
    
    result_image = Image.new('RGB', (total_width, new_height), color='white')
    
    draw = ImageDraw.Draw(result_image)
    
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        font = ImageFont.truetype(font.path, font_size)
    
    for i, img in enumerate(image_list):
        
        x_offset = i * (width + interval_margin + line_width)
        
        result_image.paste(img, (x_offset, font_size + 2 * margin))
        
        if i == 0:
            label = "Reference Image"
        else:
            label = "Candidate Image"
        
        
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = x_offset + (width - text_width) // 2
        y = margin + (font_size - text_height) // 2
        
        draw.text((x, y), label, fill="black", font=font)
        
        if i < len(image_list) - 1:
            line_x = x_offset + width + interval_margin // 2
            draw.line([(line_x, 0), (line_x, new_height)], fill="black", width=line_width)

    return result_image


def resize_image_ratio(image, max_length=800):

    width, height = image.size

    aspect_ratio = width / height
    
    if width > height:
        new_width = max_length
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_length
        new_width = int(new_height * aspect_ratio)
    
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_image