import torch
import torch.nn.functional as F
import os
import json
import sys

from model import SimpleCNN, load_and_preprocess_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

def batch_predict(model_class, device, root_path):
    test_images_dir = os.path.join(root_path, 'test_images')
    json_path = os.path.join(test_images_dir, 'image-list.json')
    model_save_path = os.path.join(root_path, 'models', 'mnist_cnn_best.pth')

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            test_list = json.load(f)
    except FileNotFoundError:
        print(f"错误：未找到文件 {json_path}。请确保文件已创建并位于 test_images 文件夹中。")
        return

    checkpoint = torch.load(model_save_path)
    loaded_model = model_class().to(device)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval() # 设置评估模式

    total_images = len(test_list)
    correct_predictions = 0

    print("-" * 30)
    print(f"--- 批量外部图片测试开始 ({total_images} 张图片) ---")

    for item in test_list:
        image_name = item["filename"]
        true_label = item["label"]
        image_path = os.path.join(test_images_dir, image_name)
        
        try:
            input_tensor = load_and_preprocess_image(image_path, device)
        except FileNotFoundError:
            print(f"警告：图片 {image_name} 未找到，跳过。")
            continue
        except Exception as e:
            print(f"处理图片 {image_name} 失败: {e}，跳过。")
            continue

        with torch.no_grad():
            output = loaded_model(input_tensor)
            # argmax 找到最大概率的索引，即预测的类别
            predicted_class = output.argmax(dim=1).item() 

        is_correct = (predicted_class == true_label)
        
        if is_correct:
            correct_predictions += 1
            status = "正确"
        else:
            status = "错误"

        print(f"图片: {image_name:<20} | 真实值: {true_label} | 预测值: {predicted_class} | 结果: {status}")

    accuracy = (correct_predictions / total_images) * 100
    
    print("-" * 30)
    print(f"批量测试完成！")
    print(f"总图片数: {total_images}, 正确数量: {correct_predictions}")
    print(f"最终准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    batch_predict(SimpleCNN, device, PROJECT_ROOT)