import os
import json
import numpy as np
import cv2

# ================= ⚙️ 配置区域 =================
# 1. 之前划分好的 JSON 文件夹的基础路径
json_base_dir = r'E:\code\dataset\TTPLA\ttpla\splitting_jsons'

# 2. 准备输出 Mask 图片的基础路径
mask_output_base = r'E:\code\dataset\TTPLA\ttpla\splitting_masks'

# 3. 类别与灰度像素值的映射字典 (已根据你的数据集真实标签更新！)
# 背景默认为 0。'void' 标签因为不在这里面，会被自动跳过并保持为 0（背景）
CLASS_MAPPING = {
    'cable': 1,           # 电力线，像素值设为 1
    'tower_lattice': 2,   # 桁架杆塔，统一归为杆塔类，像素值设为 2
    'tower_tucohy': 2,    # 管型杆塔，统一归为杆塔类，像素值设为 2
    'tower_wooden': 2     # 木质电线杆，统一归为杆塔类，像素值设为 2
}
# ===============================================

# 自动推导对应的子文件夹映射
splits = {
    'train': {
        'json_dir': os.path.join(json_base_dir, 'train_jsons'),
        'mask_dir': os.path.join(mask_output_base, 'train_masks')
    },
    'val': {
        'json_dir': os.path.join(json_base_dir, 'val_jsons'),
        'mask_dir': os.path.join(mask_output_base, 'val_masks')
    },
    'test': {
        'json_dir': os.path.join(json_base_dir, 'test_jsons'),
        'mask_dir': os.path.join(mask_output_base, 'test_masks')
    }
}

print("🚀 开始将 JSON 转换为 Mask 标签图...\n")

total_masks = 0

for split_name, paths in splits.items():
    json_dir = paths['json_dir']
    mask_dir = paths['mask_dir']
    
    if not os.path.exists(json_dir):
        print(f"⚠️ 找不到 JSON 文件夹，已跳过: {json_dir}")
        continue
        
    os.makedirs(mask_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    count = 0
    
    for json_name in json_files:
        json_path = os.path.join(json_dir, json_name)
        
        # 读取 JSON 文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 获取图像的高和宽
        img_h = data.get('imageHeight')
        img_w = data.get('imageWidth')
        
        if img_h is None or img_w is None:
            print(f"❌ 警告: {json_name} 中缺少 imageHeight 或 imageWidth 字段，跳过。")
            continue
            
        # 初始化一个全黑的背景 (像素值为 0)，数据类型必须是 uint8
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        # 遍历 JSON 中的所有多边形形状
        for shape in data.get('shapes', []):
            label = shape['label']
            points = shape['points']
            shape_type = shape.get('shape_type', 'polygon')
            
            # 如果标签不在映射表里（比如 'void'），就直接跳过
            if label not in CLASS_MAPPING:
                continue
                
            class_id = CLASS_MAPPING[label]
            
            # 将坐标点转换为 NumPy 的 int32 格式，这是 OpenCV 绘图的要求
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # 根据 shape_type 绘制
            if shape_type == 'polygon':
                # 用对应的类别 ID 填充多边形
                cv2.fillPoly(mask, [pts], color=class_id)
            else:
                pass
                
        # 保存 Mask 图片（必须保存为 .png，防止无损压缩破坏类别 ID）
        base_name = os.path.splitext(json_name)[0]
        mask_save_path = os.path.join(mask_dir, base_name + '.png')
        
        cv2.imwrite(mask_save_path, mask)
        count += 1
        total_masks += 1
        
    print(f"✅ [{split_name}] 处理完毕: 成功生成 {count} 张 Mask 图片。")

print(f"\n🎉 所有 JSON 转换完成！共计生成 Mask 图片 {total_masks} 张。")
print(f"📂 你的 Mask 标签存放在这里: {mask_output_base}")