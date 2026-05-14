import os
import cv2
import numpy as np

# ================= ⚙️ 配置区域 =================
# 1. 输入的二值 Mask 文件夹路径 (里面是像素值为 0 和 1 的标签图)
mask_base_dir = r'E:\code\dataset\TTPLA\ttpla\splitting_masks_binary'

# 2. 输出的黑白可视化 Mask 文件夹路径
vis_output_base = r'E:\code\dataset\TTPLA\ttpla\splitting_vis_masks_binary'

# 3. 颜色映射表 (二值分割专属)
# 格式: 类别 ID : (B, G, R)
COLOR_MAP = {
    0: (0, 0, 0),          # 背景: 黑色
    1: (255, 255, 255)     # cable (电力线): 白色 (B=255, G=255, R=255)
}
# ===============================================

# 自动推导子文件夹路径映射
splits = {
    'train': {
        'in_dir': os.path.join(mask_base_dir, 'train_masks'),
        'out_dir': os.path.join(vis_output_base, 'train_vis_masks')
    },
    'val': {
        'in_dir': os.path.join(mask_base_dir, 'val_masks'),
        'out_dir': os.path.join(vis_output_base, 'val_vis_masks')
    },
    'test': {
        'in_dir': os.path.join(mask_base_dir, 'test_masks'),
        'out_dir': os.path.join(vis_output_base, 'test_vis_masks')
    }
}

print("🚀 开始为二值 Mask 标签图生成黑白可视化图片...\n")

total_vis = 0

for split_name, paths in splits.items():
    in_dir = paths['in_dir']
    out_dir = paths['out_dir']
    
    if not os.path.exists(in_dir):
        print(f"⚠️ 找不到输入文件夹，已跳过: {in_dir}")
        continue
        
    os.makedirs(out_dir, exist_ok=True)
    
    # 获取所有的 png mask 文件
    mask_files = [f for f in os.listdir(in_dir) if f.endswith('.png')]
    count = 0
    
    for mask_name in mask_files:
        in_path = os.path.join(in_dir, mask_name)
        out_path = os.path.join(out_dir, mask_name)
        
        # 以单通道灰度图模式读取 mask (像素值只有 0 和 1)
        mask = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"❌ 无法读取图片: {mask_name}")
            continue
            
        # 获取图像宽高，并创建一个全黑的 3 通道 (BGR) 彩色图矩阵
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 利用 NumPy 的高级索引，一次性对整张图进行颜色替换
        for class_id, color in COLOR_MAP.items():
            if class_id == 0:
                continue # 背景本来就是初始化的黑色，直接跳过以加快速度
            # 找到 mask 中像素值等于 1 (电力线) 的位置，赋予纯白色
            color_mask[mask == class_id] = color
            
        # 保存黑白可视化图片
        cv2.imwrite(out_path, color_mask)
        count += 1
        total_vis += 1
        
    print(f"✅ [{split_name}] 处理完毕: 成功生成 {count} 张黑白可视化图片。")

print(f"\n🎉 所有二值标签可视化完成！共计生成图片 {total_vis} 张。")
print(f"📂 你的可视化图片存放在这里: {vis_output_base}")