import os
import cv2

# ================= ⚙️ 配置区域 =================
# 1. 输入的原始图片基础文件夹路径
input_base_dir = r'E:\code\dataset\TTPLA\ttpla\splitting_imgs'

# 2. 调整尺寸后保存的新基础文件夹路径 (建议加 _512 后缀区分)
output_base_dir = r'E:\code\dataset\TTPLA\ttpla\splitting_imgs_512'

# 3. 目标尺寸 (宽, 高)
target_size = (512, 512)
# ===============================================

def resize_images_recursive():
    print(f"🚀 开始递归调整可见光图像尺寸至 {target_size[0]}x{target_size[1]}...")
    
    if not os.path.exists(input_base_dir):
        print(f"❌ 错误：找不到输入文件夹 {input_base_dir}")
        return

    # 支持常见的可见光图像格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    count = 0

    for root, dirs, files in os.walk(input_base_dir):
        # 筛选出图片文件
        img_files = [f for f in files if f.lower().endswith(valid_extensions)]
        
        if not img_files:
            continue

        # 计算相对路径，以便在输出目录中克隆文件夹结构
        rel_path = os.path.relpath(root, input_base_dir)
        target_dir = os.path.join(output_base_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        for file_name in img_files:
            input_path = os.path.join(root, file_name)
            
            # 以默认彩色模式 (BGR) 读取原图
            img = cv2.imread(input_path)
            
            if img is None:
                print(f"⚠️ 无法读取图片，已跳过: {input_path}")
                continue

            # 🌟 核心区别：大幅度缩小图像时，使用 INTER_AREA 算法效果最佳，能保留更多平滑细节
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

            # 保存图片 (保持原有的格式后缀，例如 .jpg 依然保存为 .jpg)
            output_img_path = os.path.join(target_dir, file_name)
            
            cv2.imwrite(output_img_path, resized_img)
            count += 1
            
            if count % 200 == 0:
                print(f"🔄 已处理 {count} 张图片...")

    print(f"\n🎉 图像尺寸调整全部完成！")
    print(f"✅ 成功处理: {count} 张图片")
    print(f"📂 新的数据集完全保持了原有的子文件夹结构，保存在: {output_base_dir}")

if __name__ == '__main__':
    resize_images_recursive()