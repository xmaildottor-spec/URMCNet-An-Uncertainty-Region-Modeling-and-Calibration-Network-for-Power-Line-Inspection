import os

# ================= ⚙️ 配置区域 =================
# 1. 请将下面的路径替换为你想要重命名的文件夹路径
target_folder = r'E:\code\dataset\TTPLA\ttpla\TTPLA_512\splitting_imgs_512\test_imgs'

# 2. 起始的数字 (默认从 0 开始)
start_number = 0
# ===============================================

def rename_images_sequentially(folder_path, start_num):
    print(f"🚀 开始重命名文件夹内的图像：{folder_path}\n")
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"❌ 错误：找不到文件夹 '{folder_path}'，请检查路径是否正确。")
        return

    # 支持的图像格式后缀（可根据需要自行添加）
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif')

    # 获取文件夹中的所有文件，并进行排序（保证每次运行顺序一致）
    files = os.listdir(folder_path)
    files.sort()

    count = start_num
    renamed_total = 0

    for filename in files:
        # 提取文件后缀并转换为小写，方便判断
        _, extension = os.path.splitext(filename)
        
        # 判断是否为图像文件
        if extension.lower() in valid_extensions:
            old_filepath = os.path.join(folder_path, filename)
            
            # 构造新的文件名，格式化为 4 位数字，不足补零 (例如: 0000.jpg)
            # 如果你的图片可能超过 10000 张，可以改成 {:05d} (即 00000)
            new_filename = f"{count:04d}{extension.lower()}"
            new_filepath = os.path.join(folder_path, new_filename)

            # 防止原文件和新文件名字正好一样而报错
            if old_filepath == new_filepath:
                count += 1
                continue
                
            # 重命名操作
            try:
                os.rename(old_filepath, new_filepath)
                print(f"✅ 成功: {filename} -> {new_filename}")
                count += 1
                renamed_total += 1
            except FileExistsError:
                print(f"⚠️ 冲突: 目标文件 {new_filename} 已存在，跳过 {filename}。")
            except Exception as e:
                print(f"❌ 错误: 无法重命名 {filename}，原因: {e}")

    print(f"\n🎉 批量重命名完成！共处理了 {renamed_total} 个图像文件。")

if __name__ == '__main__':
    rename_images_sequentially(target_folder, start_number)