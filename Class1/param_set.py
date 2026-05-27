"""
参数配置模块
集中管理全项目所有参数，使用argparse实现参数配置
"""

import argparse
import torch


def get_args():
    """获取所有命令行参数"""
    
    parser = argparse.ArgumentParser(
        description='3D医学影像分割项目 - BraTS Brain Segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ===========================================
    # 基础配置
    # ===========================================
    parser.add_argument('--mode', type=str, default='test',
                        choices=['train', 'test', 'train_test'],
                        help='运行模式: train=仅训练, test=仅测试, train_test=训练后测试')
    
    parser.add_argument('--data_dir', type=str, 
                        default=r'D:\PythonProgram\BrainSeg',
                        help='数据根目录（包含train/val/test子目录）')
    
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='检查点保存目录')
    
    parser.add_argument('--test_output_dir', type=str, default='test_results',
                        help='测试结果输出目录')
    
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'],
                        help='训练设备')
    
    # ===========================================
    # 数据预处理参数
    # ===========================================
    parser.add_argument('--target_spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help='目标体素间距 (x, y, z)')
    
    parser.add_argument('--target_size', type=int, nargs=3, default=[128, 128, 128],
                        help='目标尺寸 (H, W, D)')
    
    parser.add_argument('--norm_method', type=str, default='z-score',
                        choices=['z-score', 'min-max'],
                        help='图像强度标准化方法')
    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader的进程数')
    
    # ===========================================
    # 模型参数
    # ===========================================
    parser.add_argument('--model', type=str, default='transbts',
                        choices=['transbts', 'kiunet'],
                        help='模型架构')
    
    parser.add_argument('--in_channels', type=int, default=4,
                        help='输入通道数（模态数）')
    
    parser.add_argument('--num_classes', type=int, default=2,
                        help='输出类别数')
    
    parser.add_argument('--base_filters', type=int, default=32,
                        help='基础滤波器数量')
    
    # TransBTS特有参数
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='TransBTS Transformer嵌入维度')
    
    parser.add_argument('--num_transformer_layers', type=int, default=4,
                        help='TransBTS Transformer层数')
    
    # ===========================================
    # 训练参数
    # ===========================================
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批次大小')
    
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='训练epoch数')
    
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    
    parser.add_argument('--use_scheduler', action='store_true',
                        help='是否使用学习率调度器')
    
    parser.add_argument('--log_interval', type=int, default=10,
                        help='每隔N个batch打印一次训练信息')
    
    parser.add_argument('--save_interval', type=int, default=1,
                        help='每隔N个epoch保存一次检查点')
    
    # ===========================================
    # 数据增强参数
    # ===========================================
    parser.add_argument('--augment', action='store_true', default=True,
                        help='是否启用数据增强（训练集）')
    
    parser.add_argument('--aug_flip_prob', type=float, default=0.5,
                        help='随机翻转概率')
    
    parser.add_argument('--aug_noise_std', type=float, default=0.01,
                        help='高斯噪声标准差')
    
    # ===========================================
    # 测试参数
    # ===========================================
    parser.add_argument('--save_predictions', action='store_true',
                        help='是否保存测试集预测结果')
    
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='测试时批次大小（通常为1）')
    
    # ===========================================
    # 断点续跑参数
    # ===========================================
    parser.add_argument('--auto_resume', action='store_true', default=True,
                        help='自动检测并续跑（无需手动指定）')
    
    parser.add_argument('--force_restart', action='store_true',
                        help='强制重新开始训练（忽略已有检查点）')
    
    # ===========================================
    # 其他参数
    # ===========================================
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help='是否pin memory（加速GPU训练）')
    
    parser.add_argument('--drop_last', action='store_true', default=True,
                        help='是否丢弃不完整的最后一批')
    
    return parser.parse_args()


def get_config():
    """获取配置字典（从args转换）"""
    args = get_args()
    
    # 转换为字典
    config = vars(args)
    
    # 设备自动选择
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 将列表参数转换为元组
    config['target_spacing'] = tuple(config['target_spacing'])
    config['target_size'] = tuple(config['target_size'])
    
    return config


def print_config(config):
    """打印配置信息"""
    print("="*60)
    print("当前配置参数")
    print("="*60)
    
    # 按类别打印
    categories = {
        '基础配置': ['mode', 'data_dir', 'checkpoint_dir', 'test_output_dir', 'device'],
        '数据预处理': ['target_spacing', 'target_size', 'norm_method', 'num_workers'],
        '模型': ['model', 'in_channels', 'num_classes', 'base_filters', 'embed_dim', 'num_transformer_layers'],
        '训练': ['batch_size', 'num_epochs', 'learning_rate', 'weight_decay', 'use_scheduler', 'log_interval'],
        '数据增强': ['augment', 'aug_flip_prob', 'aug_noise_std'],
        '测试': ['save_predictions', 'test_batch_size'],
        '断点续跑': ['auto_resume', 'force_restart'],
        '其他': ['seed', 'pin_memory', 'drop_last']
    }
    
    for category, params in categories.items():
        print(f"\n[{category}]")
        for param in params:
            if param in config:
                value = config[param]
                print(f"  {param:25s}: {value}")
    
    print("\n" + "="*60)


# 测试代码
if __name__ == '__main__':
    config = get_config()
    print_config(config)
