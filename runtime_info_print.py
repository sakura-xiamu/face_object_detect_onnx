# albumentations_hook.py
"""
运行时钩子，在程序启动时替换 albumentations 的版本检查函数
"""
import os
import sys

# 设置环境变量
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'


# 猴子补丁方法
def apply_monkey_patch():
    try:
        # 尝试导入 albumentations
        import albumentations.check_version

        # 替换版本检查函数
        def disabled_check_version(force=False):
            return

        # 应用猴子补丁
        albumentations.check_version.check_version = disabled_check_version
        print("已成功禁用 Albumentations 版本检查")
    except ImportError:
        pass  # 如果还没有导入 albumentations，不做任何事


# 应用猴子补丁
apply_monkey_patch()
