# -*- coding: utf-8 -*-
"""
心理健康预测器 - 交互式预测脚本
MindScreen Mental Health Predictor v1.0

使用方法:
  python predict.py              # 交互式模式
  python predict.py --help       # 查看帮助
  python predict.py --demo       # 演示模式
"""

import os
import sys
import joblib

# ============================================================
# 模型加载器
# ============================================================
class MentalHealthPredictor:
    """心理健康预测器"""
    
    def __init__(self, model_dir="model"):
        """初始化预测器，加载模型"""
        self.model_dir = model_dir
        self.model = None
        self.config = None
        self._load_model()
    
    def _load_model(self):
        """加载模型和配置"""
        model_path = os.path.join(self.model_dir, "mental_health_model.pkl")
        config_path = os.path.join(self.model_dir, "config.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}\n"
                f"请先运行 'python save_model.py' 训练并保存模型"
            )
        
        self.model = joblib.load(model_path)
        self.config = joblib.load(config_path)
        
    def predict(self, screen_time, social_time, sleep_hours, age, gender, exercise):
        """
        预测心理健康指标
        
        参数:
            screen_time: 每日屏幕使用时长(分钟)
            social_time: 社交媒体使用时长(分钟)
            sleep_hours: 睡眠时间(小时)
            age: 年龄
            gender: 性别 ("Female", "Male", "Other")
            exercise: 身体活动时间(分钟)
        
        返回:
            dict: {"stress_level": float, "anxiety_level": float}
        """
        # 性别编码
        gender_code = self.config["gender_encoding"].get(gender, 1)
        
        # 构建输入
        import numpy as np
        X = np.array([[screen_time, social_time, sleep_hours, age, gender_code, exercise]])
        
        # 预测
        prediction = self.model.predict(X)[0]
        
        return {
            "stress_level": round(prediction[0], 2),
            "anxiety_level": round(prediction[1], 2)
        }
    
    def get_info(self):
        """获取模型信息"""
        return self.config


# ============================================================
# 交互式界面
# ============================================================
def print_header():
    """打印欢迎界面"""
    print("\n" + "=" * 60)
    print("       🧠 MindScreen - 心理健康预测系统 v1.0")
    print("=" * 60)
    print("  基于机器学习的压力与焦虑指数预测工具")
    print("  输入您的日常行为数据，获取心理健康评估")
    print("=" * 60)


def print_help():
    """打印帮助信息"""
    print("""
使用方法:
  python predict.py              启动交互式预测
  python predict.py --help       显示此帮助信息
  python predict.py --demo       运行演示预测
  python predict.py --info       显示模型信息

交互模式命令:
  输入数据后按回车进行预测
  输入 'q' 或 'quit' 退出程序
  输入 'help' 显示帮助
  输入 'demo' 运行演示
  输入 'info' 显示模型信息
  输入 'clear' 清屏

预测指标说明:
  压力指数 (Stress Level):  1-10, 数值越高压力越大
  焦虑指数 (Anxiety Level): 1-5,  数值越高焦虑越重
""")


def get_stress_emoji(level):
    """根据压力等级返回表情"""
    if level <= 3:
        return "😊 低压力"
    elif level <= 5:
        return "😐 中等压力"
    elif level <= 7:
        return "😟 较高压力"
    else:
        return "😰 高压力"


def get_anxiety_emoji(level):
    """根据焦虑等级返回表情"""
    if level <= 1.5:
        return "😌 轻微焦虑"
    elif level <= 2.5:
        return "😕 轻度焦虑"
    elif level <= 3.5:
        return "😧 中度焦虑"
    else:
        return "😨 重度焦虑"


def get_input(prompt, input_type="int", default=None, options=None):
    """获取用户输入并验证"""
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} [{default}]: ").strip()
                if user_input == "":
                    return default
            else:
                user_input = input(f"{prompt}: ").strip()
            
            # 检查退出命令
            if user_input.lower() in ['q', 'quit', 'exit']:
                return None
            
            # 类型转换和验证
            if input_type == "int":
                value = int(user_input)
                if value < 0:
                    print("  ⚠️ 请输入非负整数")
                    continue
                return value
            elif input_type == "float":
                value = float(user_input)
                if value < 0:
                    print("  ⚠️ 请输入非负数")
                    continue
                return value
            elif input_type == "choice":
                if user_input in options:
                    return user_input
                # 尝试数字选择
                try:
                    idx = int(user_input) - 1
                    if 0 <= idx < len(options):
                        return options[idx]
                except:
                    pass
                print(f"  ⚠️ 请选择: {', '.join(options)}")
                continue
            else:
                return user_input
                
        except ValueError:
            print(f"  ⚠️ 输入格式错误，请重新输入")
        except KeyboardInterrupt:
            return None


def run_prediction(predictor):
    """运行一次预测"""
    print("\n" + "-" * 40)
    print("请输入以下信息 (输入 q 退出)")
    print("-" * 40)
    
    # 获取输入
    screen_time = get_input("📱 每日屏幕时间(分钟)", "int", 300)
    if screen_time is None:
        return False
    
    social_time = get_input("💬 社交媒体时间(分钟)", "int", 120)
    if social_time is None:
        return False
    
    sleep_hours = get_input("😴 睡眠时间(小时)", "float", 7.0)
    if sleep_hours is None:
        return False
    
    age = get_input("🎂 年龄", "int", 25)
    if age is None:
        return False
    
    print("⚧️ 性别: 1=Female, 2=Male, 3=Other")
    gender = get_input("   请选择", "choice", "Male", ["Female", "Male", "Other"])
    if gender is None:
        return False
    
    exercise = get_input("🏃 运动时间(分钟)", "int", 30)
    if exercise is None:
        return False
    
    # 预测
    print("\n⏳ 正在分析...")
    result = predictor.predict(screen_time, social_time, sleep_hours, age, gender, exercise)
    
    # 显示结果
    print("\n" + "=" * 40)
    print("           📊 预测结果")
    print("=" * 40)
    print(f"\n  压力指数: {result['stress_level']:.1f} / 10")
    print(f"           {get_stress_emoji(result['stress_level'])}")
    print(f"\n  焦虑指数: {result['anxiety_level']:.1f} / 5")
    print(f"           {get_anxiety_emoji(result['anxiety_level'])}")
    print("\n" + "=" * 40)
    
    # 建议
    print("\n💡 健康建议:")
    if result['stress_level'] > 6:
        print("  • 压力较高，建议增加休息和放松活动")
    if result['anxiety_level'] > 3:
        print("  • 焦虑偏高，建议减少社交媒体使用")
    if sleep_hours < 7:
        print("  • 睡眠不足，建议保证7-8小时睡眠")
    if exercise < 30:
        print("  • 运动偏少，建议每天至少30分钟运动")
    if screen_time > 360:
        print("  • 屏幕时间过长，建议适当休息眼睛")
    
    return True


def run_demo(predictor):
    """运行演示"""
    print("\n" + "=" * 50)
    print("           🎯 演示模式")
    print("=" * 50)
    
    demos = [
        {"name": "健康生活者", "data": (180, 60, 8.0, 30, "Male", 60)},
        {"name": "熬夜工作族", "data": (480, 240, 5.5, 28, "Female", 10)},
        {"name": "学生党", "data": (360, 180, 6.5, 20, "Other", 20)},
    ]
    
    for demo in demos:
        print(f"\n📋 案例: {demo['name']}")
        print("-" * 40)
        data = demo['data']
        print(f"  屏幕时间: {data[0]}分钟 | 社交媒体: {data[1]}分钟")
        print(f"  睡眠: {data[2]}小时 | 年龄: {data[3]} | 性别: {data[4]}")
        print(f"  运动时间: {data[5]}分钟")
        
        result = predictor.predict(*data)
        print(f"\n  ➡️ 压力指数: {result['stress_level']:.1f} {get_stress_emoji(result['stress_level'])}")
        print(f"  ➡️ 焦虑指数: {result['anxiety_level']:.1f} {get_anxiety_emoji(result['anxiety_level'])}")


def show_info(predictor):
    """显示模型信息"""
    info = predictor.get_info()
    print("\n" + "=" * 50)
    print("           ℹ️ 模型信息")
    print("=" * 50)
    print(f"\n  模型名称: {info['model_name']}")
    print(f"  版本: {info['version']}")
    print(f"  算法: {info['model_type']}")
    print(f"\n  性能指标:")
    print(f"    压力预测 R²: {info['performance']['stress_level_r2']:.4f}")
    print(f"    焦虑预测 R²: {info['performance']['anxiety_level_r2']:.4f}")
    print(f"\n  输入特征:")
    for f in info['features']:
        print(f"    • {f['name']}: {f['description']}")
    print(f"\n  输出:")
    for o in info['outputs']:
        print(f"    • {o['name']}: {o['description']} ({o['range']})")


def interactive_mode(predictor):
    """交互式模式"""
    print_header()
    print("\n输入 'help' 查看帮助, 'q' 退出\n")
    
    while True:
        try:
            cmd = input("\n按回车开始预测 (或输入命令): ").strip().lower()
            
            if cmd in ['q', 'quit', 'exit']:
                print("\n👋 感谢使用 MindScreen，再见！\n")
                break
            elif cmd == 'help':
                print_help()
            elif cmd == 'demo':
                run_demo(predictor)
            elif cmd == 'info':
                show_info(predictor)
            elif cmd == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_header()
            else:
                if not run_prediction(predictor):
                    print("\n👋 感谢使用 MindScreen，再见！\n")
                    break
                    
        except KeyboardInterrupt:
            print("\n\n👋 感谢使用 MindScreen，再见！\n")
            break


# ============================================================
# 主程序
# ============================================================
def main():
    """主函数"""
    # 解析命令行参数
    args = sys.argv[1:]
    
    if '--help' in args or '-h' in args:
        print_header()
        print_help()
        return
    
    # 加载模型
    try:
        predictor = MentalHealthPredictor()
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("\n请先运行以下命令训练并保存模型:")
        print("  python save_model.py")
        return
    
    if '--demo' in args:
        print_header()
        run_demo(predictor)
    elif '--info' in args:
        print_header()
        show_info(predictor)
    else:
        interactive_mode(predictor)


if __name__ == "__main__":
    main()
