# video_evaluation_server.py - Qwen3VL版视频/图片评估API服务器
import gradio as gr
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os
import json
import socket

class MediaEvaluationService:
    def __init__(self):
        self.device = "cuda:0"
        self.model_path = "../models/Qwen3-VL-8B-Instruct"
        self.model = None
        self.processor = None
        self.load_model()

    def load_model(self):
        """加载Qwen3VL模型"""
        print("正在加载Qwen3VL模型...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": self.device},
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        print("模型加载完成！")

    def ask_vlm(self, media_path, question_text, media_type="video"):
        """基础的视频/图片问答功能

        Args:
            media_path: 媒体文件路径
            question_text: 问题文本
            media_type: 媒体类型 ("video" 或 "image")
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": media_type,
                            media_type: media_path,
                        },
                        {"type": "text", "text": question_text},
                    ],
                }
            ]

            # 准备输入
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)

            # 生成输出
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)

            # 提取生成的部分（去除输入tokens）
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # 解码
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            response = output_text[0].strip()
            return response

        except Exception as e:
            return f"分析出错: {str(e)}"

# 全局服务实例
service = MediaEvaluationService()

def analyze_mismatches(media_path, original_prompt, media_type="video"):
    """分析媒体与原始提示之间的不匹配

    Args:
        media_path: 媒体文件路径
        original_prompt: 原始提示词
        media_type: 媒体类型 ("video" 或 "image")
    """
    if not os.path.exists(media_path):
        return f"❌ 文件不存在: {media_path}"

    media_name = "video" if media_type == "video" else "image"

    question_text = f"""
    Analyze the {media_name} and identify all mismatches with the original prompt.

    Original prompt: "{original_prompt}"

    Instructions:
    1. List ALL elements from the prompt that are missing in the {media_name}.
    2. List ALL elements from the prompt that appear incorrectly (wrong quantity, appearance, movement, timing, etc.).
    """

    if media_type == "video":
        question_text += """
    3. Consider video-specific aspects like motion, transitions, camera movement, and temporal consistency.
    """
    else:
        question_text += """
    3. Consider image-specific aspects like composition, lighting, perspective, and visual quality.
    """

    question_text += """
    4. Be precise and specific in your analysis.

    Format your response as a numbered list of issues ONLY.
    """

    return service.ask_vlm(media_path, question_text, media_type)

def get_enhanced_prompt(media_path, original_prompt, diagnosis, media_type="video"):
    """获取增强的提示词

    Args:
        media_path: 媒体文件路径
        original_prompt: 原始提示词
        diagnosis: 诊断结果
        media_type: 媒体类型 ("video" 或 "image")
    """
    if not os.path.exists(media_path):
        return f"❌ 文件不存在: {media_path}"

    media_name = "video" if media_type == "video" else "image"
    model_type = "video generation" if media_type == "video" else "image generation"

    question_text = f"""
    You are an expert prompt engineer for {model_type} models.

    Original prompt: "{original_prompt}"

    Issues with the current {media_name}:
    {diagnosis}

    Instructions:
    1. Create an improved prompt that will help the {model_type} model better match the original intention.
    2. Add specific details about """

    if media_type == "video":
        question_text += """motion, camera movement, transitions, timing, or visual elements to address the identified issues.
    3. Maintain the core idea and style of the original prompt - do not add unrelated concepts.
    4. Consider video-specific aspects like frame rate, duration, motion smoothness, and visual continuity.
    """
    else:
        question_text += """composition, lighting, perspective, colors, or visual elements to address the identified issues.
    3. Maintain the core idea and style of the original prompt - do not add unrelated concepts.
    4. Consider image-specific aspects like visual balance, depth, and overall aesthetic.
    """

    question_text += """5. The goal is to get a """ + media_name + """ closer to what was originally intended.
    6. Use techniques like emphasis words, specific quantities, spatial relationships, temporal descriptions, or other details as needed.

    Return only one well-structured, fluent sentence without any explanations.
    """

    return service.ask_vlm(media_path, question_text, media_type)

def get_negative_prompt(media_path, original_prompt, diagnosis, media_type="video"):
    """获取负面提示词

    Args:
        media_path: 媒体文件路径
        original_prompt: 原始提示词
        diagnosis: 诊断结果
        media_type: 媒体类型 ("video" 或 "image")
    """
    if not os.path.exists(media_path):
        return f"❌ 文件不存在: {media_path}"

    media_name = "video" if media_type == "video" else "image"

    question_text = f"""
    Based on the original prompt and analysis of the current {media_name}, list elements that should be avoided in {media_name} generation.

    Original prompt: "{original_prompt}"

    Current {media_name} issues:
    {diagnosis}

    Instructions:
    """

    if media_type == "video":
        question_text += """1. List quality issues to avoid (e.g., blurry frames, inconsistent motion, poor transitions, flickering, artifacts)
    """
    else:
        question_text += """1. List quality issues to avoid (e.g., blur, noise, distortion, poor composition, artifacts)
    """

    question_text += f"""2. List unwanted visual or motion elements based on the diagnosis
    3. DO NOT include any objects or concepts that are actually wanted from the original prompt.
    4. Focus on technical and quality aspects specific to {media_name} generation.

    Return only comma-separated terms for negative prompting.
    """

    return service.ask_vlm(media_path, question_text, media_type)

def evaluate_media_complete(media_path, original_prompt, media_type="video"):
    """完整的媒体评估

    Args:
        media_path: 媒体文件路径
        original_prompt: 原始提示词
        media_type: 媒体类型 ("video" 或 "image")
    """
    if not os.path.exists(media_path):
        return json.dumps({
            "error": f"文件不存在: {media_path}",
            "enhanced_prompt": "",
            "negative_prompt": "",
            "diagnosis": "",
            "media_type": media_type,
            "success": False
        }, ensure_ascii=False, indent=2)

    try:
        media_name = "视频" if media_type == "video" else "图片"
        print(f"🎬 开始完整评估{media_name}: {media_path}")

        # 1. 分析不匹配
        diagnosis = analyze_mismatches(media_path, original_prompt, media_type)
        print("✅ 诊断完成")

        # 2. 获取增强提示
        enhanced = get_enhanced_prompt(media_path, original_prompt, diagnosis, media_type)
        print("✅ 增强提示完成")

        # 3. 获取负面提示
        negative = get_negative_prompt(media_path, original_prompt, diagnosis, media_type)
        print("✅ 负面提示完成")

        result = {
            "enhanced_prompt": enhanced,
            "negative_prompt": negative,
            "diagnosis": diagnosis,
            "media_type": media_type,
            "success": True
        }

        print("✅ 完整评估完成")
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        error_result = {
            "error": f"评估失败: {str(e)}",
            "enhanced_prompt": "",
            "negative_prompt": "",
            "diagnosis": "",
            "media_type": media_type,
            "success": False
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)

def evaluate_media_two_stage(media_path, original_prompt, media_type="video"):
    """两步评估"""
    if not os.path.exists(media_path):
        return json.dumps({
            "error": f"文件不存在: {media_path}",
            "enhanced_prompt": "",
            "negative_prompt": "",
            "diagnosis": "",
            "media_type": media_type,
            "success": False
        }, ensure_ascii=False, indent=2)

    try:
        media_name = "视频" if media_type == "video" else "图片"
        print(f"🎬 开始两步评估{media_name}: {media_path}")

        # Step 1: 获取诊断
        diagnosis = analyze_mismatches(media_path, original_prompt, media_type)
        print("✅ 第一步：诊断完成")

        # Step 2: 获取增强+负面提示（在同一次VLM调用中）
        model_type = "video" if media_type == "video" else "image"
        question_text = f"""
        You are a {model_type} prompt analysis and engineering expert.

        Original prompt: "{original_prompt}"

        {model_type.capitalize()} issues identified:
        {diagnosis}

        Instructions:
        1. Based on the diagnosis, create an improved prompt that fixes the mismatches.
        2. Maintain the core meaning of the original prompt while improving clarity, accuracy, and temporal consistency.
        3. Then, list the issues or qualities to AVOID in future generations (e.g., blur, artifact, flicker).
        4. Be concise and fluent.

        Format:
        Enhanced prompt:
        <one improved sentence>
        ###
        Negative prompt:
        <comma-separated quality issues>
        """

        response = service.ask_vlm(media_path, question_text, media_type)

        enhanced_prompt = ""
        negative_prompt = ""

        if isinstance(response, str):
            sections = response.strip().split("###")
            if len(sections) == 2:
                enhanced_prompt = sections[0].replace("Enhanced prompt:", "").strip()
                negative_prompt = sections[1].replace("Negative prompt:", "").strip()

        result = {
            "enhanced_prompt": enhanced_prompt,
            "negative_prompt": negative_prompt,
            "diagnosis": diagnosis,
            "media_type": media_type,
            "success": True
        }

        print("✅ 两步评估完成")
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        error_result = {
            "error": f"两步评估失败: {str(e)}",
            "enhanced_prompt": "",
            "negative_prompt": "",
            "diagnosis": "",
            "media_type": media_type,
            "success": False
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)

def evaluate_media_one_stage(media_path, original_prompt, media_type="video"):
    """一步式评估"""
    if not os.path.exists(media_path):
        return json.dumps({
            "error": f"文件不存在: {media_path}",
            "enhanced_prompt": "",
            "negative_prompt": "",
            "diagnosis": "",
            "media_type": media_type,
            "success": False
        }, ensure_ascii=False, indent=2)

    try:
        media_name = "视频" if media_type == "video" else "图片"
        print(f"🎬 开始一步式评估{media_name}: {media_path}")

        model_type = "video" if media_type == "video" else "image"
        question_text = f"""
        You are a {model_type} understanding and prompt optimization expert.

        Original prompt: "{original_prompt}"

        Instructions:

        1. Analyze the {model_type} and list mismatches between the prompt and the visual content.
           - Missing or incorrect elements, motion inconsistencies, wrong timing, etc.
        2. Based on the analysis, propose an improved version of the prompt to fix these issues.
           - Keep the original intent and style.
        3. Finally, list quality issues or undesirable features that should be avoided in future generations.
           - e.g., blur, flicker, artifacts, overexposure.

        Format:
        Diagnosis:
        <numbered list of mismatches>
        ###
        Enhanced prompt:
        <one improved sentence>
        ###
        Negative prompt:
        <comma-separated quality terms>
        """

        response = service.ask_vlm(media_path, question_text, media_type)

        diagnosis = ""
        enhanced_prompt = ""
        negative_prompt = ""

        if isinstance(response, str):
            sections = response.strip().split("###")
            if len(sections) == 3:
                diagnosis = sections[0].replace("Diagnosis:", "").strip()
                enhanced_prompt = sections[1].replace("Enhanced prompt:", "").strip()
                negative_prompt = sections[2].replace("Negative prompt:", "").strip()

        result = {
            "diagnosis": diagnosis,
            "enhanced_prompt": enhanced_prompt,
            "negative_prompt": negative_prompt,
            "media_type": media_type,
            "success": True
        }

        print("✅ 一步式评估完成")
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        error_result = {
            "error": f"一步式评估失败: {str(e)}",
            "enhanced_prompt": "",
            "negative_prompt": "",
            "diagnosis": "",
            "media_type": media_type,
            "success": False
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)


# 创建Gradio界面
with gr.Blocks(title="媒体评估服务") as demo:
    gr.Markdown("# 🎥 Qwen3VL 视频/图片评估服务")
    gr.Markdown("基于Qwen3VL的视频和图片质量评估和提示词优化服务")

    with gr.Tab("📱 Web界面"):
        with gr.Row():
            with gr.Column():
                media_type_input = gr.Radio(
                    choices=["video", "image"],
                    value="video",
                    label="📂 媒体类型"
                )
                media_input = gr.File(label="📹 上传视频或图片文件")
                prompt_input = gr.Textbox(
                    label="🎯 原始提示词",
                    placeholder="输入生成媒体时使用的原始提示词...",
                    lines=3
                )

                with gr.Row():
                    mismatch_btn = gr.Button("🔍 分析不匹配", variant="secondary")
                    complete_btn = gr.Button("🚀 完整评估", variant="primary")

            with gr.Column():
                output_area = gr.Textbox(label="📋 评估结果", lines=15)

        # 事件绑定
        def web_mismatch_analysis(media_file, prompt, media_type):
            if media_file is None:
                return "请上传文件"
            return analyze_mismatches(media_file.name, prompt, media_type)

        def web_complete_evaluation(media_file, prompt, media_type):
            if media_file is None:
                return "请上传文件"
            return evaluate_media_complete(media_file.name, prompt, media_type)

        mismatch_btn.click(
            web_mismatch_analysis,
            inputs=[media_input, prompt_input, media_type_input],
            outputs=output_area
        )

        complete_btn.click(
            web_complete_evaluation,
            inputs=[media_input, prompt_input, media_type_input],
            outputs=output_area
        )

    with gr.Tab("🔧 API测试"):
        gr.Markdown("## API调用测试")

        with gr.Row():
            with gr.Column():
                api_media_type = gr.Radio(
                    choices=["video", "image"],
                    value="video",
                    label="媒体类型"
                )
                api_media_path = gr.Textbox(
                    label="文件路径",
                    placeholder="/path/to/your/media.mp4"
                )
                api_prompt = gr.Textbox(
                    label="原始提示词",
                    value="A cat playing with a ball"
                )

                with gr.Row():
                    test_mismatch_btn = gr.Button("测试不匹配分析")
                    test_complete_btn = gr.Button("测试完整评估")

            with gr.Column():
                api_output = gr.Textbox(label="API响应", lines=15)

        test_mismatch_btn.click(
            analyze_mismatches,
            inputs=[api_media_path, api_prompt, api_media_type],
            outputs=api_output
        )

        test_complete_btn.click(
            evaluate_media_complete,
            inputs=[api_media_path, api_prompt, api_media_type],
            outputs=api_output
        )

    with gr.Tab("📖 API文档"):
        gr.Markdown("""
        ## 🛠️ API调用方法

        ### Python调用示例：

        ```python
        from gradio_client import Client
        client = Client("http://localhost:19994")

        # 方法1: 完整评估 (推荐)
        # 评估视频
        result_json = client.predict(
            "video.mp4",           # 媒体路径
            "original prompt",     # 原始提示词
            "video",               # 媒体类型
            api_name="/evaluate_media_complete"
        )

        # 评估图片
        result_json = client.predict(
            "image.png",           # 媒体路径
            "original prompt",     # 原始提示词
            "image",               # 媒体类型
            api_name="/evaluate_media_complete"
        )

        # 方法2: 单独功能调用
        diagnosis = client.predict(
            "media.mp4", "prompt", "video",
            api_name="/analyze_mismatches"
        )

        enhanced = client.predict(
            "media.mp4", "prompt", diagnosis, "video",
            api_name="/get_enhanced_prompt"
        )

        negative = client.predict(
            "media.mp4", "prompt", diagnosis, "video",
            api_name="/get_negative_prompt"
        )
        ```

        ### 可用API端点：
        - `/analyze_mismatches` - 分析不匹配
        - `/get_enhanced_prompt` - 获取增强提示词
        - `/get_negative_prompt` - 获取负面提示词
        - `/evaluate_media_complete` - 完整评估 (推荐使用)
        - `/evaluate_media_two_stage` - 两步评估
        - `/evaluate_media_one_stage` - 一步评估

        ### 媒体类型参数：
        - `"video"` - 视频文件
        - `"image"` - 图片文件

        ### 返回格式：
        完整评估返回JSON格式：
        ```json
        {
            "enhanced_prompt": "增强的提示词",
            "negative_prompt": "负面提示词",
            "diagnosis": "诊断结果",
            "media_type": "video/image",
            "success": true
        }
        ```
        """)


def find_free_port(start_port=7860, max_attempts=10):
    """查找可用端口"""
    for port in range(start_port, start_port + max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('0.0.0.0', port))
            sock.close()
            return port
        except OSError:
            continue
    raise RuntimeError(f"无法找到可用端口 ({start_port}-{start_port+max_attempts})")

if __name__ == "__main__":
    try:
        port = find_free_port(16660)
        print(f"使用端口: {port}")
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True,
            debug=True
        )
    except Exception as e:
        print(f"启动失败: {e}")