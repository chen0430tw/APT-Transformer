"""
Simple WebUI test to diagnose button click issues
"""
import gradio as gr

print(f"Gradio version: {gr.__version__}")

def test_click():
    return "✅ Button clicked successfully!"

with gr.Blocks() as app:
    gr.Markdown("# 简单测试 - 按钮是否可点击")

    output = gr.Textbox(label="输出", value="等待点击...")
    btn = gr.Button("点击我测试", variant="primary")

    btn.click(fn=test_click, outputs=output)

    gr.Markdown("如果点击按钮后上面的文本框显示'✅ Button clicked successfully!'，说明Gradio工作正常。")

if __name__ == "__main__":
    app.launch(server_port=7861, share=False)
