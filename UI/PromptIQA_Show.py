import gradio as gr
import json
from PromptIQA import run_promptIQA

class Main_ui():
    def __init__(self) -> None: # stage_Koniq
        self.json_path = 'example.json'
        self.promptiqa = run_promptIQA.PromptIQA()
    
    def load_example(self):
        with open(self.json_path, 'r') as f:
            info = json.load(f)
        
        examples = []
        remarks = []
        
        for exp in info:
            ISPP = exp['ISPP']
            Image = exp['Image']
            Remark = exp['Remark']
            
            image, score = [], []
            for ISP_Image, ISP_Score in ISPP:
                image.append(ISP_Image)
                score.append(float(ISP_Score))
            example = [item for pair in zip(image, score) for item in pair]
                
            example.append(Image[0])
            example.append(float(Image[1]))
            
            examples.append(example)
            remarks.append(Remark)
        
        return examples, remarks
            
    def load_demo(self):
        def get_iq_score(*args):
            ISPP_I, ISPP_S, image = args[:10], args[10:20], args[-1]
            res = self.promptiqa.run(ISPP_I, ISPP_S, image)
            return res
            
        image_components = []
        score_components = []

        with gr.Blocks() as demo:
            gr.Markdown("# PromptIQA: Boosting the Performance and Generalization for No-Reference Image Quality Assessment via Prompts")
            gr.Markdown("## 1. Upload the Image-Score Pairs Prompts")
            
            ISP_idx = 1
            for row_num in [10]:
                with gr.Row():
                    for i in range(row_num):
                        with gr.Column(scale=1):
                            ISP_Image = gr.Image(label=f'Image {ISP_idx}', width=448, height=448)
                            ISP_Score = gr.Slider(0, 1, label=f"Score {ISP_idx}")
                            ISP_idx += 1
                            
                            image_components.append(ISP_Image)
                            score_components.append(ISP_Score)
                gr.Markdown("---------------------------------------")
            
            gr.Markdown("## 2. Upload the image to be evaluated.")
            with gr.Row():
                Image_To_Be_Evaluated = gr.Image(label=f'Image To Be Evaluated.', width=512, height=512)
                with gr.Column():
                    quality_score = gr.Textbox(label='Predicted Quality Score')
                    pre_button = gr.Button("Get Quality Score")
                    
            pre_button.click(get_iq_score, inputs=image_components + score_components + [Image_To_Be_Evaluated], outputs=[quality_score])
        
            examples, remarks = self.load_example()
            
            gr.Markdown("<font color=red size=72>Examples</font>")
            for idx, (remark, example) in enumerate(zip(remarks, examples)):
                gr.Markdown(f"### Example{idx + 1}: {remark}")
                gr.Examples(examples=[example], inputs=[item for pair in zip(image_components, score_components) for item in pair] + [Image_To_Be_Evaluated, quality_score])


        return demo