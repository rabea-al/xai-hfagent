from xai_components.base import InArg, OutArg, InCompArg,secret, BaseComponent, Component, xai_component
from transformers import Tool, AutoModelForCausalLM, AutoTokenizer, pipeline
from io import BytesIO
from PIL import Image
import torch

import os

hf_token = os.getenv("HF_TOKEN")

@xai_component
class HfAgentMakeTool(Component):
    run_tool: BaseComponent
    
    name: InCompArg[str]
    description: InCompArg[str]
    input_ref: InArg[str]
    tool_ref: OutArg[Tool]
    output_ref: OutArg[str]
    
    def execute(self, ctx) -> None:
        other_self = self
        
        class CustomTool(Tool):
            name = other_self.name.value 
            description = other_self.description.value 
            inputs = {"text": {"type": "string", "description": "The input question to answer."}}
            output_type = "string"

            def forward(self, text):
                response = ctx['hf_agent'](  
                    text,
                    max_length=50, 
                    temperature=0.5,   
                    top_p=0.9,
                    truncation=True
                )

                if isinstance(response, list) and len(response) > 0 and 'generated_text' in response[0]:
                    return response[0]['generated_text']
                else:
                    return "No valid response received."

        self.tool_ref.value = CustomTool()

                

@xai_component
class HfAgentInit(Component):
    agent_type: InCompArg[str]  
    tools: InArg[Tool]
    token: InArg[secret]  # إدخال التوكن كـ `InArg[secret]`
    from_env: InArg[bool]  # لتحديد المصدر

    def execute(self, ctx) -> None:
        # تحديد مصدر `token` حسب قيمة `from_env`
        hf_token = os.getenv("HF_TOKEN") if self.from_env.value else self.token.value

        tools = self.tools.value if isinstance(self.tools.value, list) else [self.tools.value]
        if not all(isinstance(tool, Tool) for tool in tools):
            raise ValueError("All items in tools must be instances of Tool.")

        model_name = self.agent_type.value or "gpt2"
        
        # تحميل tokenizer والنموذج باستخدام `hf_token`
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

        llm_engine = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, 
            device=0 if torch.cuda.is_available() else -1, 
            pad_token_id=tokenizer.eos_token_id
        )
        
        ctx['hf_agent'] = llm_engine  # تخزين llm_engine في السياق

@xai_component
class HfAgentRun(Component):
    prompt: InCompArg[str]
    document: InArg[any]
    response_text: OutArg[str]
    response_file: OutArg[str]
    
    def execute(self, ctx) -> None:
        agent = ctx['hf_agent']
        
        prompt_text = f" '{self.prompt.value}'"

        if self.document.value:
            if isinstance(self.document.value, bytes):
                image_file = BytesIO(self.document.value)
                self.document.value = Image.open(image_file)
            
            ret = agent(
                prompt_text,
                max_length=50,
                temperature=0.7,
                top_p=0.9,
                truncation=True 
            )

            if isinstance(ret, list) and len(ret) > 0 and 'generated_text' in ret[0]:
                self.response_text.value = ret[0]['generated_text']
                print("Response Text:", self.response_text.value) 
            else:
                print("No valid response received.")
        else:
            ret = agent(
                prompt_text,
                max_length=50,
                temperature=0.7,
                top_p=0.9,
                truncation=True
            )
            if isinstance(ret, list) and len(ret) > 0 and 'generated_text' in ret[0]:
                self.response_text.value = ret[0]['generated_text']
                print("Response Text:", self.response_text.value)
            else:
                print("No valid response received.")

@xai_component
class HfReadImage(Component):
    file_path: InCompArg[str]
    out_image: OutArg[Image.Image]

    def execute(self, ctx) -> None:
        self.out_image.value = Image.open(self.file_path.value) 
