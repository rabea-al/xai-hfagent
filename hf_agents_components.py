from xai_components.base import InArg, OutArg, InCompArg, BaseComponent, Component, xai_component

from transformers import OpenAiAgent
from transformers import Tool

from io import BytesIO
from PIL import Image
import tempfile

@xai_component
class HfAgentMakeTool(Component):
    run_tool: BaseComponent
    
    name: InCompArg[str]
    description: InCompArg[str]
    output_ref: InCompArg[str]
    
    tool_ref: OutArg[Tool]
    input_ref: OutArg[str]
    
    def execute(self, ctx) -> None:
        other_self = self
        
        class CustomTool(Tool):
            name = other_self.name.value
            description = other_self.description.value
            inputs = ["text"]
            output = ["text"]
            
            def __call__(self, prompt):
                
                other_self.input_ref.value = prompt
                next = other_self.run_tool
                while next:
                    next = next.do(ctx)
                return other_self.output_ref.value
            
        self.tool_ref.value = CustomTool()
                
    
@xai_component
class HfAgentInit(Component):
    agent_type: InCompArg[str]
    
    tools: InArg[list]
    

    def execute(self, ctx) -> None:
        if self.agent_type.value == 'openai':
            agent = OpenAiAgent(model="text-davinci-003", additional_tools=self.tools.value)
            ctx['hf_agent'] = agent


@xai_component
class HfAgentRun(Component):
    prompt: InCompArg[str]
    document: InArg[any]
    response_text: OutArg[str]
    response_file: OutArg[str]
    
    
    def execute(self, ctx) -> None:
        agent = ctx['hf_agent']
        
        if self.document.value:
            if isinstance(self.document.value, bytes):
                image_file = BytesIO(self.document.value)
                self.document.value = Image.open(image_file)
            ret = agent.run(self.prompt.value, document=self.document.value)
        else:
            ret = agent.run(self.prompt.value)
        
        if isinstance(ret, str):
            self.response_text.value = ret
        elif isinstance(ret, Image.Image):
            f = tempfile.NamedTemporaryFile()
            ret.save(f.name + '.png')
            self.response_file.value = f.name + '.png'

@xai_component
class HfReadImage(Component):
    file_path: InCompArg[str]
    out_image: OutArg[Image.Image]
    
    def execute(self, ctx) -> None:
        self.out_image.value = Image.open(self.file_path.value)
