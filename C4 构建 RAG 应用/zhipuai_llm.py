from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from zhipuai import ZhipuAI

# 继承自 langchain_core.language_models.llms.LLM
class ZhipuAILLM(LLM):
    # def __init__(self,model: str ="glm-4" ,temperature: float=0.2,api_key: str=None):
    #     super().__init__()
    # 默认选用 glm-4 模型
    model : str ="glm-4"
    # 温度系数
    temperature : float=0.2
    # API_Key
    api_key : str=None
    ## 为什么没有__init__呢？
    ## 下面定义的是一个_call方法
    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
    ## 这里_call是类的函数，从外部传进来的参数怎么给_call的？
    ## 自己尝试写一个类进行封装，类有方法和性质
        def gen_glm_params(prompt):
            '''
            构造 GLM 模型请求参数 messages

            请求参数：
                prompt: 对应的用户提示词
            '''
            messages = [{"role": "user", "content": prompt}]
            return messages
        
        client = ZhipuAI(
            api_key=self.api_key
        )
     
        messages = gen_glm_params(prompt)
        response = client.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature = self.temperature
        )

        if len(response.choices) > 0:
            return response.choices[0].message.content
        return "generate answer error"


    # 首先定义一个返回默认参数的方法
    @property
    def _default_params(self) -> Dict[str, Any]:
        """获取调用Ennie API的默认参数。"""
        normal_params = {
            "temperature": self.temperature,
            "model_type":self.model
            }
        # print(type(self.model_kwargs))
        return {**normal_params}

    @property
    def _llm_type(self) -> str:
        return "Zhipu"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}
