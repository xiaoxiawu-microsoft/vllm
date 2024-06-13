from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "DeepSpeed is a",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)
llm = LLM(
    model="/home/wenxh/repo/vllm/DeepSpeedExamples/benchmarks/inference/mii/huggingface-checkpoint-229000",
    trust_remote_code=True,
    quantization="fp8",
    dtype="bfloat16",
    tensor_parallel_size=1,
    max_model_len=4096,
)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    out= output.outputs[0].text
    print (prompt)
    print (out)
    print ('-------------')