"""Restricted expression frontends and local LLM proposal generation."""
import ast
import json
from pathlib import Path
from task1.expression_engine import FrameEnvironment,ARITY


def extract_expression(text):
    """Read a col-ID/comma RPN expression, skipping explanations/fences."""
    for line in text.splitlines():
        line=line.strip().strip("`")
        start=line.find("col-")
        if start<0: continue
        result=[]
        for token in line[start:].split(","):
            token=token.strip()
            if token.startswith("col-"): token=token[4:]
            if token=="/": token="division"
            result.append(token)
        expression=",".join(result)
        if len(result)>1:
            return expression
    raise ValueError("No comma-separated RPN feature in model response")


def extract_infix_expression(text,columns=24):
    """Convert a restricted mathematical expression to validated RPN tokens.

    The parsed AST is inspected, never compiled or evaluated. Constants,
    attributes, indexing, comprehensions and arbitrary function calls are rejected.
    """
    binary={ast.Add:"+",ast.Sub:"-",ast.Mult:"*",ast.Div:"division"}
    unary={"log","sqrt_abs","min_max","reciprocal"}
    def visit(node):
        if isinstance(node,ast.Name) and node.id.startswith("col_"):
            number=node.id[4:]
            if number.isdigit() and 0<=int(number)<columns:
                return [str(int(number))]
        if isinstance(node,ast.BinOp) and type(node.op) in binary:
            return visit(node.left)+visit(node.right)+[binary[type(node.op)]]
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords:
            name=node.func.id
            if name in unary and len(node.args)==1:
                return visit(node.args[0])+[name]
            if name=="mod_column" and len(node.args)==2:
                return visit(node.args[0])+visit(node.args[1])+[name]
        raise ValueError("Unsupported expression syntax")
    errors=[]
    for line in text.splitlines():
        line=line.strip().strip("`").strip()
        if not line or line.startswith("#"): continue
        try:
            tokens=visit(ast.parse(line,mode="eval").body)
            if len(tokens)<2: raise ValueError("Expression must transform a feature")
            if sum(t in OPERATORS_NAMES for t in tokens)>5:
                raise ValueError("More than five operators")
            return ",".join(tokens)
        except (SyntaxError,ValueError,RecursionError) as exc:
            errors.append(str(exc))
    raise ValueError("No valid restricted infix expression: "+"; ".join(errors[-2:]))


OPERATORS_NAMES=set(ARITY)


class LocalProposer:
    def __init__(self,model_name,revision=None,seed=42):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        self.torch=torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        local=Path(model_name).is_dir()
        if local and (Path(model_name)/"source.json").exists():
            source=json.loads((Path(model_name)/"source.json").read_text())
            revision=revision or source["revision"]
        self.tokenizer=AutoTokenizer.from_pretrained(model_name,revision=revision,local_files_only=local)
        self.model=AutoModelForCausalLM.from_pretrained(model_name,revision=revision,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,local_files_only=local)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.provenance={"model":model_name,"revision":getattr(self.model.config,"_commit_hash",None) or revision,
                         "backend":"local transformers", "seed":seed,"max_new_tokens":128, "temperature":.7}

    def __call__(self,prompt):
        text=self.tokenizer.apply_chat_template([
            {"role":"system","content":"You are a feature engineering researcher. Follow the requested expression syntax exactly."},
            {"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
        inputs=self.tokenizer(text,return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            outputs=self.model.generate(**inputs,max_new_tokens=128,do_sample=True,temperature=.7,top_p=.9,
                pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0,inputs["input_ids"].shape[-1]:],skip_special_tokens=True)
