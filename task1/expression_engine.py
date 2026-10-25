"""Standalone numerical expression evaluator with training-only transforms."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ARITY={"log":1,"sqrt_abs":1,"min_max":1,"reciprocal":1,
       "+":2,"-":2,"*":2,"division":2,"mod_column":2}


class FrameEnvironment:
    def __init__(self,frames,columns):
        self.frames={k:v[columns].reset_index(drop=True) for k,v in frames.items()}
        self.columns=columns
        self.features=[str(i) for i in range(len(columns))]
        self.ops=list(ARITY);self.max_length=31

    def parse(self,expression):
        tokens=expression.split(",");depth=0
        if len(tokens)>self.max_length:raise ValueError("Expression exceeds feature grammar budget")
        for token in tokens:
            if token in self.features:depth+=1
            elif token in ARITY:
                if depth<ARITY[token]:raise ValueError("Not enough operands: "+token)
                depth+=1-ARITY[token]
            else:raise ValueError("Unknown feature or operator: "+token)
        if depth!=1:raise ValueError("Expression must produce exactly one value")
        return tuple(tokens)

    @staticmethod
    def _calculate(operator,values):
        first=values[0]
        if operator=="log":return np.log(first.abs().where(first!=0,1e-5))
        if operator=="sqrt_abs":return first.abs().pow(.5)
        if operator=="reciprocal":return 1/first.where(first!=0,1e-5)
        second=values[1]
        if operator=="+":return first+second
        if operator=="-":return first-second
        if operator=="*":return first*second
        if operator=="division":return first/second.where(second!=0,1e-5)
        if operator=="mod_column":return first.mod(second).fillna(0)
        raise ValueError("Unknown numerical operator")

    def generate(self,expression,split):
        stack=[]
        for token in self.parse(expression):
            if token in self.features:
                column=int(token)
                stack.append((self.frames["train"].iloc[:,column],self.frames[split].iloc[:,column]))
                continue
            width=ARITY[token];arguments=stack[-width:];del stack[-width:]
            training=[a for a,b in arguments];evaluation=[b for a,b in arguments]
            if token=="min_max":
                scaler=MinMaxScaler().fit(training[0].to_numpy().reshape(-1,1))
                left=pd.Series(scaler.transform(training[0].to_numpy().reshape(-1,1)).ravel(),index=training[0].index)
                right=left if split=="train" else pd.Series(scaler.transform(evaluation[0].to_numpy().reshape(-1,1)).ravel(),index=evaluation[0].index)
            else:
                left=self._calculate(token,training)
                right=left if split=="train" else self._calculate(token,evaluation)
            stack.append((left,right))
        return stack[0][1].replace([np.inf,-np.inf],np.nan).fillna(-1).clip(-1e8,1e8).astype(np.float32)

    def random_expression(self):
        def sample(depth,force_operator=False):
            if depth>=2 or (not force_operator and np.random.random()<.5):return [str(np.random.choice(self.features))]
            operator=str(np.random.choice(self.ops));tokens=[]
            for _ in range(ARITY[operator]):tokens.extend(sample(depth+1))
            return tokens+[operator]
        return ",".join(sample(0,force_operator=True))
