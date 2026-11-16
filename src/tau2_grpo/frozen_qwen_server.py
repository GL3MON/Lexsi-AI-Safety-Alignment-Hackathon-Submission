"""Frozen Transformers Qwen endpoint for native Tau2 customer, judge and summary."""
import argparse,threading

def main():
 import uvicorn
 from fastapi import FastAPI
 from tau2_grpo.rl_config import load_settings
 from tau2_grpo.rl_post20_eval import LocalFrozenClient
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--port',type=int,default=8004);a=p.parse_args();client=LocalFrozenClient(load_settings(a.config));lock=threading.Lock();app=FastAPI()
 @app.get('/v1/models')
 def models():return dict(object='list',data=[dict(id='qwen3.5-4b',object='model')])
 @app.post('/v1/chat/completions')
 def create(body:dict):
  with lock:
   try:return client.create(**body).model_dump()
   finally:client.torch.cuda.empty_cache()
 uvicorn.run(app,host='127.0.0.1',port=a.port,log_level='warning')
if __name__=='__main__':main()
