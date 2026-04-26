from transformers import AutoConfig
c = AutoConfig.from_pretrained(r'D:\deploy\modules\modles\Qwen3.5-0.8B', trust_remote_code=True)
print('All attributes:')
for attr in dir(c):
    if not attr.startswith('_'):
        try:
            val = getattr(c, attr)
            if not callable(val):
                print(f'  {attr}: {val}')
        except:
            pass
